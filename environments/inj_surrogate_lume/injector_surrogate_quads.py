import torch
import os
import numpy as np
from lume_model.utils import variables_from_yaml
from lume_model.models import TorchModel, TorchModule


class SurrogateNN:

    def __init__(self):
        self.pv_module = None

        # lume module instance
        self.lume_module = None

    def create_torch_model(self):
        _env_root = os.path.dirname(os.path.realpath(__file__))

        # load sim_to_nn transformers
        self.input_sim_to_nn = torch.load(os.path.join(_env_root, "model/input_sim_to_nn.pt"))
        self.output_sim_to_nn = torch.load(os.path.join(_env_root, "model/output_sim_to_nn.pt"))

        # load pv_to_sim transformers
        self.input_pv_to_sim = torch.load(os.path.join(_env_root, 'model/input_pv_to_sim.pt'))
        self.output_pv_to_sim = torch.load(os.path.join(_env_root, 'model/output_pv_to_sim.pt'))

        # load in- and output variable specification
        self.input_variables, self.output_variables = variables_from_yaml(
            os.path.join(_env_root, "model/pv_variables.yml")
        )

        # lume model
        self.model = os.path.join(_env_root, "model/model.pt")
        # pv lume model from yaml
        self.pv_model = os.path.join(_env_root, "model/pv_model.yml")

        try:
            # simply load from YAML file
            lume_model = TorchModel(self.pv_model)
        except:
            # create TorchModel
            lume_model = TorchModel(
                model=self.model,
                input_variables=self.input_variables,
                output_variables=self.output_variables,
                input_transformers=
                [
                    self.input_pv_to_sim,
                    self.input_sim_to_nn
                ],  # pv_to_sim before sim_to_nn
                output_transformers=
                [
                    self.output_sim_to_nn,
                    self.output_pv_to_sim
                ],  # sim_to_nn before pv_to_sim
            )
        return lume_model

    def create_torch_module(self, lume_model):
        _env_root = os.path.dirname(os.path.realpath(__file__))
        # pv lume module from yaml
        self.pv_module = (os.path.join(_env_root, "model/pv_module.yml"))
        try:
            # simply load from YAML file
            lume_module = TorchModule(self.pv_module)
        except:
            # wrap in TorchModule
            lume_module = TorchModule(
                model=lume_model,
                input_order=lume_model.input_names,
                output_order=lume_model.output_names,
            )

        self.lume_module = lume_module
        return lume_module

    def pred_sim_units(self, x, lume_module=None):

        if lume_module is None:
            lume_module = self.lume_module
        with torch.no_grad():
            predictions = lume_module(x)
        return predictions

    # functions to convert between sim and machine units for data
    # def sim_to_pv(self, sim_vals):
    #     return self.output_pv_to_sim.untransform(sim_vals)
    #
    # def pv_to_sim(self, pv_vals):
    #     pv_vals = torch.as_tensor(np.array([pv_vals]))
    #     return self.input_pv_to_sim.transform(pv_vals)
