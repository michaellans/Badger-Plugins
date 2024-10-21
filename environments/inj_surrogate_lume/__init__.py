from typing import Dict
import time
import torch
import numpy as np
from badger import environment
from badger.errors import BadgerInterfaceChannelError


class Environment(environment.Environment):

    name = 'inj_surrogate_lume'
    variables = {
        'CAMR:IN20:186:R_DIST': [210.00, 500.00],
        'Pulse_length': [1.81, 7.27],
        'FBCK:BCI0:1:CHRG_S': [0.25, 0.25],
        'SOLN:IN20:121:BACT': [0.37, 0.50],
        'QUAD:IN20:121:BACT': [-0.02, 0.02],
        'QUAD:IN20:122:BACT': [-0.02, 0.02],
        'ACCL:IN20:300:L0A_ADES': [58.00, 58.00],
        'ACCL:IN20:300:L0A_PDES': [-25.00, 10.00],
        'ACCL:IN20:400:L0B_ADES': [70.00, 70.00],
        'ACCL:IN20:400:L0B_PDES': [-25.00, 10.00],
        'QUAD:IN20:361:BACT': [-4.00, -1.00],
        'QUAD:IN20:371:BACT':  [1.00, 4.30],
        'QUAD:IN20:425:BACT': [-7.55, -1.00],
        'QUAD:IN20:441:BACT': [-1.00, 7.55],
        # 'QUAD:IN20:511:BACT': [-1.00, 7.55],  # make these only available through interface
        # 'QUAD:IN20:525:BACT': [-7.55, -1.00],
    }
    observables = [
        'OTRS:IN20:571:XRMS',
        'OTRS:IN20:571:YRMS',
        'sigma_z',
        'norm_emit_x',
        'norm_emit_y',
    ]

    _variables = {
        'CAMR:IN20:186:R_DIST': 424.00,
        'Pulse_length': 1.86,
        'FBCK:BCI0:1:CHRG_S': 0.25,
        'SOLN:IN20:121:BACT': 0.478,
        'QUAD:IN20:121:BACT': -0.0021,
        'QUAD:IN20:122:BACT': -0.00063,
        'ACCL:IN20:300:L0A_ADES': 58.00,
        'ACCL:IN20:300:L0A_PDES': -9.54,
        'ACCL:IN20:400:L0B_ADES': 70.0,
        'ACCL:IN20:400:L0B_PDES': 9.85,
        'QUAD:IN20:361:BACT': -2.00,
        'QUAD:IN20:371:BACT': 2.00,
        'QUAD:IN20:425:BACT': -1.00,
        'QUAD:IN20:441:BACT': -0.18,
        'QUAD:IN20:511:BACT': 2.85,
        'QUAD:IN20:525:BACT': -3.22,
    }
    _observations = {
        'OTRS:IN20:571:XRMS': None,
        'OTRS:IN20:571:YRMS': None,
        'sigma_z': None,
        'norm_emit_x': None,
        'norm_emit_y': None,
    }
    _model = None
    _modified = True
    waiting_time: float = 0

    def get_variables(self, variable_names):
        try:
            variable_outputs = {v: self._variables[v] for v in variable_names}
        except KeyError:
            raise BadgerInterfaceChannelError

        return variable_outputs

    def set_variables(self, variable_inputs: Dict[str, float]):
        for var, x in variable_inputs.items():
            if self._variables[var] != x:
                self._variables[var] = x
                self._modified = True

    def get_observables(self, observable_names):
        dt = self.waiting_time
        time.sleep(dt)

        if not self._modified:
            return {k: self._observations[k] for k in observable_names}

        # Lazy loading
        if self._model is None:
            self.load_model()

        self._modified = False

        # Predict with model
        model = self._model

        assert model is not None, 'Model failed to initialize!'

        # Set solenoid, SQ, CQ to values from optimization step
        x_in = list(self._variables.values())
        x_in = torch.as_tensor(np.array([x_in]))

        # Update predictions
        y_out = model.pred_sim_units(x_in)

        self._observations['OTRS:IN20:571:XRMS'] = y_out[0].detach().item()
        self._observations['OTRS:IN20:571:YRMS'] = y_out[1].detach().item()
        self._observations['sigma_z'] = y_out[2].detach().item()
        self._observations['norm_emit_x'] = y_out[3].detach().item()
        self._observations['norm_emit_y'] = y_out[4].detach().item()

        return {k: self._observations[k] for k in observable_names}

    def load_model(self):
        # Lazy importing
        from .injector_surrogate_quads import SurrogateNN

        self._model = model = SurrogateNN()
        torch_model = model.create_torch_model()
        model.create_torch_module(torch_model)

        self._model = model

    def get_bounds(self, variable_names):
        bounds = {}
        for variable_name in variable_names:
            lb, ub = variable_name + ".LB", variable_name + ".UB"
            bounds[variable_name] = [self.interface.get_value(lb), self.interface.get_value(ub)]
        return bounds
