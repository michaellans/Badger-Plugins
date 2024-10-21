import time
import numpy as np
from badger import environment
from badger.errors import BadgerInterfaceChannelError, BadgerNoInterfaceError


class Environment(environment.Environment):

    name = "sphere_epics"
    variables = {
        "SPHERE:X1": [-1, 1],
        "SPHERE:X2": [-1, 1],  # hysteresis
    }
    observables = ["SPHERE:F"]

    timeout: float = 3  # timeout in seconds
    overshoot_fraction: float = 0.1
    trim_delay: float = 1

    def get_variables(self, variable_names: list[str]) -> dict:
        if self.interface is None:
            raise BadgerNoInterfaceError

        # Fetch the interface until all values are not None
        time_start = time.time()
        variable_outputs = self.interface.get_values(variable_names)
        while None in variable_outputs.values():
            time.sleep(0.1 * np.random.rand())
            time_elapsed = time.time() - time_start
            if time_elapsed > self.timeout:
                raise BadgerInterfaceChannelError

            variable_outputs = self.interface.get_values(variable_names)

        return variable_outputs

    def set_variables(self, variable_inputs: dict[str, float]):
        if self.interface is None:
            raise BadgerNoInterfaceError

        # if overshooting to mitigate hysteresis is active
        if self.overshoot_fraction != 0.0:
            hysteresis_elements = [
                name for name in variable_inputs if self.is_hysteresis(name)
            ]

            # Early return if no hysteresis elements
            if not hysteresis_elements:
                self.interface.set_values(variable_inputs)
                return

            # get the current values for each
            current_vals = self.interface.get_values(hysteresis_elements)

            # if we want to make negative changes then first overshoot in the
            # negative direction
            negative_changes = []
            for name in hysteresis_elements:
                if variable_inputs[name] < current_vals[name]:
                    negative_changes.append(name)

            if negative_changes:
                # if there are any negative changes then overshoot in the neg direction
                bounds = self._get_bounds(negative_changes)
                overshoot_values = {
                    name: variable_inputs[name]
                    - (bounds[name][1] - bounds[name][0]) * self.overshoot_fraction
                    for name in negative_changes
                }

                # set the overshoot values
                self.interface.set_values(overshoot_values)
                time.sleep(self.trim_delay)

        self.interface.set_values(variable_inputs)

    def is_hysteresis(self, variable_name: str) -> bool:
        if "X2" in variable_name:
            return True

        return False

    def get_bounds(self, variable_names: list[str]) -> dict:
        return {name: [-1, 1] for name in variable_names}
