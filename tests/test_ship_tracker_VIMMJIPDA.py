import colav_simulator.common.math_functions as mf
import colav_simulator.core.controllers as controllers
import colav_simulator.core.guidances as guidances
import colav_simulator.core.models as models
import colav_simulator.core.sensing as sensorss
import colav_simulator.core.ship as ship
import colav_simulator.core.stochasticity as stochasticity
import colav_simulator.core.tracking.trackers as trackers
import numpy as np
from colav_simulator.scenario_management import ScenarioGenerator
from matplotlib import pyplot as plt
from colav_simulator.simulator import Simulator
import colav_simulator.common.paths as dp

legend_size = 10  # legend size
fig_size = [25, 13]  # figure1 size in cm
dpi_value = 150  # figure dpi value

if __name__ == "__main__":

    scenario_file = dp.scenarios / "VIMMJIPDA.yaml"
    scenario_generator = ScenarioGenerator()
    scenario_data = scenario_generator.generate(config_file=scenario_file)
    simulator = Simulator()
    simulator.toggle_liveplot_visibility(True)
    output = simulator.run([scenario_data])
    print("done")
