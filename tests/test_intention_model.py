"""_summary_For testing simulator with intention model.
For this to work, the submodules in the path ../colav_simulator/colav_simulator/core/colav/cpp_to_py_interfaces/
need to be built. See docs on the submodules
"""

from colav_simulator.simulator import Simulator
from colav_simulator.core.colav.im import IMInterface as imI 

import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.core.guidances as guidance



if __name__ == "__main__":

    colav_config = ci.Config()
    colav_config.layer1 = ci.LayerConfig()
    colav_config.layer1.im = imI.IMParams.IntentionModelParameters()
    colav_config.layer2 = ci.LayerConfig()
    colav_config.layer2.los = guidance.LOSGuidanceParams()
    colav_config.name = ci.COLAVType.IM
    colav_builder = ci.COLAVBuilder()
    simulator = Simulator()

    # use plot_static MATLAB file, a file should be generated to the output folder after running the simulator
    output = simulator.run(ownship_colav_system = colav_builder.construct_colav(config= colav_config))
    print("done")
