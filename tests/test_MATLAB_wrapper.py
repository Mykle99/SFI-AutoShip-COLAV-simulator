"""_summary_ For admiting params from MATLAB and returning simulation data to MATLAB
"""

from colav_simulator.simulator import Simulator
from colav_simulator.scenario_management import ScenarioGenerator
from colav_simulator.core.colav.psbmpc import PSBMPCInterface as psbmpcI 
from colav_simulator.core.colav.im import IMInterface as imI
import colav_simulator.common.map_functions as map_functions

import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.core.guidances as guidance
import numpy as np
import pandas as pd
import time



# Path to datafiles
path = '/mnt/c/Users/JeMyk/OneDrive/Dokumenter/NTNU/Industriell kybernetikk/3 - Haust 2023/TTK32 - Autonome systemer - Risiko og tillitsbygging/Prosjekt/'

# Read parameters from MATLAB
df = pd.read_csv(path + 'sim_params_from_MATLAB.txt', header = None)
param_1 = df.values[0][0] # value_d_safe
param_2 = df.values[0][1] # value_G1

# Real simulation data

# # init
colav_config = ci.Config()
colav_config.layer1 = ci.LayerConfig()
colav_config.layer2 = ci.LayerConfig()
colav_config.layer3 = ci.LayerConfig()
colav_builder = ci.COLAVBuilder()
simulator = Simulator()

# # name
colav_config.name = ci.COLAVType.PSBMPC # "PSBMPC"

# # layer 1 # #
# This is where the params that are read from MATLAB must be applied
# For the test of the static obstacle cost; G_1 through G_4 and d_safe
# can be varied.
colav_config.layer1.psbmpc = psbmpcI.PSBMPCParamsWrapper()
colav_config.layer1.psbmpc.ownshipparams = psbmpcI.KinematicShip()
colav_config.layer1.psbmpc.cpeparams = psbmpcI.CPE(psbmpcI.CPEMethod.CE)
colav_config.layer1.psbmpc.psbmpcparams = psbmpcI.PSBMPCParams()

# Static obstacle parameters
i_d_safe = 3          # Do not change
i_G1 = 16             # Do not change
# i_G2 = 17           # Do not change
i_G3 = 18             # Do not change
i_G4 = 19             # Do not change
# i_d_so_concider = 5 # Do not change

value_d_safe = param_1  # This should come from MATLAB, standard value: 4.5
value_G1 = param_2      # This should come from MATLAB, standard value: 100.0
# value_G2 = 5.0        # Standard value: 5.0
value_G3 = 0.18         # Maybe interesting to set from MATLAB, standard value: 1.4    # Tune val. : 0.14
value_G4 = 0.13         # Standard value: 1.0                                          # Tune val. : 0.13
# d_so_concider = 150.0 # Standard value: 150.0

# Trajectory deviation parameters
# Maybe interesting to set some of these from MATLAB,
# since, when there are no dynamic obstacles, the algorithm
# only weighs the cost of trajectory deviation and static obstacles 
# i_K_u = 11     # Do not chnage
# i_K_du = 12    # Do not chnage
# i_K_chi = 13   # Do not chnage
# i_K_dchi = 14  # Do not chnage
# i_K_e = 15     # Do not chnage

# value_K_u = 6.3     # Standard value: 6.3
# value_K_du = 3.6    # Standard value: 3.6
# value_K_chi = 1.8   # Standard value: 1.8
# value_K_dchi = 1.3  # Standard value: 1.3
# value_K_e = 0.0005  # Standard value: 0.0005

# Setting the parameters in the PSBMPC
colav_config.layer1.psbmpc.psbmpcparams.set_par_double(index = i_d_safe, value = value_d_safe)
colav_config.layer1.psbmpc.psbmpcparams.set_par_double(index = i_G1, value = value_G1)
# colav_config.layer1.psbmpc.psbmpcparams.set_par_double(index = i_G2, value = value_G2)
colav_config.layer1.psbmpc.psbmpcparams.set_par_double(index = i_G3, value = value_G3)
colav_config.layer1.psbmpc.psbmpcparams.set_par_double(index = i_G4, value = value_G4)
# colav_config.layer1.psbmpc.psbmpcparams.set_par_double(index = i_d_so_concider, value = d_so_concider)
# colav_config.layer1.psbmpc.psbmpcparams.set_par_double(index = i_K_u, value = value_K_u)
# colav_config.layer1.psbmpc.psbmpcparams.set_par_double(index = i_K_du, value = value_K_du)
# colav_config.layer1.psbmpc.psbmpcparams.set_par_double(index = i_K_chi, value = value_K_chi)
# colav_config.layer1.psbmpc.psbmpcparams.set_par_double(index = i_K_dchi, value = value_K_dchi)
# colav_config.layer1.psbmpc.psbmpcparams.set_par_double(index = i_K_e, value = value_K_e)

# # layer 2
# num_ships = 2
colav_config.layer2.im = imI.IMParams.default_parameters(2)

# # layer 3
colav_config.layer3.los = guidance.LOSGuidanceParams()

# # running the simulation
scenario_generator = ScenarioGenerator()
scenario_data_list = scenario_generator.generate_configured_scenarios()
output = simulator.run(
    scenario_data_list, ownship_colav_system = colav_builder.construct_colav(
        config = colav_config
    )
)

# Obtaining simulation data
t_data = np.array(output[0]["episode_simdata_list"][0]["vessel_data"][0].timestamps)
os_xy_data = np.array(output[0]["episode_simdata_list"][0]["vessel_data"][0].xy)
os_x_data, os_y_data = os_xy_data[0], os_xy_data[1]

min_depth = 5 # 5 m min depth for the vessel
min_distance_to_land = 10 # 10 m land/shore buffer
grounding_hazards = map_functions.extract_grounding_hazards_from_entire_enc(
    min_depth, min_distance_to_land, simulator.enc
)

d_data = []
for i in range(len(os_x_data)):
    d_data.append(map_functions.min_distance_to_hazards(
        grounding_hazards, os_x_data[i], os_y_data[i])
    )
    #if i%50 == 0:
    #    simulator.enc.start_display()
    #    simulator.enc.add_vessels((0, int(os_x_data[i]), int(os_y_data[i]), 30, "black"))
    #    try:
    #        simulator.enc.draw_circle([int(os_x_data[i]), int(os_y_data[i])], d_data[i], "magenta", thickness = 0, fill = True, alpha = 0.6)
    #    except:
    #        print("The ship grounded.")
    #    try:
    #        for multipolygon in grounding_hazards:
    #            for polygon in multipolygon.geoms:
    #                simulator.enc.draw_polygon(polygon, color = "red", alpha = 0.6)
    #    except:
    #        try:
    #            for polygon in grounding_hazards.geoms:
    #                simulator.enc.draw_polygon(polygon, color = "red", alpha = 0.6)
    #        except:
    #            simulator.enc.draw_polygon(grounding_hazards, color = "red", alpha = 0.6)
    #    simulator.enc.show_display()

# Save simulation data to be read by MATLAB
np.savetxt(path + 'sim_data_from_python.csv', [t_data, d_data], delimiter = ',')

# 31513161252054 matches the number which is checked for in MATLAB
with open(path + 'python_sim_complete.txt', 'w') as file:
    file.write("31513161252054")
time.sleep(1)
