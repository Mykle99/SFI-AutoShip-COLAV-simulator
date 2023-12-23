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
import pickle

legend_size = 10  # legend size
fig_size = [25, 13]  # figure1 size in cm
dpi_value = 150  # figure dpi value

if __name__ == "__main__":

    scenario_file = dp.scenarios / "VIMMJIPDA.yaml"
    # scenario_file = dp.scenarios / "VIMMJIPDA_Multi_Target.yaml"
    scenario_generator = ScenarioGenerator()
    scenario_data = scenario_generator.generate(config_file=scenario_file)
    simulator = Simulator()
    simulator.toggle_liveplot_visibility(False)
    output = simulator.run([scenario_data])
    print("done")


    #Prints to understand the parts of the output:
    # print("Len (output)", len(output), "\n")
    # print("Type(output)", type(output), "\n")
    # print("Loop through el in outout")
    # for el in output:
    #     print("len(el(simdata))", len(el['episode_simdata_list']))
    #     print("type(el[simdata])", type(el['episode_simdata_list']), "\n")
    #     print("Loop through items in el[simdata]")
    #     # print("len(el['enc'])", len(el['enc']))
    #     for episode in el['episode_simdata_list']:
    #         # print(i)
    #         print("len(episode)", len(episode), "\n")
    #         print("type(episode)", type(episode),"\n")
    #         print("Print items in episode")
    #         for items in episode:
    #             print(items)
                

    # For printing vessel_data for Evaluator Tool
    # print(len(output[0]['episode_simdata_list'][0]['vessel_data']))

    # for el in output[0]['episode_simdata_list'][0]['vessel_data']:
    #     print(type(el))



    #For printing sim_data
    # for el in output[0]['episode_simdata_list'][0]['sim_data'].iloc[0,1]:
    #     print(el)
    #     print(output[0]['episode_simdata_list'][0]['sim_data'].iloc[0,1][el], "\n")
    # print(output[0]['episode_simdata_list'][0]['sim_data'].iloc[799,0]) # print row 0 for ship_0
    
    
    
    # print(output[0]['episode_simdata_list'][0]['ship_info']['Ship0'])  #For printing Ship_info

    
    # simulator.visualizer.save_live_plot_animation()
    
    # Code to attempt to save data to a folder and plot
    
    

    # folder_path = '/home/ragnarnw/Github_test/Plotting_SR_2023/'
    # folder_path_backup = '/home/ragnarnw/Github_test/Plotting_SR_2023/Backup/'
    # file_name = 'Test1_KF'
    # file_name_backup = 'Test1_KF_backup'

    # # 1000 funker ikke av en eller annen grunn, så holder meg til 500 enn så lenge
    # with open(folder_path + file_name, 'wb') as file:
    #     pickle.dump(output[0]['episode_simdata_list'], file)

    # with open(folder_path_backup + file_name_backup, 'wb') as file:
    #     pickle.dump(output[0]['episode_simdata_list'], file)

    # folder_path = '/home/ragnarnw/Github_test/Plotting_SR_2023/'
    # file_name_backup = 'Test1_termination_VIMMJIPDA'
    # with open(folder_path + file_name_backup, 'wb') as file:
    #     pickle.dump(output[0]['episode_simdata_list'], file)