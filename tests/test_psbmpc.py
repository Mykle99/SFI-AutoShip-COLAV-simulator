"""_summary_ For testing simulator with psbmpc.
For this to work, the submodules in the path ../colav_simulator/colav_simulator/core/colav/cpp_to_py_interfaces/
need to be built. See docs on the submodules
"""

from colav_simulator.simulator import Simulator
from colav_simulator.core.colav.psbmpc import PSBMPCInterface as psbmpcI 
from colav_simulator.core.colav.im import IMInterface as imI

import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.core.guidances as guidance
import numpy as np



if __name__ == "__main__":

    # chose which test to perform
    print("Chose which test to perform by setting the bools:\n"
        '"test_manual", "test_dict" and "test_from_yaml"')
    test_manual = False
    test_dict = False
    test_from_yaml = True

    if test_manual:

        # init
        colav_config = ci.Config()
        colav_config.layer1 = ci.LayerConfig()
        colav_config.layer2 = ci.LayerConfig()
        colav_config.layer3 = ci.LayerConfig()
        colav_config.layer4 = ci.LayerConfig()
        colav_config.layer5 = ci.LayerConfig()
        colav_builder = ci.COLAVBuilder()
        simulator = Simulator()

        # cpe method
        cpe_ce = psbmpcI.CPEMethod.CE
        cpe_mcskf = psbmpcI.CPEMethod.MCSKF4D

        # name and layers
        num_ships = 2 # 1 ownship + 1 obstacle ship = 2 ships (IM only works with 2 ships currently)
        colav_config.name = ci.COLAVType.PSBMPC # "PSBMPC"
        colav_config.layer1.psbmpc_params = psbmpcI.PSBMPCParams()
        colav_config.layer2.psbmpc_ownship = psbmpcI.KinematicShip()
        colav_config.layer3.psbmpc_cpe = psbmpcI.CPE(cpe_ce)
        colav_config.layer4.im = imI.IMParams.default_parameters(num_ships)
        colav_config.layer5.los = guidance.LOSGuidanceParams()

        # running the simulation
        output = simulator.run(ownship_colav_system = colav_builder.construct_colav(config = colav_config))
        print("Simulation completed.")
    
    elif test_dict:

        # setting up the dicts
        # n_bins param in im_params has to match the Bayesian network
        # max_number_of_obstacles in im_params has to be 1

        # cpe_method set here since it is used both in psbmpc_params and cpe_params
        cpe_method = psbmpcI.CPEMethod.CE
        cpe_other_method = psbmpcI.CPEMethod.MCSKF4D

        # setting all setable parameters of the COLAV system in the config_dict
        psbmpc_params_dict = {
            "n_M" : 1,
            "n_do_ps" : 5,
            "p_step_opt" : 10,
            "p_step_do" : 2,
            "p_step_grounding" : 4,
            "T" : 120,
            "dt" : 1.0,
            "t_ts" : 60,
            "d_safe": 4.5,
            "d_do_relevant" : 500,
            "d_so_relevant" : 150,
            "K_coll" : 25.0,
            "T_coll" : 100,
            "kappa_SO" : 20.0,
            "kappa_GW" : 25.0,
            "kappa_RA" : 15.0,
            "K_u" : 6.3,
            "K_du" : 3.6,
            "K_chi" : 1.8,
            "K_dchi" : 1.3,
            "K_e" : 0.0005,
            "G_1" : 100.0,
            "G_2" : 5.0,
            "G_3" : 1.4,
            "G_4" : 1.0,
            "epsilon_rdp" : 2.0,
            "u_offsets" : [np.array([1.0, 0.5, 0.0])],
            "chi_offsets" : [np.array([-60.0, -45.0, -30.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0])],
            "CPE_method" : cpe_method,
            "prediction_method" : psbmpcI.PredictionMethod.ERK1,
            "guidance_method" : psbmpcI.GuidanceMethod.LOS
        }

        expanding_dbn = {
            "min_time_s" : 20,
            "max_time_s" : 300,
            "min_course_change_rad" : 0.2617994,
            "min_speed_change_m_s" : 1.5
        }

        ample_time_s = {
            "mu" : 200,
            "sigma" : 100,
            "max" : 1000,
            "n_bins" : 30,
            "minimal_accepted_by_ownship" : 20
        }   

        safe_distance_m = {
            "mu" : 200,
            "sigma" : 30,
            "max" : 800,
            "n_bins" : 30
        }

        risk_distance_m = {
            "mu" : 1500,
            "sigma" : 250,
            "max" : 2500,
            "n_bins" : 30
        }

        risk_distance_front_m = {
            "mu" : 1500,
            "sigma" : 250,
            "max" : 2500,
            "n_bins" : 30
        }

        safe_distance_midpoint_m = {
            "mu" : 600,
            "sigma" : 20,
            "max" : 2500,
            "n_bins" : 30
        }

        safe_distance_front_m = {
            "mu" : 100,
            "sigma" : 50,
            "max" : 1000,
            "n_bins" : 30
        }

        change_in_course_rad = {
            "minimal_change_since_init_state" : 0.2617994,
            "minimal_change_since_last_state" : 0.06
        }

        change_in_speed_m_s = {
            "minimal_change" : 1.5
        }

        colregs_situation_borders_rad = {
            "HO_uncertainty_start" : 2.79,
            "HO_start" : 2.96,
            "HO_stop" : -2.96,
            "HO_uncertainty_stop" : -2.79,
            "OT_uncertainty_start" : 1.74,
            "OT_start" : 2.18,
            "OT_stop" : -2.18,
            "OT_uncertainty_stop" : -1.74
        }

        set_startpoint = {
            "min_time_cpa" : 60,
        }

        time_step_removal = {
            "min_timesteps_in_state_history" : 7,
            "unmodeled_behaviour_threshold" : 0.1,
            "time_cpa_threshold" : 50
        }

        priority_probability = {
            "lower" : 0.05,
            "similar" : 0.90,
            "higher" : 0.05
        }

        im_params = {
            "number_of_network_evaluation_samples" : 100000,
            "max_number_of_obstacles" : 1,
            "time_into_trajectory" : 0,
            "starting_distance" : 10000,
            "starting_cpa_distance" : 15000,
            "expanding_dbn" : expanding_dbn,
            "ample_time_s" : ample_time_s,
            "safe_distance_m" : safe_distance_m,
            "risk_distance_m" : risk_distance_m,
            "risk_distance_front_m" : risk_distance_front_m,
            "safe_distance_midpoint_m" : safe_distance_midpoint_m,
            "safe_distance_front_m" : safe_distance_front_m,
            "change_in_course_rad" : change_in_course_rad,
            "change_in_speed_m_s" : change_in_speed_m_s,
            "colregs_situation_borders_rad" : colregs_situation_borders_rad,
            "set_startpoint" : set_startpoint,
            "time_step_removal" : time_step_removal,
            "ignoring_safety_probability" : 0,
            "colregs_compliance_probability" : 0.99,
            "good_seamanship_probability" : 0.99,
            "unmodeled_behaviour" : 0.00001,
            "priority_probability" : priority_probability
        }

        los_params = {
            "pass_angle_threshold": 60.0,
            "R_a": 8.0,
            "K_p": 0.06,
            "K_i": 0.002,
            "e_int_max": 30.0
        }

        ownship_params = {
            "length" : 5.0,
            "width" : 3.0,
            "T_U" : 1.44,
            "T_chi" : 0.92,
            "R_a" : 5.0,
            "LOS_LD" : 66.0,
            "LOS_K_i" : 0.0,
            "active_waypoint" : 0
        }

        cpe_params = {
            "CPE_method" : cpe_method,
            "n_CE" : 500,
            "n_MCSKF" : 500,
            "alpha_n" : 0.9,
            "gate" : 11.618285980628054,
            "rho" : 0.9,
            "max_it" : 6,
            "q" : 8e-4,
            "r" : 0.001
        }

        config_dict = {
            "name": "PSBMPC",
            "layer1" : {"psbmpc_params" : psbmpc_params_dict},
            "layer2" : {"psbmpc_ownship" : ownship_params},
            "layer3" : {"psbmpc_cpe" : cpe_params},
            "layer4" : {"im" : im_params},
            "layer5" : {"los" : los_params}
        }

        # init
        colav_config = ci.Config.from_dict(config_dict)
        colav_builder = ci.COLAVBuilder()
        simulator = Simulator()

        # running the simulation
        output = simulator.run(ownship_colav_system = colav_builder.construct_colav(config = colav_config))
        print("Simulation completed.")

    elif test_from_yaml:
        
        # not completed due to psbmpcI and np in params
        print("head_on_psbmpc.yaml in the scenarios folder can be used "
            "with this test. The file can be chosen in the simulator.yaml file "
            "in the config folder. (In simulator.yaml; set simulator: "
            'scenario files: ["head_on_psbmpc.yaml])'
        )

        # init
        simulator = Simulator()

        # running the simulation
        output = simulator.run()
        print("Simulation completed.")
        