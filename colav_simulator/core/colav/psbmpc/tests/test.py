import colav_simulator.common.config_parsing as cp
import colav_simulator.core.colav.colav_interface as cI
import colav_simulator.core.guidances as gI
import numpy as np

from colav_simulator.core.colav.psbmpc import PSBMPCInterface as psbmpcI
from colav_simulator.core.colav.im import IMInterface as imI



if __name__ == "__main__":

    # to_dict() and from_dict() functionality tests
    test_params_1 = psbmpcI.PSBMPCParams()
    print("\nImport test: ")
    print("PSBMPCParams class object from normal setup procedure: ", psbmpcI.PSBMPCParams, "\n")
    
    # note that all parameters has a range of valid input values
    n_M = 1
    n_do_ps = 1
    p_step_opt = 1
    p_step_do = 1
    p_step_grounding = 1
    T = 60
    dt = 60
    t_ts = 60
    d_safe = 60
    d_do_relevant = 60
    d_so_relevant = 60
    K_coll = 60
    T_coll = 60
    kappa_SO = 60
    kappa_GW = 60
    kappa_RA = 60 
    K_u = 60
    K_du = 60
    K_chi = 2
    K_dchi = 2
    K_e = 60
    G_1 = 60
    G_2 = 60
    G_3 = 60
    G_4 = 60
    epsilon_rdp = 60
    u_offsets = [np.array([1, 0.5, 1, -0.5, -1], dtype = np.float64)]
    chi_offsets = [np.array([
        -90, -80, -70, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0, -5.0, 0.0,
        -5.0, 0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0
    ], dtype = np.float64)]
    cpe_method = psbmpcI.CPEMethod.CE
    pred_method = psbmpcI.PredictionMethod.Linear
    guidance_method = psbmpcI.GuidanceMethod.LOS

    psbmpc_params_dict = {
        "n_M" : n_M,
        "n_do_ps" : n_do_ps,
        "p_step_opt" : p_step_opt,
        "p_step_do" : p_step_do,
        "p_step_grounding" : p_step_grounding,
        "T" : T,
        "dt" : dt,
        "t_ts" : t_ts,
        "d_safe" : d_safe,
        "d_do_relevant" : d_do_relevant,
        "d_so_relevant" : d_so_relevant,
        "K_coll" : K_coll,
        "T_coll" : T_coll,
        "kappa_SO" : kappa_SO,
        "kappa_GW" : kappa_GW,
        "kappa_RA" : kappa_RA,
        "K_u" : K_u,
        "K_du" : K_du,
        "K_chi" : K_chi,
        "K_dchi" : K_dchi,
        "K_e" : K_e,
        "G_1" : G_1,
        "G_2" : G_2,
        "G_3" : G_3,
        "G_4" : G_4,
        "epsilon_rdp" : epsilon_rdp,
        "u_offsets" : u_offsets,
        "chi_offsets" : chi_offsets,
        "CPE_method" : cpe_method,
        "prediction_method" : pred_method,
        "guidance_method" : guidance_method
    }

    test_params_2 = psbmpcI.PSBMPCParams.from_dict(psbmpc_params_dict)
    print("from_dict() classmethod and to_dict() test:")
    print("PSBMPCParams class object made from dictionary: ", test_params_2)
    print("from_dict() monkey patch works as intended iff: ")
    print("psbmpc_params_dict.items() == psbmpc_params_dict")
    if dict(list(psbmpc_params_dict.items())[:-5]) == dict(list(test_params_2.to_dict().items())[:-5]):
        if dict(list(psbmpc_params_dict.items())[-3:]) == dict(list(test_params_2.to_dict().items())[-3:]):
            if np.array_equal(test_params_2.to_dict()["u_offsets"], psbmpc_params_dict["u_offsets"]) == True:
                if np.allclose(test_params_2.to_dict()["chi_offsets"], psbmpc_params_dict["chi_offsets"]) == True:
                    print("Check met. The dictionaries are equal to one another.")
                    print("psbmpc_params_dict :", psbmpc_params_dict.items())

    # init

    # psbmpc params
    n_M = 1
    n_do_ps = 1
    p_step_opt = 1
    p_step_do = 1
    p_step_grounding = 1
    T = 60
    dt = 60
    t_ts = 60
    d_safe = 60
    d_do_relevant = 60
    d_so_relevant = 60
    K_coll = 60
    T_coll = 60
    kappa_SO = 60
    kappa_GW = 60
    kappa_RA = 60 
    K_u = 60
    K_du = 60
    K_chi = 2
    K_dchi = 2
    K_e = 60
    G_1 = 60
    G_2 = 60
    G_3 = 60
    G_4 = 60
    epsilon_rdp = 60
    u_offsets = [np.array([1, 0.5, 1, -0.5, -1], dtype = np.float64)]
    chi_offsets = [np.array([
        -90, -80, -70, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0, -5.0, 0.0,
        -5.0, 0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0
    ], dtype = np.float64)]
    cpe_method = psbmpcI.CPEMethod.CE
    pred_method = psbmpcI.PredictionMethod.Linear
    guidance_method = psbmpcI.GuidanceMethod.LOS

    # ownship params
    l = 12
    w = 4
    T_U = 1.77
    T_chi = 0.97
    R_a = 20
    LOS_LD = 250
    LOS_K_i = 0.25
    active_wpt = 2

    # cpe params
    cpe_method = psbmpcI.CPEMethod.CE # or psbmpcI.CPEMethod.MCSKF4D
    n_CE = 350
    n_MCSKF = 750
    alpha_n = 0.912
    gate = 11.237521
    rho = 0.896
    max_it = 12
    q = 0.01231
    r = 0.1

    # im params
    num_ships = 3

    # params used by IMParams
    number_of_network_evaluation_samples = 150000
    max_number_of_obstacles = num_ships - 1 # must be set to num_ships - 1 (else segmantation fault)
    time_into_trajectory = 100
    starting_distance = 10020
    starting_cpa_distance = 15020
    expanding_dbn_min_time_s = 22
    expanding_dbn_max_time_s = 302
    expanding_dbn_min_course_change_rad = 0.222222
    expanding_dbn_min_speed_change_m_s = 1.52
    ample_time_s_mu = 202
    ample_time_s_sigma = 102
    ample_time_s_max = 1002
    ample_time_s_n_bins = 32 
    ample_time_s_minimal_accepted_by_ownship = 22
    safe_distance_m_mu = 202
    safe_distance_m_sigma = 32
    safe_distance_m_max = 802
    safe_distance_m_n_bins = 32
    risk_distance_m_mu = 1502
    risk_distance_m_sigma = 252
    risk_distance_m_max = 2502
    risk_distance_m_n_bins = 32
    risk_distance_front_m_mu = 1502
    risk_distance_front_m_sigma = 252
    risk_distance_front_m_max = 2502
    risk_distance_front_m_n_bins = 32
    safe_distance_midpoint_m_mu = 602
    safe_distance_midpoint_m_sigma = 22
    safe_distance_midpoint_m_max = 2502
    safe_distance_midpoint_m_n_bins = 32
    safe_distance_front_m_mu = 102
    safe_distance_front_m_sigma = 52
    safe_distance_front_m_max = 1002
    safe_distance_front_m_n_bins = 32
    change_in_course_rad_minimal_change_since_init_state = 0.222222
    change_in_course_rad_minimal_change_since_last_state = 0.062
    change_in_speed_m_s_minimal_change = 1.52
    colregs_situation_borders_rad_HO_uncertainty_start = 2.792
    colregs_situation_borders_rad_HO_start = 2.962
    colregs_situation_borders_rad_HO_stop = -2.962
    colregs_situation_borders_rad_HO_uncertainty_stop = -2.792
    colregs_situation_borders_rad_OT_uncertainty_start = 1.742
    colregs_situation_borders_rad_OT_start = 2.182
    colregs_situation_borders_rad_OT_stop = -2.182
    colregs_situation_borders_rad_OT_uncertainty_stop = -1.742
    set_startpoint_min_time_cpa = 62
    time_step_removal_min_timesteps_in_state_history = 2
    time_step_removal_unmodeled_behaviour_threshold = 0.2
    time_step_removal_time_cpa_threshold = 52
    ignoring_safety_probability = 0.2
    colregs_compliance_probability = 0.92
    good_seamanship_probability = 0.92
    unmodeled_behaviour = 0.00002
    priority_probability_lower = 0.052
    priority_probability_similar = 0.92
    priority_probability_higher = 0.052

    psbmpc_params_dict = {
        "n_M" : n_M,
        "n_do_ps" : n_do_ps,
        "p_step_opt" : p_step_opt,
        "p_step_do" : p_step_do,
        "p_step_grounding" : p_step_grounding,
        "T" : T,
        "dt" : dt,
        "t_ts" : t_ts,
        "d_safe": d_safe,
        "d_do_relevant" : d_do_relevant,
        "d_so_relevant" : d_so_relevant,
        "K_coll" : K_coll,
        "T_coll" : T_coll,
        "kappa_SO" : kappa_SO,
        "kappa_GW" : kappa_GW,
        "kappa_RA" : kappa_RA,
        "K_u" : K_u,
        "K_du" : K_du,
        "K_chi" : K_chi,
        "K_dchi" : K_dchi,
        "K_e" : K_e,
        "G_1" : G_1,
        "G_2" : G_2,
        "G_3" : G_3,
        "G_4" : G_4,
        "epsilon_rdp" : epsilon_rdp,
        "u_offsets" : u_offsets,
        "chi_offsets" : chi_offsets,
        "CPE_method" : cpe_method,
        "prediction_method" : pred_method,
        "guidance_method" : guidance_method
    }

    expanding_dbn = {
        "min_time_s" : expanding_dbn_min_time_s,
        "max_time_s" : expanding_dbn_max_time_s,
        "min_course_change_rad" : expanding_dbn_min_course_change_rad,
        "min_speed_change_m_s" : expanding_dbn_min_speed_change_m_s
    }

    ample_time_s = {
        "mu" : ample_time_s_mu,
        "sigma" : ample_time_s_sigma,
        "max" : ample_time_s_max,
        "n_bins" : ample_time_s_n_bins,
        "minimal_accepted_by_ownship" : ample_time_s_minimal_accepted_by_ownship
    }   

    safe_distance_m = {
        "mu" : safe_distance_m_mu,
        "sigma" : safe_distance_m_sigma,
        "max" : safe_distance_m_max,
        "n_bins" : safe_distance_m_n_bins
    }

    risk_distance_m = {
        "mu" : risk_distance_m_mu,
        "sigma" : risk_distance_m_sigma,
        "max" : risk_distance_m_max,
        "n_bins" : risk_distance_m_n_bins
    }

    risk_distance_front_m = {
        "mu" : risk_distance_front_m_mu,
        "sigma" : risk_distance_front_m_sigma,
        "max" : risk_distance_front_m_max,
        "n_bins" : risk_distance_front_m_n_bins
    }

    safe_distance_midpoint_m = {
        "mu" : safe_distance_midpoint_m_mu,
        "sigma" : safe_distance_midpoint_m_sigma,
        "max" : safe_distance_midpoint_m_max,
        "n_bins" : safe_distance_midpoint_m_n_bins
    }

    safe_distance_front_m = {
        "mu" : safe_distance_front_m_mu,
        "sigma" : safe_distance_front_m_sigma,
        "max" : safe_distance_front_m_max,
        "n_bins" : safe_distance_front_m_n_bins
    }

    change_in_course_rad = {
        "minimal_change_since_init_state" : change_in_course_rad_minimal_change_since_init_state,
        "minimal_change_since_last_state" : change_in_course_rad_minimal_change_since_last_state
    }

    change_in_speed_m_s = {
        "minimal_change" : change_in_speed_m_s_minimal_change
    }

    colregs_situation_borders_rad = {
        "HO_uncertainty_start" : colregs_situation_borders_rad_HO_uncertainty_start,
        "HO_start" : colregs_situation_borders_rad_HO_start,
        "HO_stop" : colregs_situation_borders_rad_HO_stop,
        "HO_uncertainty_stop" : colregs_situation_borders_rad_HO_uncertainty_stop,
        "OT_uncertainty_start" : colregs_situation_borders_rad_OT_uncertainty_start,
        "OT_start" : colregs_situation_borders_rad_OT_start,
        "OT_stop" : colregs_situation_borders_rad_OT_stop,
        "OT_uncertainty_stop" : colregs_situation_borders_rad_OT_uncertainty_stop
    }

    set_startpoint = {
        "min_time_cpa" : set_startpoint_min_time_cpa,
    }

    time_step_removal = {
        "min_timesteps_in_state_history" : time_step_removal_min_timesteps_in_state_history,
        "unmodeled_behaviour_threshold" : time_step_removal_unmodeled_behaviour_threshold,
        "time_cpa_threshold" : time_step_removal_time_cpa_threshold
    }

    priority_probability = {
        "lower" : priority_probability_lower,
        "similar" : priority_probability_similar,
        "higher" : priority_probability_higher
    }

    im_params = {
        "number_of_network_evaluation_samples" : number_of_network_evaluation_samples,
        "max_number_of_obstacles" : max_number_of_obstacles,
        "time_into_trajectory" : time_into_trajectory,
        "starting_distance" : starting_distance,
        "starting_cpa_distance" : starting_cpa_distance,
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
        "ignoring_safety_probability" : ignoring_safety_probability,
        "colregs_compliance_probability" : colregs_compliance_probability,
        "good_seamanship_probability" : good_seamanship_probability,
        "unmodeled_behaviour" : unmodeled_behaviour,
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
        "length" : l,
        "width" : w,
        "T_U" : T_U,
        "T_chi" : T_chi,
        "R_a" : R_a,
        "LOS_LD" : LOS_LD,
        "LOS_K_i" : LOS_K_i,
        "active_waypoint" : active_wpt
    }

    cpe_params = {
        "CPE_method" : cpe_method,
        "n_CE" : n_CE,
        "n_MCSKF" : n_MCSKF,
        "alpha_n" : alpha_n,
        "gate" : gate,
        "rho" : rho,
        "max_it" : max_it,
        "q" : q,
        "r" : r
    }

    config_dict = {
        "name": "PSBMPC",
        "layer1" : {"psbmpc_params" : psbmpc_params_dict},
        "layer2" : {"psbmpc_ownship" : ownship_params},
        "layer3" : {"psbmpc_cpe" : cpe_params},
        "layer4" : {"im" : im_params},
        "layer5" : {"los" : los_params}
    }

    # tests
    config = cI.LayerConfig()
    config.psbmpc = cp.convert_settings_dict_to_paramsclass(psbmpcI.PSBMPCParams, config_dict["layer1"]["psbmpc_params"])

    read_psbmpc_params = config.psbmpc.to_dict()
    print("\n\nTesting Config.to_dict(): ", read_psbmpc_params)

    config = cI.Config.from_dict(config_dict)
    print("\n\nConfig: ", config)

    print("\n\nTesting Config.to_dict(): ", config.to_dict())

    print("\n\nTesting that config.layer1.psbmpc_params is of type PSBMPCParams: ", config.layer1.psbmpc_params)

    print("\n\nTesting that config.layer2.psbmpc_ownship is of type KinematicShip: ", config.layer2.psbmpc_ownship)

    print("\n\nTesting that config.layer3.psbmpc_cpe is of type CPE: ", config.layer3.psbmpc_cpe)

    print("\n\nTesting that config.layer4.im is of type IM: ", config.layer4.im)

    print("\n\nTesting that config.layer5.los is LOSGuidanceParams: ", config.layer5.los)

    _psbmpc_params = config.layer1.psbmpc_params
    print("\n\n PSBMPCParams object: ", _psbmpc_params)
    print("Also testing from_dict() and to_dict() monkey patched to PSBMPCParams class:")
    print("n_M", _psbmpc_params.get_par_int(0), "\n",
        "n_do_ps", _psbmpc_params.get_par_int(1), "\n",
        "p_step_opt", _psbmpc_params.get_par_int(2), "\n",
        "p_step_do", _psbmpc_params.get_par_int(3), "\n",
        "p_step_grounding", _psbmpc_params.get_par_int(4), "\n",
        "T", _psbmpc_params.get_par_double(0), "\n",
        "dt", _psbmpc_params.get_par_double(1), "\n",
        "t_ts", _psbmpc_params.get_par_double(2), "\n",
        "d_safe", _psbmpc_params.get_par_double(3), "\n",
        "d_do_relevant", _psbmpc_params.get_par_double(4), "\n",
        "d_so_relevant", _psbmpc_params.get_par_double(5), "\n",
        "K_coll", _psbmpc_params.get_par_double(6), "\n",
        "T_coll", _psbmpc_params.get_par_double(7), "\n",
        "kappa_SO", _psbmpc_params.get_par_double(8), "\n",
        "kappa_GW", _psbmpc_params.get_par_double(9), "\n",
        "kappa_RA", _psbmpc_params.get_par_double(10), "\n",
        "K_u", _psbmpc_params.get_par_double(11), "\n",
        "K_du", _psbmpc_params.get_par_double(12), "\n",
        "K_chi", _psbmpc_params.get_par_double(13), "\n",
        "K_dchi", _psbmpc_params.get_par_double(14), "\n",
        "K_e", _psbmpc_params.get_par_double(15), "\n",
        "G_1", _psbmpc_params.get_par_double(16), "\n",
        "G_2", _psbmpc_params.get_par_double(17), "\n",
        "G_3", _psbmpc_params.get_par_double(18), "\n",
        "G_4", _psbmpc_params.get_par_double(19), "\n",
        "epsilon_rdp", _psbmpc_params.get_par_double(20), "\n",
        "u_offsets", _psbmpc_params.get_par_vector(0), "\n",
        "chi_offsets", _psbmpc_params.get_par_vector(1), "\n",
        "CPE_method", _psbmpc_params.get_cpe_method(), "\n",
        "prediction_method", _psbmpc_params.get_prediction_method(), "\n",
        "guidance_method", _psbmpc_params.get_guidance_method())

    _psbmpc_ownship = psbmpcI.KinematicShip(config.layer2.psbmpc_ownship)
    print("\n\nTesting KinematicShip constructor from another KinematicShip instance")
    print("Also testing from_dict() and to_dict() monkey patched to KinematicShip class:")
    print("KinematicShip obj.: ", _psbmpc_ownship, "\n", 
        "length: ", _psbmpc_ownship.get_length(), "\n", 
        "width: ", _psbmpc_ownship.get_width(), "\n",
        "T_U: ", _psbmpc_ownship.get_T_U(), "\n", 
        "T_chi: ", _psbmpc_ownship.get_T_chi(), "\n", 
        "R_a: ", _psbmpc_ownship.get_R_a(), "\n",
        "LOS_LD: ", _psbmpc_ownship.get_LOS_LD(), "\n",
        "LOS_K_i: ", _psbmpc_ownship.get_LOS_K_i(), "\n", 
        "active_waypoint: ", _psbmpc_ownship.get_wp_counter(), "\n")

    _psbmpc_cpe = psbmpcI.CPE(config.layer3.psbmpc_cpe)
    print("\n\nTesting CPE constructor from CPE_Method")
    print("Also testing from_dict() and to_dict() monkey patched to CPE class:")
    print("CPE obj.: ", _psbmpc_cpe, "\n", 
        "CPE_method: ", _psbmpc_cpe.get_cpe_method(), "\n", 
        "n_CE: ", _psbmpc_cpe.get_n_CE(), "\n",
        "n_MCSKF: ", _psbmpc_cpe.get_n_MCSKF(), "\n", 
        "alpha_n: ", _psbmpc_cpe.get_alpha_n(), "\n", 
        "gate: ", _psbmpc_cpe.get_gate(), "\n",
        "rho: ", _psbmpc_cpe.get_rho(), "\n",
        "max_it: ", _psbmpc_cpe.get_max_it(), "\n", 
        "q: ", _psbmpc_cpe.get_q(), "\n",
        "r: ", _psbmpc_cpe.get_r(), "\n")

    _im_params = imI.IMParams.copy_parameters_to_new_instance(config.layer4.im)
    print('\n\nTesting "IntentionModelParameters constructor"; "copy_parameters_to_new_instance"\n',
        "Also testing from_dict() and to_dict() monkey patched to IntentionModelParameters:\n",
        "number_of_network_evaluation_samples: ", _im_params.number_of_network_evaluation_samples, "\n",
        "max_number_of_obstacles: ", _im_params.max_number_of_obstacles, "\n",
        "time_into_trajectory: ", _im_params.time_into_trajectory, "\n",
        "starting_distance: ", _im_params.starting_distance, "\n",
        "starting_cpa_distance: ", _im_params.starting_cpa_distance, "\n",
        "expanding_dbn.min_time_s: ", _im_params.expanding_dbn.min_time_s, "\n",
        "expanding_dbn.max_time_s: ", _im_params.expanding_dbn.max_time_s, "\n",
        "expanding_dbn.min_course_change_rad: ", _im_params.expanding_dbn.min_course_change_rad, "\n",
        "expanding_dbn.min_speed_change_m_s: ", _im_params.expanding_dbn.min_speed_change_m_s, "\n",
        "ample_time_s.mu: ", _im_params.ample_time_s.mu, "\n",
        "ample_time_s.sigma: ", _im_params.ample_time_s.sigma, "\n",
        "ample_time_s.max: ", _im_params.ample_time_s.max, "\n",
        "ample_time_s.n_bins: ", _im_params.ample_time_s.n_bins, "\n",
        "ample_time_s.minimal_accepted_by_ownship: ", _im_params.ample_time_s.minimal_accepted_by_ownship, "\n",
        "safe_distance_m.mu: ", _im_params.safe_distance_m.mu, "\n",
        "safe_distance_m.sigma: ", _im_params.safe_distance_m.sigma, "\n",
        "safe_distance_m.max: ", _im_params.safe_distance_m.max, "\n",
        "safe_distance_m.n_bins: ", _im_params.safe_distance_m.n_bins, "\n",
        "risk_distance_m.mu: ", _im_params.risk_distance_m.mu, "\n",
        "risk_distance_m.sigma: ", _im_params.risk_distance_m.sigma, "\n",
        "risk_distance_m.max: ", _im_params.risk_distance_m.max, "\n",
        "risk_distance_m.n_bins: ", _im_params.risk_distance_m.n_bins, "\n",
        "risk_distance_front_m.mu: ", _im_params.risk_distance_front_m.mu, "\n",
        "risk_distance_front_m.sigma: ", _im_params.risk_distance_front_m.sigma, "\n",
        "risk_distance_front_m.max: ", _im_params.risk_distance_front_m.max, "\n",
        "risk_distance_front_m.n_bins: ", _im_params.risk_distance_front_m.n_bins, "\n",
        "safe_distance_midpoint_m.mu: ", _im_params.safe_distance_midpoint_m.mu, "\n",
        "safe_distance_midpoint_m.sigma: ", _im_params.safe_distance_midpoint_m.sigma, "\n",
        "safe_distance_midpoint_m.max: ", _im_params.safe_distance_midpoint_m.max, "\n",
        "safe_distance_midpoint_m.n_bins: ", _im_params.safe_distance_midpoint_m.n_bins , "\n",
        "safe_distance_front_m.mu: ", _im_params.safe_distance_front_m.mu, "\n",
        "safe_distance_front_m.sigma: ", _im_params.safe_distance_front_m.sigma, "\n",
        "safe_distance_front_m.max: ", _im_params.safe_distance_front_m.max, "\n",
        "safe_distance_front_m.n_bins: ", _im_params.safe_distance_front_m.n_bins , "\n",
        "change_in_course_rad.minimal_change_since_init_state: ", _im_params.change_in_course_rad.minimal_change_since_init_state, "\n",
        "change_in_course_rad.minimal_change_since_last_state: ", _im_params.change_in_course_rad.minimal_change_since_last_state, "\n",
        "change_in_speed_m_s.minimal_change: ", _im_params.change_in_speed_m_s.minimal_change, "\n",
        "colregs_situation_borders_rad.HO_uncertainty_start: ", _im_params.colregs_situation_borders_rad.HO_uncertainty_start, "\n",
        "colregs_situation_borders_rad.HO_start: ", _im_params.colregs_situation_borders_rad.HO_start, "\n",
        "colregs_situation_borders_rad.HO_stop: ", _im_params.colregs_situation_borders_rad.HO_stop, "\n",
        "colregs_situation_borders_rad.HO_uncertainty_stop: ", _im_params.colregs_situation_borders_rad.HO_uncertainty_stop, "\n",
        "colregs_situation_borders_rad.OT_uncertainty_start: ", _im_params.colregs_situation_borders_rad.OT_uncertainty_start, "\n",
        "colregs_situation_borders_rad.OT_start: ", _im_params.colregs_situation_borders_rad.OT_start, "\n",
        "colregs_situation_borders_rad.OT_stop: ", _im_params.colregs_situation_borders_rad.OT_stop, "\n",
        "colregs_situation_borders_rad.OT_uncertainty_stop: ", _im_params.colregs_situation_borders_rad.OT_uncertainty_stop, "\n",
        "set_startpoint.min_time_cpa: ", _im_params.set_startpoint.min_time_cpa, "\n",
        "time_step_removal.min_timesteps_in_state_history: ", _im_params.time_step_removal.min_timesteps_in_state_history, "\n",
        "time_step_removal.unmodeled_behaviour_threshold: ", _im_params.time_step_removal.unmodeled_behaviour_threshold, "\n",
        "time_step_removal.time_cpa_threshold: ", _im_params.time_step_removal.time_cpa_threshold, "\n",
        "ignoring_safety_probability: ", _im_params.ignoring_safety_probability, "\n",
        "colregs_compliance_probability: ", _im_params.colregs_compliance_probability, "\n",
        "good_seamanship_probability: ", _im_params.good_seamanship_probability, "\n",
        "unmodeled_behaviour: ", _im_params.unmodeled_behaviour, "\n",
        'priority_probability["lower"]: ', _im_params.priority_probability["lower"], "\n",
        'priority_probability["similar"]: ', _im_params.priority_probability["similar"], "\n",
        'priority_probability["higher"]: ', _im_params.priority_probability["higher"])

    _los = config.layer5.los
    losguidanceparams = gI.LOSGuidanceParams.from_dict(los_params)
    los_to_dict_test = gI.LOSGuidanceParams.to_dict(losguidanceparams)
    print("\n\nTesting LOS from_dict():\n",
        "pass_angle_threshold: ", _los.pass_angle_threshold, "\n",
        "R_a: ", _los.R_a, "\n",
        "K_p: ", _los.K_p, "\n",
        "K_i: ", _los.K_i, "\n",
        "e_int_max: ", _los.e_int_max, "\n",
        "\nTesting LOS to_dict():\n",
        "pass_angle_threshold ", losguidanceparams.pass_angle_threshold, "\n",
        "R_a ", losguidanceparams.R_a, "\n",
        "K_p ", losguidanceparams.K_p, "\n",
        "K_i ", losguidanceparams.K_i, "\n",
        "e_int_max ", losguidanceparams.e_int_max)
