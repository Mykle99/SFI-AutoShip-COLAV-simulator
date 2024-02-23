from colav_simulator.core.colav.im import IMInterface
from dataclasses import dataclass



@dataclass
class IMParamsWrapper:
    """Parameter wrapper class for the IMParams."""


# Monkey patching IMParamsWrapper class to the IMInterface Python module
IMInterface.IMParamsWrapper = IMParamsWrapper


# Monkey patch of to_dict() and from_dict() for the IMParamsWrapper class
def to_dict(self) -> dict:
    expanding_dbn = {
        "min_time_s" : self.expanding_dbn.min_time_s,
        "max_time_s" : self.expanding_dbn.max_time_s,
        "min_course_change_rad" : self.expanding_dbn.min_course_change_rad,
        "min_speed_change_m_s" : self.expanding_dbn.min_speed_change_m_s
    }

    ample_time_s = {
        "mu" : self.ample_time_s.mu,
        "sigma" : self.ample_time_s.sigma,
        "max" : self.ample_time_s.max,
        "n_bins" : self.ample_time_s.n_bins,
        "minimal_accepted_by_ownship" : self.ample_time_s.minimal_accepted_by_ownship
    }   

    safe_distance_m = {
        "mu" : self.safe_distance_m.mu,
        "sigma" : self.safe_distance_m.sigma,
        "max" : self.safe_distance_m.max,
        "n_bins" : self.safe_distance_m.n_bins
    }

    risk_distance_m = {
        "mu" : self.risk_distance_m.mu,
        "sigma" : self.risk_distance_m.sigma,
        "max" : self.risk_distance_m.max,
        "n_bins" : self.risk_distance_m.n_bins
    }

    risk_distance_front_m = {
        "mu" : self.risk_distance_front_m.mu,
        "sigma" : self.risk_distance_front_m.sigma,
        "max" : self.risk_distance_front_m.max,
        "n_bins" : self.risk_distance_front_m.n_bins
    }

    safe_distance_midpoint_m = {
        "mu" : self.safe_distance_midpoint_m.mu,
        "sigma" : self.safe_distance_midpoint_m.sigma,
        "max" : self.safe_distance_midpoint_m.max,
        "n_bins" : self.safe_distance_midpoint_m.n_bins
    }

    safe_distance_front_m = {
        "mu" : self.safe_distance_front_m.mu,
        "sigma" : self.safe_distance_front_m.sigma,
        "max" : self.safe_distance_front_m.max,
        "n_bins" : self.safe_distance_front_m.n_bins
    }

    change_in_course_rad = {
        "minimal_change_since_init_state" : self.change_in_course_rad.minimal_change_since_init_state,
        "minimal_change_since_last_state" : self.change_in_course_rad.minimal_change_since_last_state
    }

    change_in_speed_m_s = {
        "minimal_change" : self.change_in_speed_m_s.minimal_change
    }

    colregs_situation_borders_rad = {
        "HO_uncertainty_start" : self.colregs_situation_borders_rad.HO_uncertainty_start,
        "HO_start" : self.colregs_situation_borders_rad.HO_start,
        "HO_stop" : self.colregs_situation_borders_rad.HO_stop,
        "HO_uncertainty_stop" : self.colregs_situation_borders_rad.HO_uncertainty_stop,
        "OT_uncertainty_start" : self.colregs_situation_borders_rad.OT_uncertainty_start,
        "OT_start" : self.colregs_situation_borders_rad.OT_start,
        "OT_stop" : self.colregs_situation_borders_rad.OT_stop,
        "OT_uncertainty_stop" : self.colregs_situation_borders_rad.OT_uncertainty_stop
    }

    set_startpoint = {
        "min_time_cpa" : self.set_startpoint.min_time_cpa
    }

    time_step_removal = {
        "min_timesteps_in_state_history" : self.time_step_removal.min_timesteps_in_state_history,
        "unmodeled_behaviour_threshold" : self.time_step_removal.unmodeled_behaviour_threshold,
        "time_cpa_threshold" : self.time_step_removal.time_cpa_threshold
    }

    priority_probability = {
        "lower" : self.priority_probability["lower"],
        "similar" : self.priority_probability["similar"],
        "higher" : self.priority_probability["higher"]
    }

    output = {
        "number_of_network_evaluation_samples" : self.number_of_network_evaluation_samples,
        "max_number_of_obstacles" : self.max_number_of_obstacles,
        "time_into_trajectory" : self.time_into_trajectory,
        "starting_distance" : self.starting_distance,
        "starting_cpa_distance" : self.starting_cpa_distance,
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
        "ignoring_safety_probability" : self.ignoring_safety_probability,
        "colregs_compliance_probability" : self.colregs_compliance_probability,
        "good_seamanship_probability" : self.good_seamanship_probability,
        "unmodeled_behaviour" : self.unmodeled_behaviour,
        "priority_probability" : priority_probability
    }
    return output

@classmethod
def from_dict(cls, data: dict) -> IMInterface.IMParamsWrapper:
    num_ships = data["max_number_of_obstacles"] + 1
    imparams = IMInterface.IMParams.default_parameters(num_ships) # currently only 2 as input works
    imparams.number_of_network_evaluation_samples = data["number_of_network_evaluation_samples"]
    imparams.max_number_of_obstacles = data["max_number_of_obstacles"]
    imparams.time_into_trajectory = data["time_into_trajectory"]
    imparams.starting_distance = data["starting_distance"]
    imparams.starting_cpa_distance = data["starting_cpa_distance"]
    imparams.expanding_dbn.min_time_s = data["expanding_dbn"]["min_time_s"]
    imparams.expanding_dbn.max_time_s = data["expanding_dbn"]["max_time_s"]
    imparams.expanding_dbn.min_course_change_rad = data["expanding_dbn"]["min_course_change_rad"]
    imparams.expanding_dbn.min_speed_change_m_s = data["expanding_dbn"]["min_speed_change_m_s"]
    imparams.ample_time_s.mu = data["ample_time_s"]["mu"]
    imparams.ample_time_s.sigma = data["ample_time_s"]["sigma"]
    imparams.ample_time_s.max = data["ample_time_s"]["max"]
    imparams.ample_time_s.n_bins = data["ample_time_s"]["n_bins"]
    imparams.ample_time_s.minimal_accepted_by_ownship = data["ample_time_s"]["minimal_accepted_by_ownship"]
    imparams.safe_distance_m.mu = data["safe_distance_m"]["mu"]
    imparams.safe_distance_m.sigma = data["safe_distance_m"]["sigma"]
    imparams.safe_distance_m.max = data["safe_distance_m"]["max"]
    imparams.safe_distance_m.n_bins = data["safe_distance_m"]["n_bins"]
    imparams.risk_distance_m.mu = data["risk_distance_m"]["mu"]
    imparams.risk_distance_m.sigma = data["risk_distance_m"]["sigma"]
    imparams.risk_distance_m.max = data["risk_distance_m"]["max"]
    imparams.risk_distance_m.n_bins = data["risk_distance_m"]["n_bins"]
    imparams.risk_distance_front_m.mu = data["risk_distance_front_m"]["mu"]
    imparams.risk_distance_front_m.sigma = data["risk_distance_front_m"]["sigma"]
    imparams.risk_distance_front_m.max = data["risk_distance_front_m"]["max"]
    imparams.risk_distance_front_m.n_bins = data["risk_distance_front_m"]["n_bins"]
    imparams.safe_distance_midpoint_m.mu = data["safe_distance_midpoint_m"]["mu"]
    imparams.safe_distance_midpoint_m.sigma = data["safe_distance_midpoint_m"]["sigma"]
    imparams.safe_distance_midpoint_m.max = data["safe_distance_midpoint_m"]["max"]
    imparams.safe_distance_midpoint_m.n_bins = data["safe_distance_midpoint_m"]["n_bins"]
    imparams.safe_distance_front_m.mu = data["safe_distance_front_m"]["mu"]
    imparams.safe_distance_front_m.sigma = data["safe_distance_front_m"]["sigma"]
    imparams.safe_distance_front_m.max = data["safe_distance_front_m"]["max"]
    imparams.safe_distance_front_m.n_bins = data["safe_distance_front_m"]["n_bins"]
    imparams.change_in_course_rad.minimal_change_since_init_state = data["change_in_course_rad"]["minimal_change_since_init_state"]
    imparams.change_in_course_rad.minimal_change_since_last_state = data["change_in_course_rad"]["minimal_change_since_last_state"]
    imparams.change_in_speed_m_s.minimal_change = data["change_in_speed_m_s"]["minimal_change"]
    imparams.colregs_situation_borders_rad.HO_uncertainty_start = data["colregs_situation_borders_rad"]["HO_uncertainty_start"]
    imparams.colregs_situation_borders_rad.HO_start = data["colregs_situation_borders_rad"]["HO_start"]
    imparams.colregs_situation_borders_rad.HO_stop = data["colregs_situation_borders_rad"]["HO_stop"]
    imparams.colregs_situation_borders_rad.HO_uncertainty_stop = data["colregs_situation_borders_rad"]["HO_uncertainty_stop"]
    imparams.colregs_situation_borders_rad.OT_uncertainty_start = data["colregs_situation_borders_rad"]["OT_uncertainty_start"]
    imparams.colregs_situation_borders_rad.OT_start = data["colregs_situation_borders_rad"]["OT_start"]
    imparams.colregs_situation_borders_rad.OT_stop = data["colregs_situation_borders_rad"]["OT_stop"]
    imparams.colregs_situation_borders_rad.OT_uncertainty_stop = data["colregs_situation_borders_rad"]["OT_uncertainty_stop"]
    imparams.set_startpoint.min_time_cpa = data["set_startpoint"]["min_time_cpa"]
    imparams.time_step_removal.min_timesteps_in_state_history = data["time_step_removal"]["min_timesteps_in_state_history"]
    imparams.time_step_removal.unmodeled_behaviour_threshold = data["time_step_removal"]["unmodeled_behaviour_threshold"]
    imparams.time_step_removal.time_cpa_threshold = data["time_step_removal"]["time_cpa_threshold"]
    imparams.ignoring_safety_probability = data["ignoring_safety_probability"]
    imparams.colregs_compliance_probability = data["colregs_compliance_probability"]
    imparams.good_seamanship_probability = data["good_seamanship_probability"]
    imparams.unmodeled_behaviour = data["unmodeled_behaviour"]
    imparams.priority_probability["lower"] = data["priority_probability"]["lower"]
    imparams.priority_probability["similar"] = data["priority_probability"]["similar"]
    imparams.priority_probability["higher"] = data["priority_probability"]["higher"]
    return imparams


# Monkey patching to_dict() and from_dict() to the im_params class
IMInterface.IMParamsWrapper.to_dict = to_dict
IMInterface.IMParamsWrapper.from_dict = from_dict
