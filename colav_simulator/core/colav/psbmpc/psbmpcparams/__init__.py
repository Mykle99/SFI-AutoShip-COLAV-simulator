from colav_simulator.core.colav.psbmpc import PSBMPCInterface
import numpy as np



# Monkey patch of to_dict() and from_dict() for the PSBMPCParams class

# Read from all getter methods, which themselves reads from the
# C++ class member variables, and then writes to the output dict
def to_dict(self) -> dict:
    output = {
        "n_M" : self.get_par_int(0),
        "n_do_ps" : self.get_par_int(1),
        "p_step_opt" : self.get_par_int(2),
        "p_step_do" : self.get_par_int(3),
        "p_step_grounding" : self.get_par_int(4),
        "T" : self.get_par_double(0),
        "dt" : self.get_par_double(1),
        "t_ts" : self.get_par_double(2),
        "d_safe" : self.get_par_double(3),
        "d_do_relevant" : self.get_par_double(4),
        "d_so_relevant" : self.get_par_double(5),
        "K_coll" : self.get_par_double(6),
        "T_coll" : self.get_par_double(7),
        "kappa_SO" : self.get_par_double(8),
        "kappa_GW" : self.get_par_double(9),
        "kappa_RA" : self.get_par_double(10),
        "K_u" : self.get_par_double(11),
        "K_du" : self.get_par_double(12),
        "K_chi" : self.get_par_double(13),
        "K_dchi" : self.get_par_double(14),
        "K_e" : self.get_par_double(15),
        "G_1" : self.get_par_double(16),
        "G_2" : self.get_par_double(17),
        "G_3" : self.get_par_double(18),
        "G_4" : self.get_par_double(19),
        "epsilon_rdp" : self.get_par_double(20),
        "u_offsets" : self.get_par_vector(0),
        "chi_offsets" : self.get_par_vector(1),
        "CPE_method" : self.get_cpe_method(),
        "prediction_method" : self.get_prediction_method(),
        "guidance_method" : self.get_guidance_method(),
        "use_intention_model" : self.get_par_bool(0),
        "use_path_pruning_ownship" : self.get_par_bool(1),
        "use_path_pruning_targetship" : self.get_par_bool(2),
        "use_GPU" : self.get_par_bool(3)
    }
    return output

# Read from data and write to setter methods. The setters  
# writes to the C++ PSBMPC Parameters class
@classmethod
def from_dict(cls, data: dict) -> PSBMPCInterface.PSBMPCParams:
    # check where the data is comming from
    if not isinstance(data["u_offsets"][0], np.ndarray): # if data from a .yaml file
        fix_u_offsets = np.array(data["u_offsets"])
        fix_chi_offsets = np.array(data["chi_offsets"])
        data["u_offsets"] = [fix_u_offsets]
        data["chi_offsets"] = [fix_chi_offsets]
        
        if data["CPE_method"] == "CE":
            data["CPE_method"] = PSBMPCInterface.CPEMethod.CE
        elif data["CPE_method"] == "MCSKF4D":
            data["CPE_method"] = PSBMPCInterface.CPEMethod.MCSKF4D

        if data["prediction_method"] == "Linear":
            data["prediction_method"] = PSBMPCInterface.PredictionMethod.Linear
        elif data["prediction_method"] == "ERK1":
            data["prediction_method"] = PSBMPCInterface.PredictionMethod.ERK1

        if data["guidance_method"] == "LOS":
            data["guidance_method"] = PSBMPCInterface.GuidanceMethod.LOS
        elif data["guidance_method"] == "WPP":
            data["guidance_method"] = PSBMPCInterface.GuidanceMethod.WPP
        elif data["guidance_method"] == "CH":
            data["guidance_method"] = PSBMPCInterface.GuidanceMethod.CH
    
    psbmpcparams = PSBMPCInterface.PSBMPCParams()
    psbmpcparams.set_par_int(0, data["n_M"])
    psbmpcparams.set_par_int(1, data["n_do_ps"])
    psbmpcparams.set_par_int(2, data["p_step_opt"])
    psbmpcparams.set_par_int(3, data["p_step_do"])
    psbmpcparams.set_par_int(4, data["p_step_grounding"])
    psbmpcparams.set_par_double(0, data["T"])
    psbmpcparams.set_par_double(1, data["dt"])
    psbmpcparams.set_par_double(2, data["t_ts"])
    psbmpcparams.set_par_double(3, data["d_safe"])
    psbmpcparams.set_par_double(4, data["d_do_relevant"])
    psbmpcparams.set_par_double(5, data["d_so_relevant"])
    psbmpcparams.set_par_double(6, data["K_coll"])
    psbmpcparams.set_par_double(7, data["T_coll"])
    psbmpcparams.set_par_double(8, data["kappa_SO"])
    psbmpcparams.set_par_double(9, data["kappa_GW"])
    psbmpcparams.set_par_double(10, data["kappa_RA"])
    psbmpcparams.set_par_double(11, data["K_u"])
    psbmpcparams.set_par_double(12, data["K_du"])
    psbmpcparams.set_par_double(13, data["K_chi"])
    psbmpcparams.set_par_double(14, data["K_dchi"])
    psbmpcparams.set_par_double(15, data["K_e"])
    psbmpcparams.set_par_double(16, data["G_1"])
    psbmpcparams.set_par_double(17, data["G_2"])
    psbmpcparams.set_par_double(18, data["G_3"])
    psbmpcparams.set_par_double(19, data["G_4"])
    psbmpcparams.set_par_double(20, data["epsilon_rdp"])
    psbmpcparams.set_par_vector(0, data["u_offsets"])
    psbmpcparams.set_par_vector(1, data["chi_offsets"])
    psbmpcparams.set_cpe_method(data["CPE_method"])
    psbmpcparams.set_prediction_method(data["prediction_method"])
    psbmpcparams.set_guidance_method(data["guidance_method"])
    psbmpcparams.set_par_bool(0, data["use_intention_model"])
    psbmpcparams.set_par_bool(1, data["use_path_pruning_ownship"])
    psbmpcparams.set_par_bool(2, data["use_path_pruning_targetship"])
    psbmpcparams.set_par_bool(3, data["use_GPU"])
    print("Inside here")
    print('data["use_intention_model"]: ', data["use_intention_model"])
    print('data["use_path_pruning_ownship"]: ', data["use_path_pruning_ownship"])
    print('data["use_path_pruning_targetship"]:', data["use_path_pruning_targetship"])
    print('data["use_GPU"]: ', data["use_GPU"])
    return psbmpcparams


# Monkey patching to_dict() and from_dict() to the PSBMPCParams class
PSBMPCInterface.PSBMPCParams.to_dict = to_dict
PSBMPCInterface.PSBMPCParams.from_dict = from_dict
