from colav_simulator.core.colav.psbmpc import PSBMPCInterface
import numpy as np



# Monkey patch of to_dict() and from_dict() for the SBMPCParams class

# Read from all getter methods, which themselves reads from the
# C++ class member variables, and then writes to the output dict
def to_dict(self) -> dict:
    output = {
        "n_M" : self.get_par_int(0),
        "n_do_ps" : self.get_par_int(1),
        "p_step_opt" : self.get_par_int(2),
        "p_step_grounding" : self.get_par_int(3),
        "T" : self.get_par_double(0),
        "dt" : self.get_par_double(1),
        "t_ts" : self.get_par_double(2),
        "d_safe" : self.get_par_double(3),
        "d_close" : self.get_par_double(4),
        "d_do_relevant" : self.get_par_double(5),
        "d_so_relevant" : self.get_par_double(6),
        "K_coll" : self.get_par_double(7),
        "phi_AH" : self.get_par_double(8),
        "phi_OT" : self.get_par_double(9),
        "phi_HO" : self.get_par_double(10),
        "phi_CR" : self.get_par_double(11),
        "kappa" : self.get_par_double(12),
        "kappa_TC" : self.get_par_double(13),
        "K_u" : self.get_par_double(14),
        "K_du" : self.get_par_double(15),
        "K_chi_strb" : self.get_par_double(16),
        "K_dchi_strb" : self.get_par_double(17),
        "K_chi_port" : self.get_par_double(18),
        "K_dchi_port" : self.get_par_double(19),
        "K_sgn" : self.get_par_double(20),
        "T_sgn" : self.get_par_double(21),
        "q" : self.get_par_double(22),
        "p" : self.get_par_double(23),
        "G_1" : self.get_par_double(24),
        "G_2" : self.get_par_double(25),
        "G_3" : self.get_par_double(26),
        "G_4" : self.get_par_double(27),
        "epsilon_rdp" : self.get_par_double(28),
        "u_offsets" : self.get_par_vector(0),
        "chi_offsets" : self.get_par_vector(1),
        "prediction_method" : self.get_prediction_method(),
        "guidance_method" : self.get_guidance_method()
    }
    return output

# Read from data and write to setter methods. The setters  
# writes to the C++ SBMPC Parameters class
@classmethod
def from_dict(cls, data: dict) -> PSBMPCInterface.SBMPCParams:
    # check where the data is comming from
    if not isinstance(data["u_offsets"][0], np.ndarray): # if data from a .yaml file
        fix_u_offsets = np.array(data["u_offsets"])
        fix_chi_offsets = np.array(data["chi_offsets"])
        data["u_offsets"] = [fix_u_offsets]
        data["chi_offsets"] = [fix_chi_offsets]

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
    
    sbmpcparams = PSBMPCInterface.SBMPCParams()
    sbmpcparams.set_par_int(0, data["n_M"])
    sbmpcparams.set_par_int(1, data["n_do_ps"])
    sbmpcparams.set_par_int(2, data["p_step_opt"])
    sbmpcparams.set_par_int(3, data["p_step_grounding"])
    sbmpcparams.set_par_double(0, data["T"])
    sbmpcparams.set_par_double(1, data["dt"])
    sbmpcparams.set_par_double(2, data["t_ts"])
    sbmpcparams.set_par_double(3, data["d_safe"])
    sbmpcparams.set_par_double(4, data["d_close"])
    sbmpcparams.set_par_double(5, data["d_do_relevant"])
    sbmpcparams.set_par_double(6, data["d_so_relevant"])
    sbmpcparams.set_par_double(7, data["K_coll"])
    sbmpcparams.set_par_double(8, data["phi_AH"])
    sbmpcparams.set_par_double(9, data["phi_OT"])
    sbmpcparams.set_par_double(10, data["phi_HO"])
    sbmpcparams.set_par_double(11, data["phi_CR"])
    sbmpcparams.set_par_double(12, data["kappa"])
    sbmpcparams.set_par_double(13, data["kappa_TC"])
    sbmpcparams.set_par_double(14, data["K_u"])
    sbmpcparams.set_par_double(15, data["K_du"])
    sbmpcparams.set_par_double(16, data["K_chi_strb"])
    sbmpcparams.set_par_double(17, data["K_dchi_strb"])
    sbmpcparams.set_par_double(18, data["K_chi_port"])
    sbmpcparams.set_par_double(19, data["K_dchi_port"])
    sbmpcparams.set_par_double(20, data["K_sgn"])
    sbmpcparams.set_par_double(21, data["T_sgn"])
    sbmpcparams.set_par_double(22, data["q"])
    sbmpcparams.set_par_double(23, data["p"])
    sbmpcparams.set_par_double(24, data["G_1"])
    sbmpcparams.set_par_double(25, data["G_2"])
    sbmpcparams.set_par_double(26, data["G_3"])
    sbmpcparams.set_par_double(27, data["G_4"])
    sbmpcparams.set_par_double(28, data["epsilon_rdp"])
    sbmpcparams.set_par_vector(0, data["u_offsets"])
    sbmpcparams.set_par_vector(1, data["chi_offsets"])
    sbmpcparams.set_prediction_method(data["prediction_method"])
    sbmpcparams.set_guidance_method(data["guidance_method"])
    return sbmpcparams


# Monkey patching to_dict() and from_dict() to the SBMPCParams class
PSBMPCInterface.SBMPCParams.to_dict = to_dict
PSBMPCInterface.SBMPCParams.from_dict = from_dict
