from colav_simulator.core.colav.psbmpc import PSBMPCInterface
import numpy as np


# Monkey patch of to_dict() and from_dict() for the ObstaclePredictor class

# Read from all getter methods, which themselves reads from the
# C++ class member variables, and then writes to the output dict
def to_dict(self) -> dict:
    output = {
        "r_ct" : self.get_r_ct(),
        "path_prediction_shape" : self.get_path_prediction_shape(),
        "chi_offsets" : self.get_chi_offsets()
    }
    return output


# Read from data and write to setter methods. The setters  
# writes to the C++ ObstaclePredictor class
@classmethod
def from_dict(cls, data: dict) -> PSBMPCInterface.ObstaclePredictor:
    targetship = PSBMPCInterface.ObstaclePredictor()
    targetship.set_r_ct(data["r_ct"])

    if isinstance(data["path_prediction_shape"], str):
        if data["path_prediction_shape"] == "SMOOTH":
            data["path_prediction_shape"] = PSBMPCInterface.PathPredictionShape.SMOOTH
        elif data["path_prediction_shape"] == "LINEAR":
            data["path_prediction_shape"] = PSBMPCInterface.PathPredictionShape.LINEAR

    targetship.set_path_prediction_shape(data["path_prediction_shape"])

    if not isinstance(data["chi_offsets"][0], np.ndarray): # if data from a .yaml file
        fix_chi_offsets = np.array(data["chi_offsets"])
        data["chi_offsets"] = fix_chi_offsets
    
    targetship.set_chi_offsets(data["chi_offsets"])
    
    return targetship


# Monkey patching to_dict() and from_dict() to the ObstaclePredictor class
PSBMPCInterface.ObstaclePredictor.to_dict = to_dict
PSBMPCInterface.ObstaclePredictor.from_dict = from_dict
