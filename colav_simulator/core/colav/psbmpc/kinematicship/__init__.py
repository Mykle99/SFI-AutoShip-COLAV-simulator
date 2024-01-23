from colav_simulator.core.colav.psbmpc import PSBMPCInterface



# Monkey patch of to_dict() and from_dict() for the KinematicShip class

# Read from all getter methods, which themselves reads from the
# C++ class member variables, and then writes to the output dict
def to_dict(self) -> dict:
    output = {
        "length" : self.get_length(),
        "width" : self.get_width(),
        "T_U" : self.get_T_U(),
        "T_chi" : self.get_T_chi(),
        "R_a" : self.get_R_a(),
        "LOS_LD" : self.get_LOS_LD(),
        "LOS_K_i" : self.get_LOS_K_i(),
        "active_waypoint" : self.get_wp_counter(),
        "path_prediction_shape" : self.get_path_prediction_shape()
    }
    return output

# Read from data and write to setter methods. The setters  
# writes to the C++ KinematicShip class
@classmethod
def from_dict(cls, data: dict) -> PSBMPCInterface.KinematicShip:
    kinematicship = PSBMPCInterface.KinematicShip()
    kinematicship.set_length(data["length"])
    kinematicship.set_width(data["width"])
    kinematicship.set_T_U(data["T_U"])
    kinematicship.set_T_chi(data["T_chi"])
    kinematicship.set_R_a(data["R_a"])
    kinematicship.set_LOS_LD(data["LOS_LD"])
    kinematicship.set_LOS_K_i(data["LOS_K_i"])
    kinematicship.set_wp_counter(data["active_waypoint"])

    if isinstance(data["path_prediction_shape"], str):
        if data["path_prediction_shape"] == "SMOOTH":
            data["path_prediction_shape"] = PSBMPCInterface.PathPredictionShape.SMOOTH
        elif data["path_prediction_shape"] == "LINEAR":
            data["path_prediction_shape"] = PSBMPCInterface.PathPredictionShape.LINEAR

    kinematicship.set_path_prediction_shape(data["path_prediction_shape"])
    
    return kinematicship


# Monkey patching to_dict() and from_dict() to the KinematicShip class
PSBMPCInterface.KinematicShip.to_dict = to_dict
PSBMPCInterface.KinematicShip.from_dict = from_dict
