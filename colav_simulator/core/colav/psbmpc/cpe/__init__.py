from colav_simulator.core.colav.psbmpc import PSBMPCInterface



# Monkey patch of to_dict() and from_dict() for the CPE class

# Read from all getter methods, which themselves reads from the
# C++ class member variables, and then writes to the output dict
def to_dict(self) -> dict:
    output = {
        "CPE_method" : self.get_cpe_method(),
        "n_CE" : self.get_n_CE(),
        "n_MCSKF" : self.get_n_MCSKF(),
        "alpha_n" : self.get_alpha_n(),
        "gate" : self.get_gate(),
        "rho" : self.get_rho(),
        "max_it" : self.get_max_it(),
        "q" : self.get_q(),
        "r" : self.get_r()
    }
    return output

# Read from data and write to setter methods. The setters  
# writes to the C++ CPE class
@classmethod
def from_dict(cls, data: dict) -> PSBMPCInterface.CPE:
    # check where the data is comming from
    if isinstance(data["CPE_method"], str):
        if data["CPE_method"] == "CE":
            data["CPE_method"] = PSBMPCInterface.CPEMethod.CE
        elif data["CPE_method"] == "MCSKF4D":
            data["CPE_method"] = PSBMPCInterface.CPEMethod.MCSKF4D

    cpe = PSBMPCInterface.CPE(data["CPE_method"])
    cpe.set_n_CE(data["CPE_method"]),
    cpe.set_n_CE(data["n_CE"]),
    cpe.set_n_MCSKF(data["n_MCSKF"]),
    cpe.set_alpha_n(data["alpha_n"]),
    cpe.set_gate(data["gate"]),
    cpe.set_rho(data["rho"]),
    cpe.set_max_it(data["max_it"]),
    cpe.set_q(data["q"]),
    cpe.set_r(data["r"])
    return cpe


# Monkey patching to_dict() and from_dict() to the CPE class
PSBMPCInterface.CPE.to_dict = to_dict
PSBMPCInterface.CPE.from_dict = from_dict
