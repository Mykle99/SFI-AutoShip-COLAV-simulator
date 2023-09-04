from colav_simulator.core.colav.psbmpc import PSBMPCInterface
from colav_simulator.core.colav.psbmpc.psbmpcparams import *
from colav_simulator.core.colav.psbmpc.kinematicship import *
from colav_simulator.core.colav.psbmpc.cpe import *
from dataclasses import dataclass



@dataclass
class PSBMPCParamsWrapper:
    """Parameter wrapper class for the three classes which has settable parameters and are used by the PSBMPC algorithm."""

    psbmpcparams: PSBMPCInterface.PSBMPCParams = None
    ownshipparams: PSBMPCInterface.KinematicShip = None
    cpeparams: PSBMPCInterface.CPE = None


# Monkey patching PSBMPCParamsWrapper class to the PSBMPCInterface Python module
PSBMPCInterface.PSBMPCParamsWrapper = PSBMPCParamsWrapper


# Using monkey patched to_dict() methods to read from all getter
# methods, which themselves reads from the C++ class member 
# variables, and then writes to the output dict
def to_dict(self) -> dict:
    output = {
        "psbmpc_params" : PSBMPCInterface.PSBMPCParams.to_dict(self.psbmpcparams),
        "psbmpc_ownship_params" : PSBMPCInterface.KinematicShip.to_dict(self.ownshipparams),
        "psbmpc_cpe_params" : PSBMPCInterface.CPE.to_dict(self.cpeparams)
    }
    return output

# Read from data and write to setter methods through
# respective monkey patched from_dict() methods.
@classmethod
def from_dict(cls, data: dict) -> PSBMPCInterface.PSBMPCParamsWrapper:
    psbmpcparamswrapper = PSBMPCInterface.PSBMPCParamsWrapper()
    psbmpcparamswrapper.psbmpcparams = PSBMPCInterface.PSBMPCParams.from_dict(data["psbmpc_params"])
    psbmpcparamswrapper.ownshipparams = PSBMPCInterface.KinematicShip.from_dict(data["psbmpc_ownship_params"])
    psbmpcparamswrapper.cpeparams = PSBMPCInterface.CPE.from_dict(data["psbmpc_cpe_params"])
    return psbmpcparamswrapper


# Monkey patching to_dict() and from_dict() to the PSBMPCParamsWrapper class
PSBMPCInterface.PSBMPCParamsWrapper.to_dict = to_dict
PSBMPCInterface.PSBMPCParamsWrapper.from_dict = from_dict
