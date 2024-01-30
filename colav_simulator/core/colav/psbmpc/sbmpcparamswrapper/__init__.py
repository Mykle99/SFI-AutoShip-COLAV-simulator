from colav_simulator.core.colav.psbmpc import PSBMPCInterface
from colav_simulator.core.colav.psbmpc.sbmpcparams import *
from colav_simulator.core.colav.psbmpc.kinematicship import *
from colav_simulator.core.colav.psbmpc.obstaclepredictor import *
from dataclasses import dataclass



@dataclass
class SBMPCParamsWrapper:
    """Parameter wrapper class for the four classes which has settable parameters and are used by the PSBMPC algorithm."""

    sbmpcparams: PSBMPCInterface.SBMPCParams = None
    ownshipparams: PSBMPCInterface.KinematicShip = None
    targetshipparams: PSBMPCInterface.ObstaclePredictor = None


# Monkey patching SBMPCParamsWrapper class to the PSBMPCInterface Python module
PSBMPCInterface.SBMPCParamsWrapper = SBMPCParamsWrapper


# Using monkey patched to_dict() methods to read from all getter
# methods, which themselves reads from the C++ class member 
# variables, and then writes to the output dict
def to_dict(self) -> dict:
    output = {
        "sbmpc_params" : PSBMPCInterface.SBMPCParams.to_dict(self.sbmpcparams),
        "sbmpc_ownship_params" : PSBMPCInterface.KinematicShip.to_dict(self.ownshipparams),
        "sbmpc_targetship_params" : PSBMPCInterface.ObstaclePredictor.to_dict(self.targetshipparams),
    }
    return output

# Read from data and write to setter methods through
# respective monkey patched from_dict() methods.
@classmethod
def from_dict(cls, data: dict) -> PSBMPCInterface.SBMPCParamsWrapper:
    sbmpcparamswrapper = PSBMPCInterface.PSBMPCParamsWrapper()
    sbmpcparamswrapper.sbmpcparams = PSBMPCInterface.SBMPCParams.from_dict(data["sbmpc_params"])
    sbmpcparamswrapper.ownshipparams = PSBMPCInterface.KinematicShip.from_dict(data["sbmpc_ownship_params"])
    sbmpcparamswrapper.targetshipparams = PSBMPCInterface.ObstaclePredictor.from_dict(data["sbmpc_targetship_params"])
    return sbmpcparamswrapper


# Monkey patching to_dict() and from_dict() to the SBMPCParamsWrapper class
PSBMPCInterface.SBMPCParamsWrapper.to_dict = to_dict
PSBMPCInterface.SBMPCParamsWrapper.from_dict = from_dict
