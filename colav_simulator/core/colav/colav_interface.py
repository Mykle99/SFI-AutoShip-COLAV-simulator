"""
    colav_interface.py

    Summary:
        Contains the interface used by all COLAV planning algorithms that
        wants to be run with the COLAV simulator.

        To add a new COLAV planning algorithm:

        1: Import the algorithm in this file.
        2: Add the algorithm name as a type to the COLAVType enum.
        3: Add the algorithm as an optional entry to the LayerConfig class.
        4: Create a new wrapper class for your COLAV algorithm,
        which implements (inherits as this is python) this interface. It should take in a Config object as input.
        5: Add an entry in the COLAVBuilder class, which builds it from config if the type matches.
        See an example for the Kuwata VO below.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from seacharts.enc import ENC
from colav_simulator.core.colav.psbmpc import PSBMPCInterface as psbmpcI
from colav_simulator.core.colav.im import IMInterface as imI 

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.map_functions as map_functions
import colav_simulator.core.colav.kuwata_vo_alg.kuwata_vo as kvo
import colav_simulator.core.colav.sbmpc.sbmpc as sb_mpc
import colav_simulator.core.guidances as guidance
import matplotlib.pyplot as plt
import numpy as np
import os



class COLAVType(Enum):
    """Enum for the different COLAV algorithms currently compatible with the simulator."""

    VO = 0     # Kuwata VO, with LOS guidance to provide velocity references.
    SBMPC = 1  # SB-MPC, provide trajectory offsets
    IM = 2     # Ship Intention Inference Model
    PSBMPC = 3 # Probabilistic SB-MPC


@dataclass
class LayerConfig:
    """Configuration class for the parameters of a single layer/algorithm in the COLAV planning hierarchy."""

    vo: Optional[kvo.VOParams] = field(default_factory = lambda: kvo.VOParams())
    los: Optional[guidance.LOSGuidanceParams] = None
    sbmpc: Optional[sb_mpc.SBMPCParams] = None
    im: Optional[imI.IMParams.IntentionModelParameters] = None
    psbmpc_params: Optional[psbmpcI.PSBMPCParams] = None
    psbmpc_ownship : Optional[psbmpcI.KinematicShip] = None
    psbmpc_cpe : Optional[psbmpcI.CPE] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = LayerConfig()
        if "vo" in config_dict:
            config.vo = cp.convert_settings_dict_to_dataclass(kvo.VOParams, config_dict["vo"])

        if "los" in config_dict:
            config.los = cp.convert_settings_dict_to_dataclass(guidance.LOSGuidanceParams, config_dict["los"])

        if "sbmpc" in config_dict:
            config.sbmpc = cp.convert_settings_dict_to_dataclass(sb_mpc.SBMPCParams, config_dict["sbmpc"])

        if "im" in config_dict:
            config.im = cp.convert_settings_dict_to_paramsclass(imI.IMParams.IntentionModelParameters, config_dict["im"])

        if "psbmpc_params" in config_dict:
            config.psbmpc_params = cp.convert_settings_dict_to_paramsclass(psbmpcI.PSBMPCParams, config_dict["psbmpc_params"])

        if "psbmpc_ownship" in config_dict:
            config.psbmpc_ownship = cp.convert_settings_dict_to_paramsclass(psbmpcI.KinematicShip, config_dict["psbmpc_ownship"])

        if "psbmpc_cpe" in config_dict:
            config.psbmpc_cpe = cp.convert_settings_dict_to_paramsclass(psbmpcI.CPE, config_dict["psbmpc_cpe"])

        return config

    def to_dict(self) -> dict:
        config_dict = {}

        if self.vo is not None:
            config_dict["vo"] = self.vo.to_dict()

        if self.los is not None:
            config_dict["los"] = self.los.to_dict()

        if self.sbmpc is not None:
            config_dict["sbmpc"] = self.sbmpc.to_dict()

        if self.im is not None:
            config_dict["im"] = self.im.to_dict()
        
        if self.psbmpc_params is not None:
            config_dict["psbmpc_params"] = self.psbmpc_params.to_dict()

        if self.psbmpc_ownship is not None:
            config_dict["psbmpc_ownship"] = self.psbmpc_ownship.to_dict()

        if self.psbmpc_cpe is not None:
            config_dict["psbmpc_cpe"] = self.psbmpc_cpe.to_dict()

        return config_dict


@dataclass
class Config:
    """Configuration class for managing COLAV system parameters for all considered layers in the COLAV hierarchy."""

    name: COLAVType = COLAVType.VO
    layer1: LayerConfig = field(default_factory=lambda: LayerConfig())
    layer2: Optional[LayerConfig] = None
    layer3: Optional[LayerConfig] = None
    layer4: Optional[LayerConfig] = None
    layer5: Optional[LayerConfig] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(name=COLAVType[config_dict["name"]], layer1=LayerConfig.from_dict(config_dict["layer1"]))

        if "layer2" in config_dict:
            config.layer2 = LayerConfig.from_dict(config_dict["layer2"])

        if "layer3" in config_dict:
            config.layer3 = LayerConfig.from_dict(config_dict["layer3"])
        
        if "layer4" in config_dict:
            config.layer4 = LayerConfig.from_dict(config_dict["layer4"])

        if "layer5" in config_dict:
            config.layer5 = LayerConfig.from_dict(config_dict["layer5"])

        return config

    def to_dict(self) -> dict:
        config_dict = {"name": self.name.name, "layer1": self.layer1.to_dict()}

        if self.layer2 is not None:
            config_dict["layer2"] = self.layer2.to_dict()

        if self.layer3 is not None:
            config_dict["layer3"] = self.layer3.to_dict()

        if self.layer4 is not None:
            config_dict["layer4"] = self.layer4.to_dict()

        if self.layer5 is not None:
            config_dict["layer5"] = self.layer5.to_dict()

        return config_dict


class ICOLAV(ABC):
    @abstractmethod
    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[ENC] = None,
        goal_state: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Plans a (hopefully) collision free trajectory for the ship to follow.

        Args:
            t (float): The current time.
            waypoints (np.ndarray): The waypoints to follow, typically used for COLAV planners assuming a nominal path/trajectory as input.
            speed_plan (np.ndarray): The speed plan to follow. typically used for COLAV planners assuming a nominal path/trajectory as input.
            ownship_state (np.ndarray): The ownship state [x, y, psi, u, v, r]. Used as start state in case of high level planners.
            do_list (list): List of information on dynamic obstacles. This is a list of tuples of the form (id, state [x, y, Vx, Vy], covariance, length, width).
            enc (Optional[ENC]): The relevant Electronic Navigational Chart (ENC) for static obstacle info. Defaults to None.
            goal_state (Optional[np.ndarray]): The goal state [x, y, psi, u, v, r], typically used for high level COLAV planners where no nominal path/trajectory is assumed. Defaults to None.
            **kwargs: Additional arguments to the COLAV planning algorithm, e.g. the own-ship length.

        Returns:
            np.ndarray: The planned poses, velocities and accelerations (vstacked) from the COLAV planning algorithm. Must be compatible with the control system you are using.
        """

    @abstractmethod
    def get_current_plan(self) -> np.ndarray:
        """Returns the current planned trajectory.

        Returns:
            np.ndarray: The most recent planned poses, velocities and accelerations (vstacked) over the COLAV planning horizon (if any). Must be compatible with the control system you are using.
        """

    @abstractmethod
    def get_colav_data(self) -> dict:
        """Returns the plotting data relevant for the COLAV planning algorithm. This includes e.g. the predicted trajectory, considered obstacles, optimal inputs etc..

        Returns:
            dict: The relevant data used in the COLAV planning algorithm.
        """

    @abstractmethod
    def plot_results(self, ax_map: plt.Axes, enc: ENC, plt_handles: dict, **kwargs) -> dict:
        """Plots the COLAV planning algorithm results data, e.g. the predicted trajectory, considered obstacles, optimal inputs etc..

        Args:
            ax_map (plt.Axes): Map axes to plot on.
            enc (senc.ENC): ENC object.
            plt_handles (dict): Dictionary of plot handles.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary of plot handles."""


class VOWrapper(ICOLAV):
    """The VO wrapper is a Kuwata VO-based reactive COLAV planning system, where LOS-guidance is used to provide velocity references."""

    def __init__(self, config: Config, **kwargs) -> None:
        assert config.layer1.vo is not None, "Kuwata VO must be on the first layer for the VO wrapper."
        self._vo = kvo.VO(config.layer1.vo)

        assert config.layer2 and config.layer2.los is not None, "LOS guidance must be on the second layer for the VO wrapper."
        self._los = guidance.LOSGuidance(config.layer2.los)

        self._t_prev = 0.0
        self._initialized = False

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[ENC] = None,
        goal_state: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        if not self._initialized:
            self._t_prev = t
            self._initialized = True

        references = self._los.compute_references(waypoints, speed_plan, None, ownship_state, t - self._t_prev)
        self._t_prev = t
        course_ref = references[2, 0]
        speed_ref = references[3, 0]
        vel_ref = np.array([speed_ref * np.cos(course_ref), speed_ref * np.sin(course_ref)])
        return self._vo.plan(t, vel_ref, ownship_state, do_list, enc)

    def get_current_plan(self) -> np.ndarray:
        return self._vo.get_current_plan()

    def get_colav_data(self) -> dict:
        return {}

    def plot_results(self, ax_map: plt.Axes, enc: ENC, plt_handles: dict, **kwargs) -> dict:
        return plt_handles


class SBMPCWrapper(ICOLAV):
    """SBMPC wrapper"""

    def __init__(self, config: Config, **kwargs) -> None:
        assert config.layer1.sbmpc is not None, "SBMPC must be on the first layer for the SBMPC wrapper."
        self._sbmpc = sb_mpc.SBMPC(config.layer1.sbmpc)

        assert config.layer2.los is not None, "LOS guidance must be on the second layer for the SBMPC wrapper."
        self._los = guidance.LOSGuidance(config.layer2.los)

        self._t_prev = 0.0
        self._initialized = False
        self._t_run_sbmpc_last = 0.0
        self._speed_os_best = 1.0
        self._course_os_best = 0.0

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[ENC] = None,
        goal_state: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        if not self._initialized:
            self._t_prev = t
            self._initialized = True

        references = self._los.compute_references(waypoints, speed_plan, None, ownship_state, t - self._t_prev)
        self._t_prev = t
        course_ref = references[2, 0]
        speed_ref = references[3, 0]
        if t - self._t_run_sbmpc_last >= 5.0:
            self._speed_os_best, self._course_os_best = self._sbmpc.get_optimal_ctrl_offset(speed_ref, course_ref, ownship_state, do_list)
            self._t_run_sbmpc_last = t
            # print(f"SBMPC course output: {np.rad2deg(course_ref) + self._course_os_best} | Best course offset: {self._course_os_best} | Nominal course ref: {course_ref}")
            # print(f"SBMPC speed output: {speed_ref * self._speed_os_best} | Best speed offset: {self._speed_os_best} | Nominal speed ref: {speed_ref}")
        references[2, 0] += np.deg2rad(self._course_os_best)
        references[3, 0] = speed_ref * self._speed_os_best
        return references

    def get_current_plan(self) -> np.ndarray:
        refs = np.zeros((9, 1))
        return refs

    def get_colav_data(self) -> dict:
        return {}

    def plot_results(self, ax_map: plt.Axes, enc: ENC, plt_handles: dict, **kwargs) -> dict:
        return plt_handles


class IMWrapper(ICOLAV):
    """Intention Model wrapper"""

    def __init__(self, config: Config, **kwargs) -> None:
        assert config.layer1.im is not None, "IM must be on the first layer for the IM wrapper."

        assert config.layer2.los is not None, "LOS guidance must be on the second layer for the IM wrapper."
        self._los = guidance.LOSGuidance(config.layer2.los)

        self.ship_intentions = {}
        self.parameters = imI.IMParams.default_parameters(2)
        #Priors
        self.intention_model_path = "colav_simulator/core/colav/cpp_to_py_interfaces/external/ship_intention_inference/files/intention_models/intention_model_from_code.xdsl"

        #For writing to file
        self.intention_prediction_file = "output/intention_files/intention_file.csv"
        with open(self.intention_prediction_file, 'w') as intentionFile:
            intentionFile.write("mmsi,x,y,time,colreg_compliant,good_seamanship,unmodeled_behaviour,has_turned_portwards,has_turned_starboardwards,change_in_speed,is_changing_course,CR_PS,CR_SS,HO,OT_en,OT_ing,priority_lower,priority_similar,priority_higher,risk_of_collision,current_risk_of_collision,start\n")

        #For live plot
        #self.fig_im, self.axs_im = plt.subplots(7, 1)
        #plt.xlabel('Time')
        #plt.title('Ship Intentions')
        self._t_plot_im_last = 0.0

        self._t_prev = 0.0
        self._initialized = False
        self._t_run_im_last = 0.0
        self._speed_os_best = 1.0
        self._course_os_best = 0.0

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[ENC] = None,
        goal_state: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        if not self._initialized:
            self._t_prev = t
            self._initialized = True

        references = self._los.compute_references(waypoints, speed_plan, None, ownship_state, t - self._t_prev)
        self._t_prev = t
        course_ref = references[2, 0]
        speed_ref = references[3, 0]
        if t - self._t_run_im_last >= 2.0: # Change this for faster runtime
            self._t_run_im_last = t

            # Intention Model
            ship_states = imI.IM.IntVector4dMap()
            mmsi_list = imI.IM.IntVector()

            #TODO: ownship ID needs to be in ship list
            os_id = 0
            mmsi_list.append(os_id)
            x, y, psi, u, v, r = ownship_state
            cog = psi
            sog = np.linalg.norm(np.array([u, v]))
            ship_states[os_id] = np.array([x, y, cog, sog])

            for do in do_list:
                mmsi_list.append(do[0])
                x, y, Vx, Vy = do[1]
                cog = np.arctan2(Vy, Vx)
                sog = np.linalg.norm(np.array([Vx, Vy]))
                ship_states[do[0]] = np.array([x, y, cog, sog])

            #comment out this section for intention model on dynamic obstacles only
            if os_id in self.ship_intentions:
                self.ship_intentions[os_id].run_intention_inference(ship_states, mmsi_list, t)
                x, y = ship_states[os_id][0:2]
                self.ship_intentions[os_id].save_intention_predictions_to_file(self.intention_prediction_file,\
                                                                                                x, y, t)


            for ship_id in mmsi_list:
                if ship_id != os_id:
                    if ship_id in self.ship_intentions:
                        self.ship_intentions[ship_id].run_intention_inference(ship_states, mmsi_list, t)
                        x, y = ship_states[ship_id][0:2]
                        self.ship_intentions[ship_id].save_intention_predictions_to_file(self.intention_prediction_file,\
                                                                                            x, y, t)
                    else:
                        dist = imI.IMGeometry.evaluateDistance(ship_states[ship_id][imI.IMGeometry.PX] - ship_states[os_id][imI.IMGeometry.PX],\
                                                        ship_states[ship_id][imI.IMGeometry.PY] - ship_states[os_id][imI.IMGeometry.PY])

                        own_ship_sog = ship_states[os_id][3]
                        if  ((dist < self.parameters.starting_distance) \
                            and (own_ship_sog > 0.1)):

                            #comment out this section for intention model on dynamic obstacles only
                            if os_id not in self.ship_intentions:
                                self.ship_intentions[os_id] = imI.IM.IntentionModel(self.intention_model_path \
                                                                                                , self.parameters \
                                                                                                , os_id \
                                                                                                , ship_states)

                            if ship_id not in self.ship_intentions:
                                self.ship_intentions[ship_id] = imI.IM.IntentionModel(self.intention_model_path \
                                                                                        , self.parameters \
                                                                                        , ship_id \
                                                                                        , ship_states)


        references[2, 0] += np.deg2rad(self._course_os_best)
        references[3, 0] = speed_ref * self._speed_os_best

        return references

    def get_current_plan(self) -> np.ndarray:
        refs = np.zeros((9, 1))
        return refs

    def get_colav_data(self) -> dict:
        return {}

    def plot_results(self, ax_map: plt.Axes, enc: ENC, plt_handles: dict, **kwargs) -> dict:
        if (self._t_run_im_last - self._t_plot_im_last >= 5.0):
            self._t_plot_im_last = self._t_run_im_last
            for ship_id in self.ship_intentions.keys():
                plt_handles[("ship_intentions", ship_id)] = self.ship_intentions[ship_id].get_intention_model_predictions()
                # predicted_intentions = self.ship_intentions[ship_id].get_intention_model_predictions()
                # import pdb
                # pdb.set_trace()
                # self.axs_im[0].plot(self._t_run_im_last, predicted_intentions["intention_colregs_compliant"]["true"], label=f'Ship {ship_id}')
                # self.axs_im[0].legend()
                # self.axs_im[1].plot(self._t_run_im_last, predicted_intentions["intention_good_seamanship"]["true"], label=f'Ship {ship_id}')
                # self.axs_im[1].legend()
                # self.axs_im[2].plot(self._t_run_im_last, predicted_intentions["unmodelled_behaviour"]["true"], label=f'Ship {ship_id}')
                # self.axs_im[2].legend()
                # self.axs_im[3].plot(self._t_run_im_last, predicted_intentions["has_turned_portwards"]["true"], label=f'Ship {ship_id}')
                # self.axs_im[3].legend()
                # self.axs_im[4].plot(self._t_run_im_last, predicted_intentions["has_turned_starboardwards"]["true"], label=f'Ship {ship_id}')
                # self.axs_im[4].legend()
                # self.axs_im[5].plot(self._t_run_im_last, predicted_intentions["change_in_speed"]["similar"], label=f'Ship {ship_id}')
                # self.axs_im[5].legend()
                # self.axs_im[6].plot(self._t_run_im_last, predicted_intentions["is_changing_course"]["true"], label=f'Ship {ship_id}')
                # self.axs_im[6].legend()

        return plt_handles


class PSBMPCWrapper(ICOLAV):
    """PSBMPC wrapper"""

    def __init__(self, config: Config, **kwargs) -> None:
        assert config.layer2.psbmpc_ownship is not None, "A kinematic ship model of the ownship must be on the second layer for the PSBMPC wrapper."
        self._psbmpc_ownship = psbmpcI.KinematicShip(config.layer2.psbmpc_ownship)

        assert config.layer3.psbmpc_cpe is not None, "A Collision Probability Estimator must be on the third layer for the PSBMPC wrapper."
        self._psbmpc_cpe = psbmpcI.CPE(config.layer3.psbmpc_cpe)

        assert config.layer1.psbmpc_params is not None, "PSBMPC must be on the first layer for the PSBMPC wrapper."
        self._psbmpc_params = config.layer1.psbmpc_params 
        self._psbmpc = psbmpcI.PSBMPC(self._psbmpc_ownship, self._psbmpc_cpe, self._psbmpc_params)

        assert config.layer4.im is not None, "IM must be on the fourth layer for the PSBMPC wrapper."
        self._im_params = imI.IMParams.copy_parameters_to_new_instance(config.layer4.im)

        assert config.layer5.los is not None, "LOS guidance must be on the fifth layer for the PSBMPC wrapper."
        self._los = guidance.LOSGuidance(config.layer5.los)

        self._obstacle_predictor = psbmpcI.ObstaclePredictor(self._psbmpc_params)

        # IM currently only works for 2 ships (1 ownship + 1 obstacle ship)
        self._ship_intentions = {}
        
        # IM Priors
        _relative_path_to_im = "cpp_to_py_interfaces/external/ship_intention_inference/files/intention_models/intention_model_from_code.xdsl"
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        self._intention_model_path = os.path.join(_script_dir, _relative_path_to_im)

        # Writing IM data to files
        self._intention_prediction_file = "colav_simulator/output/intention_files/intention_file.csv"
        self._trajectory_prediction_file = "colav_simulator/output/intention_files/trajectory_file.csv"
        with open(self._intention_prediction_file, 'w') as intentionFile:
            intentionFile.write("mmsi,x,y,time,colreg_compliant,good_seamanship,unmodeled_behaviour,has_turned_portwards,has_turned_starboardwards,change_in_speed,is_changing_course,CR_PS,CR_SS,HO,OT_en,OT_ing,priority_lower,priority_similar,priority_higher,risk_of_collision,current_risk_of_collision,start\n")
        with open(self._trajectory_prediction_file, 'w') as intentionFile:
            intentionFile.write("mmsi,time,traj_id,x,y,prob\n")

        self._t_prev = 0.0
        self._initialized = False
        self._t_run_im_last = 0.0
        self._t_run_psbmpc_last = 0.0
        self._speed_os_best = 1.0
        self._course_os_best = 0.0
        self._trajectory_os_best = []
        self._min_depth = 5
        self._obstacles = []
        self._obs_pred_hor_T = self._psbmpc_params.get_par_double(0)
        self._obs_pred_dt = self._psbmpc_params.get_par_double(1)
        self._relevant_polygons = None
        _n_obs_pred_scen = self._psbmpc_params.get_par_int(1)
        self._obs_pred_scen_Prob = np.ones(_n_obs_pred_scen)
        self._obs_pred_scen_Prob = self._obs_pred_scen_Prob/np.sum(self._obs_pred_scen_Prob)

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: ENC,
        goal_state: Optional[np.ndarray] = None,
        V_w: float = 0.0,
        wind_direction: np.ndarray = np.array([0, 0]),
        **kwargs
    ) -> np.ndarray:
        if not self._initialized:
            self._t_prev = t
            self._initialized = True

            relevant_polygons_shapely = map_functions.extract_relevant_grounding_hazards_as_union(self._min_depth, enc, False)
            self._relevant_polygons = map_functions.from_MultiPolygon_to_ndarray(relevant_polygons_shapely)

            for do in do_list:
                obs_id = do[0]
                obs_x, obs_y, obs_Vx, obs_Vy = do[1]
                obs_covar = psbmpcI.flatten(do[2])
                obs_len = do[3]
                obs_wid = do[4]
                A, B, C, D = obs_len/2, obs_len/2, obs_wid/2, obs_wid/2 # (A, B, C, D are from AIS message, dimension quantifiers)
                obs_aug_state = np.array([obs_x, obs_y, obs_Vx, obs_Vy, A, B, C, D, obs_id])

                self._obstacles.append(
                    psbmpcI.TrackedObstacle(obs_aug_state, obs_covar, self._obs_pred_scen_Prob, False, self._obs_pred_hor_T, self._obs_pred_dt)
                )
            
        # Format data to be compatible with IM
        ship_states = imI.IM.IntVector4dMap()
        mmsi_list = imI.IM.IntVector()

        # Ownship ID needs to be in the ship list
        os_id = 0
        mmsi_list.append(os_id)
        x, y, psi, u, v, _ = ownship_state # last var is r (unused)
        cog = psi
        sog = np.linalg.norm(np.array([u, v]))
        ship_states[os_id] = np.array([x, y, cog, sog])
        os_PSBMPC = np.array([x, y, cog, sog]) 

        for do in do_list:
            obs_id = do[0]
            mmsi_list.append(obs_id)
            x, y, Vx, Vy = do[1]
            cog = np.arctan2(Vy, Vx)
            sog = np.linalg.norm(np.array([Vx, Vy]))
            ship_states[do[0]] = np.array([x, y, cog, sog])

            # Update dynamic obstacles
            obs_covar = psbmpcI.flatten(do[2])
            obs_len = do[3]
            obs_wid = do[4]
            A, B, C, D = obs_len/2, obs_len/2, obs_wid/2, obs_wid/2
            obs_aug_state = np.array([x, y, Vx, Vy, A, B, C, D, obs_id])
            for obstacle in self._obstacles:
                if obstacle.get_ID() == obs_id:
                    obstacle.update_with_state_and_cov(
                        obs_aug_state, obs_covar, False, self._obs_pred_dt
                    )
                else:
                    self._obstacles.append(
                        psbmpcI.TrackedObstacle(obs_aug_state, obs_covar, self._obs_pred_scen_Prob, False, self._obs_pred_hor_T, self._obs_pred_dt)
                    )

        # Intention Model
        if t - self._t_run_im_last >= 5.0:
            self._t_run_im_last = t
            did_intention_inference_run = False

            # Calculates intentions. If not initialized, initializes if distance is small enough and sog is large enough
            for ship_id in mmsi_list:
                if ship_id != os_id:
                    if ship_id in self._ship_intentions:
                        self._ship_intentions[ship_id].run_intention_inference(ship_states, mmsi_list, t)
                        x, y = ship_states[ship_id][0:2]
                        did_intention_inference_run = True
                    else:
                        dist = imI.IMGeometry.evaluateDistance(
                            ship_states[ship_id][imI.IMGeometry.PX] - ship_states[os_id][imI.IMGeometry.PX], 
                            ship_states[ship_id][imI.IMGeometry.PY] - ship_states[os_id][imI.IMGeometry.PY]
                        )
                        
                        own_ship_sog = ship_states[os_id][3]
                        if ((dist < self._im_params.starting_distance) and (own_ship_sog > 0.1)):
                            self._ship_intentions[ship_id] = imI.IM.IntentionModel(self._intention_model_path, self._im_params, ship_id, ship_states)
                            self._ship_intentions[ship_id].run_intention_inference(ship_states, mmsi_list, t)
                            did_intention_inference_run = True
                        
                    if did_intention_inference_run:
                        did_intention_inference_run = False

                        # Comment this part out if you do not want to save to file
                        self._ship_intentions[ship_id].save_intention_predictions_to_file(self._intention_prediction_file, x, y, t)

                        # Reading intention nodes
                        intention_information = self._ship_intentions[ship_id].get_intention_model_predictions()
                        I_CC = intention_information["intention_colregs_compliant"]["true"]
                        I_P_rho_higher = intention_information[f"priority_intention_to_ship{str(os_id)}"]["higher"]
                        I_U = intention_information["unmodelled_behaviour"]["true"]
                        
                        # Setting COLREGS compliant evasive maneuver- and will fulfill give-way obligation probabilities
                        Pr_CCEM = I_CC
                        Pr_WGW = (1 - I_P_rho_higher) * I_CC * (1 - I_U)

                        for obstacle in self._obstacles:
                            if obstacle.get_ID() == ship_id:
                                obstacle.set_Pr_CCEM(Pr_CCEM)
                                obstacle.set_Pr_WGW(Pr_WGW)
                                print(f"Pr_CCEM^{ship_id} set to {Pr_CCEM}")
                                print(f"Pr_W GW^{ship_id} set to {Pr_WGW}")
                                
            self._obstacles = self._obstacle_predictor(self._obstacles, os_PSBMPC, self._psbmpc_params)

            for ship_id in mmsi_list:
                if (ship_id != os_id) and (ship_id in self._ship_intentions):
                    trajectory_candidates = {}
                    pred_obs_trajectories = self._obstacles[0].get_trajectories()
                    # Change state in predicted trajectories from [x, y, Vx, Vy] to [x, y, cog, sog]
                    for i in range(len(pred_obs_trajectories)):
                        Vx_i = pred_obs_trajectories[i][2]
                        Vy_i = pred_obs_trajectories[i][3]

                        cog_i = np.arctan2(Vy_i, Vx_i)
                        sog_i = np.linalg.norm(np.array([Vx_i, Vy_i]))

                        pred_obs_trajectories[i][2] = cog_i
                        pred_obs_trajectories[i][3] = sog_i

                        trajectory_candidates[i] = pred_obs_trajectories[i]

                    # TODO: Change time_into_trajectory in parameters.h to be at the time step with the biggest deviation in cog. 
                    # Might be a tuning parameter
                    self._ship_intentions[ship_id].run_trajectory_inference(ship_states, mmsi_list, trajectory_candidates, self._obs_pred_dt)

                    # Comment this part out if you do not want to save to file
                    self._ship_intentions[ship_id].save_trajectories_to_file(self._trajectory_prediction_file, t, trajectory_candidates)

                    traj_probabilities_dict = self._ship_intentions[ship_id].get_traj_probabilities() 
                    traj_probabilities_list = np.array(list(traj_probabilities_dict.values()))

                    # Pr_s^ship_id is only set during situation (not before, not after)
                    # "len(traj_probabilities_list) = 1 before and after situation"
                    # len(traj_probabilities_list) = len(self._obs_pred_scen_Prob) during situation
                    if len(traj_probabilities_list) == len(self._obs_pred_scen_Prob):
                        self._obs_pred_scen_Prob = np.array(list(traj_probabilities_dict.values()))

                        for obstacle in self._obstacles:
                            if obstacle.get_ID() == ship_id:
                                obstacle.set_scenario_probabilities(self._obs_pred_scen_Prob)
                                print(f"Pr_s^{ship_id} set to {self._obs_pred_scen_Prob}")

        references = self._los.compute_references(waypoints, speed_plan, None, ownship_state, t - self._t_prev)
        self._t_prev = t
        course_ref = references[2, 0]
        speed_ref = references[3, 0]
        if t - self._t_run_psbmpc_last >= 5.0: #self._relevant_polygons should be changed with []
            os_psbmpc_pred = self._psbmpc.calculate_optimal_offsets( 
                speed_ref, course_ref, waypoints, os_PSBMPC, V_w, wind_direction, [], self._obstacles, False
            )
            self._speed_os_best = os_psbmpc_pred.u_opt
            self._course_os_best = os_psbmpc_pred.chi_opt
            self._trajectory_os_best = os_psbmpc_pred.predicted_trajectory
            self._t_run_psbmpc_last = t
            print(f"PSBMPC course output: {np.rad2deg(course_ref + self._course_os_best)} | Best course offset: {np.rad2deg(self._course_os_best)} | Nominal course ref: {np.rad2deg(course_ref)}")
            print(f"PSBMPC speed output: {speed_ref * self._speed_os_best} | Best speed offset: {self._speed_os_best} | Nominal speed ref: {speed_ref}")
        references[2, 0] += self._course_os_best
        references[3, 0] = speed_ref * self._speed_os_best
        return references

    def get_current_plan(self) -> np.ndarray:
        refs = np.zeros((9, 1))
        return refs

    def get_colav_data(self) -> dict:
        return {}

    def plot_results(self, ax_map: plt.Axes, enc: ENC, plt_handles: dict, **kwargs) -> dict:
        return plt_handles


class COLAVBuilder:
    @classmethod
    def construct_colav(cls, config: Optional[Config] = None) -> ICOLAV:
        """Builds a colav system from the configuration, if it is specified.

        Args:
            config (Optional[models.Config]): COLAV configuration. Defaults to None.

        Returns:
            ICOLAV: The COLAV system (if any config), e.g. Kuwata VO.
        """
        if config and config.name == COLAVType.VO:
            colav = VOWrapper(config)
        elif config and config.name == COLAVType.SBMPC:
            colav = SBMPCWrapper(config)
        elif config and config.name == COLAVType.IM:
            colav = IMWrapper(config)
        elif config and config.name == COLAVType.PSBMPC:
            colav = PSBMPCWrapper(config)
        else:
            config = Config()
            config.layer2 = LayerConfig()
            config.layer2.los = guidance.LOSGuidanceParams()
            colav = VOWrapper(config)

        return colav
