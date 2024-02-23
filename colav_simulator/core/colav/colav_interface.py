"""
    colav_interface.py

    Summary:
        Contains the interface used by all COLAV planning algorithms that
        wants to be run with the COLAV simulator.

        To add a new COLAV planning algorithm internally to the simulator:

        1: Import necessary algorithm modules in this file.
        2: Add the algorithm name as a type to the COLAVType enum.
        3: Add the algorithm as an optional entry to the LayerConfig class.
        4: Create a new wrapper class for your COLAV algorithm,
        which implements (inherits as this is python) the ICOLAV interface. It should take in a Config object as input.
        5: Add an entry in the COLAVBuilder class, which builds it from config if the type matches.
        See an example for the Kuwata VO and SBMPC below.
        6: Add configuration support for the algorithm by expanding the `colav` entry under `schemas/scenario.yaml` in the `ship_list` section.

        Alternatively, to be able to use a third-party COLAV planning algorithm:

        1: Import this module in your own code.
        2: Create a wrapper class for your COLAV algorithm that implements the ICOLAV interface.
        3: Provide your third-party algorithm to the simulator at run-time (see Simulator class in simulator.py).

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from seacharts.enc import ENC
from colav_simulator.core.colav.psbmpc import PSBMPCInterface as psbmpcI
from colav_simulator.core.colav.im import IMInterface as imI
from shapely.geometry import Point 

import colav_simulator.common.config_parsing as cp
import colav_simulator.common.map_functions as map_functions
import colav_simulator.core.colav.kuwata_vo_alg.kuwata_vo as kvo
import colav_simulator.core.colav.sbmpc.sbmpc as sb_mpc
import colav_simulator.core.guidances as guidance
import colav_simulator.core.stochasticity as stochasticity
import colav_simulator.common.paths as dp
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import math



class COLAVType(Enum):
    """Enum for the different COLAV algorithms currently compatible with the simulator."""

    VO = 0        # Kuwata VO, with LOS guidance to provide velocity references.
    SBMPC = 1     # SB-MPC, provide trajectory offsets
    IM = 2        # Ship Intention Inference Model
    PSBMPC = 3    # Probabilistic SB-MPC
    SBMPC_CPP = 4 # SB-MPC C++ implementation


@dataclass
class LayerConfig:
    """Configuration class for the parameters of a single layer/algorithm in the COLAV planning hierarchy.

    Each layer represent a specific COLAV algorithm, and the parameters are specific to the algorithm. For example with three layers,

    the first layer will be a static obstacle collision-free planner, run e.g. only at the start of the mission,
    the second layer is a mid-level MPC-based COLAV system, that can handle both static and dynamic obstacles (and the COLREGS),
    the third layer is a lower level reactive VO-based COLAV, that handles emergency maneuvers and close encounters if the mid-level planner fails.

    NOTE: This class is typically only used when you want to configure the COLAV system parameters from a scenario file. However,
          an easier option is to configure the COLAV system externally, and pass the COLAV object to the simulator at run-time
          (see examples/dummy_planner.py for an example of this). This is recommended if you want to use a third-party COLAV algorithm.
    """

    vo: Optional[kvo.VOParams] = field(default_factory=lambda: kvo.VOParams())
    los: Optional[guidance.LOSGuidanceParams] = None
    sbmpc: Optional[sb_mpc.SBMPCParams] = None
    im: Optional[imI.IMParamsWrapper] = None
    psbmpc: Optional[psbmpcI.PSBMPCParamsWrapper] = None
    sbmpc_cpp : Optional[psbmpcI.SBMPCParamsWrapper] = None

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
            config.im = cp.convert_settings_dict_to_paramsclass(imI.IMParamsWrapper, config_dict["im"])

        if "psbmpc" in config_dict:
            config.psbmpc = cp.convert_settings_dict_to_paramsclass(psbmpcI.PSBMPCParamsWrapper, config_dict["psbmpc"])

        if "sbmpc_cpp" in config_dict:
            config.sbmpc_cpp = cp.convert_settings_dict_to_paramsclass(psbmpcI.SBMPCParamsWrapper, config_dict["sbmpc_cpp"])

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
        
        if self.psbmpc is not None:
            config_dict["psbmpc"] = self.psbmpc.to_dict()

        if self.sbmpc_cpp is not None:
            config_dict["sbmpc_cpp"] = self.sbmpc_cpp.to_dict()

        return config_dict


@dataclass
class Config:
    """Configuration class for managing COLAV system parameters for all considered layers in the COLAV hierarchy."""

    name: COLAVType = COLAVType.VO
    layer1: LayerConfig = field(default_factory=lambda: LayerConfig())
    layer2: Optional[LayerConfig] = None
    layer3: Optional[LayerConfig] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config(name=COLAVType[config_dict["name"]], layer1=LayerConfig.from_dict(config_dict["layer1"]))

        if "layer2" in config_dict:
            config.layer2 = LayerConfig.from_dict(config_dict["layer2"])

        if "layer3" in config_dict:
            config.layer3 = LayerConfig.from_dict(config_dict["layer3"])

        return config

    def to_dict(self) -> dict:
        config_dict = {"name": self.name.name, "layer1": self.layer1.to_dict()}

        if self.layer2 is not None:
            config_dict["layer2"] = self.layer2.to_dict()

        if self.layer3 is not None:
            config_dict["layer3"] = self.layer3.to_dict()

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
        w: Optional[stochasticity.DisturbanceData] = None,
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
            w (Optional[stochasticity.DisturbanceData]): The stochastic disturbance data. Defaults to None.
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
        w: Optional[stochasticity.DisturbanceData] = None,
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
    """SBMPC wrapper for the Python implementation in this repository."""

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
        w: Optional[stochasticity.DisturbanceData] = None,
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


class SBMPCCPPWrapper(ICOLAV):
    """SBMPC wrapper for the C++ implementation in thecolavrepo repository."""

    def __init__(self, config: Config, **kwargs) -> None:
        assert config.layer1.sbmpc_cpp.sbmpcparams is not None, "SBMPC parameters must be defined in the SBMPCParamsWrapper class."
        self._sbmpc_params = config.layer1.sbmpc_cpp.sbmpcparams

        assert config.layer1.sbmpc_cpp.ownshipparams is not None, "A kinematic ship model of the ownship must be defined in the PSBMPCParamsWrapper class."
        self._sbmpc_ownship = psbmpcI.KinematicShip(config.layer1.sbmpc_cpp.ownshipparams)

        assert config.layer1.sbmpc_cpp is not None, "SBMPC must be on the first layer for the SBMPC wrapper."
        self._sbmpc = psbmpcI.SBMPC(self._sbmpc_ownship, self._sbmpc_params)

        assert config.layer2.los is not None, "LOS guidance must be on the second layer for the SBMPC wrapper."
        self._los = guidance.LOSGuidance(config.layer2.los)

        self._sbmpc_targetshipparams = config.layer1.sbmpc_cpp.targetshipparams.to_dict()

        self._obstacle_predictor = psbmpcI.ObstaclePredictor(
            self._sbmpc_params, 
            self._sbmpc_targetshipparams["r_ct"], 
            self._sbmpc_targetshipparams["path_prediction_shape"],
            self._sbmpc_targetshipparams["chi_offsets"]
        )

        self._t_prev = 0.0
        self._initialized = False
        self._t_run_sbmpc_last = 0.0
        self._t_upd_static_obstacle_last = 0.0
        self._speed_os_best = 1.0
        self._course_os_best = 0.0
        self._trajectory_os_best = []
        self._min_depth = 5
        self._min_distance_to_land =  10
        self._radius_of_coverage = 550
        self._angle_of_coverage_behind = 37.5
        self._obstacles = []
        self._obs_pred_hor_T = self._sbmpc_params.get_par_double(0)
        self._obs_pred_dt = self._sbmpc_params.get_par_double(1)
        self._epsilon_rdp = 25
        self._grounding_hazards_in_enc = None
        self._relevant_grounding_hazards = None
        self._new_static_obstacle_data = True
        _n_obs_pred_scen = self._sbmpc_params.get_par_int(1)
        self._obs_pred_scen_Prob = np.ones(_n_obs_pred_scen) 
        self._obs_pred_scen_Prob = self._obs_pred_scen_Prob/np.sum(self._obs_pred_scen_Prob) # Set to uniform distribution

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: Optional[ENC] = None,
        goal_state: Optional[np.ndarray] = None,
        w: Optional[stochasticity.DisturbanceData] = None,
        V_w: float = 0.0,
        wind_direction: np.ndarray = np.array([0, 0]),
        **kwargs
    ) -> np.ndarray:
        if not self._initialized:
            self._t_prev = t
            self._initialized = True

            self._grounding_hazards_in_enc = map_functions.extract_grounding_hazards_from_entire_enc(
                self._min_depth, self._min_distance_to_land, enc
            )
            ownship_state_cor = [ownship_state[1], ownship_state[0], math.degrees(ownship_state[2])]
            rel_grounding_hazards = map_functions.extract_grounding_hazards_from_relevant_sector_in_enc(
                self._grounding_hazards_in_enc, ownship_state_cor, self._radius_of_coverage, self._angle_of_coverage_behind, enc, False
            )
            gdf = gpd.GeoSeries(rel_grounding_hazards)
            simplified_geometries = gdf.simplify(self._epsilon_rdp, preserve_topology = True)
            self._relevant_grounding_hazards = map_functions.multi_polygon_to_list_of_ndarray_flip_x_y(simplified_geometries)
            self._new_static_obstacle_data = True

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
        
        for do in do_list:
            obs_id = do[0]
            x, y, Vx, Vy = do[1]

            # Update dynamic obstacles
            obs_covar = psbmpcI.flatten(do[2])
            obs_len = do[3]
            obs_wid = do[4]
            A, B, C, D = obs_len/2, obs_len/2, obs_wid/2, obs_wid/2
            obs_aug_state = np.array([x, y, Vx, Vy, A, B, C, D, obs_id])
            
            in_tracked_obstacles = False
            for obstacle in self._obstacles:
                if obstacle.get_ID() == obs_id:
                    obstacle.update_with_state_and_cov(
                        obs_aug_state, obs_covar, False, self._obs_pred_dt
                    )
                    in_tracked_obstacles = True
                    break
            if not in_tracked_obstacles:
                self._obstacles.append(
                    psbmpcI.TrackedObstacle(obs_aug_state, obs_covar, self._obs_pred_scen_Prob, False, self._obs_pred_hor_T, self._obs_pred_dt)
                )
        
        # Defining os_SBMPC
        x, y, psi, u, v, _ = ownship_state # last var is r (unused)
        cog = psi
        sog = np.linalg.norm(np.array([u, v]))
        os_SBMPC = np.array([x, y, cog, sog]) 

        self._obstacles = self._obstacle_predictor(
            self._obstacles, 
            os_SBMPC, 
            self._sbmpc_params,
            psbmpcI.PathPredictionShape.SMOOTH
        )

        references = self._los.compute_references(waypoints, speed_plan, None, ownship_state, t - self._t_prev)
        self._t_prev = t
        course_ref = references[2, 0]
        speed_ref = references[3, 0]
        if t - self._t_run_sbmpc_last >= 1.5:

            if t - self._t_upd_static_obstacle_last >= 1.5:
                ownship_state_cor = [ownship_state[1], ownship_state[0], math.degrees(ownship_state[2])]
                rel_grounding_hazards = map_functions.extract_grounding_hazards_from_relevant_sector_in_enc(
                    self._grounding_hazards_in_enc, ownship_state_cor, self._radius_of_coverage, self._angle_of_coverage_behind, enc, False
                )
                gdf = gpd.GeoSeries(rel_grounding_hazards)
                simplified_geometries = gdf.simplify(self._epsilon_rdp, preserve_topology = True)
                self._relevant_grounding_hazards = map_functions.multi_polygon_to_list_of_ndarray_flip_x_y(simplified_geometries)
                self._new_static_obstacle_data = True
                self._t_upd_static_obstacle_last = t

            os_sbmpc_pred = self._sbmpc.calculate_optimal_offsets(
                speed_ref, course_ref, waypoints, os_SBMPC, V_w, wind_direction, self._relevant_grounding_hazards, \
                self._obstacles, self._new_static_obstacle_data, False 
            )
            
            self._new_static_obstacle_data = False
            self._speed_os_best = os_sbmpc_pred.u_opt
            self._course_os_best = os_sbmpc_pred.chi_opt
            self._trajectory_os_best = os_sbmpc_pred.predicted_trajectory
            self._t_run_sbmpc_last = t
            print(f"SBMPC course output: {round(np.rad2deg(course_ref), 2) + self._course_os_best} | Best course offset: {round(np.rad2deg(self._course_os_best), 2)} | Nominal course ref: {round(np.rad2deg(course_ref), 2)}")
            print(f"SBMPC speed output: {speed_ref * self._speed_os_best} | Best speed offset: {self._speed_os_best} | Nominal speed ref: {speed_ref}")
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


class IMWrapper(ICOLAV):
    """Intention Model wrapper"""

    def __init__(self, config: Config, **kwargs) -> None:
        assert config.layer1.im is not None, "IM must be on the first layer for the IM wrapper."

        assert config.layer2.los is not None, "LOS guidance must be on the second layer for the IM wrapper."
        self._los = guidance.LOSGuidance(config.layer2.los)

        self.ship_intentions = {}
        self.parameters = imI.IMParams.default_parameters(2)
        
        # IM Priors
        self.intention_model_path = str(dp.im / "intention_model_from_code.xdsl")

        # Writing IM data to file
        self._intention_prediction_file = str(dp.intention_output / "intention_file.csv")
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
        w: Optional[stochasticity.DisturbanceData] = None,
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
                        dist = imI.Geometry.evaluateDistance(ship_states[ship_id][imI.Geometry.PX] - ship_states[os_id][imI.Geometry.PX],\
                                                        ship_states[ship_id][imI.Geometry.PY] - ship_states[os_id][imI.Geometry.PY])

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
        assert config.layer1.psbmpc.psbmpcparams is not None, "PSBMPC parameters must be defined in the PSBMPCParamsWrapper class."
        self._psbmpc_params = config.layer1.psbmpc.psbmpcparams

        assert config.layer1.psbmpc.ownshipparams is not None, "A kinematic ship model of the ownship must be defined in the PSBMPCParamsWrapper class."
        self._psbmpc_ownship = psbmpcI.KinematicShip(config.layer1.psbmpc.ownshipparams)

        assert config.layer1.psbmpc.cpeparams is not None, "A Collision Probability Estimator must be defined in the PSBMPCParamsWrapper class."
        self._psbmpc_cpe = psbmpcI.CPE(config.layer1.psbmpc.cpeparams)

        assert config.layer1.psbmpc is not None, "PSBMPC must be on the first layer for the PSBMPC wrapper."
        self._psbmpc = psbmpcI.PSBMPC(self._psbmpc_ownship, self._psbmpc_cpe, self._psbmpc_params)

        assert config.layer2.im is not None, "IM must be on the second layer for the PSBMPC wrapper."
        self._im_params = imI.IMParams.copy_parameters_to_new_instance(config.layer2.im)

        assert config.layer3.los is not None, "LOS guidance must be on the third layer for the PSBMPC wrapper."
        self._los = guidance.LOSGuidance(config.layer3.los)

        self._psbmpc_targetshipparams = config.layer1.psbmpc.targetshipparams.to_dict()

        self._obstacle_predictor = psbmpcI.ObstaclePredictor(
            self._psbmpc_params, 
            self._psbmpc_targetshipparams["r_ct"], 
            self._psbmpc_targetshipparams["path_prediction_shape"],
            self._psbmpc_targetshipparams["chi_offsets"]
        )

        # IM currently only works for 2 ships (1 ownship + 1 obstacle ship)
        self._ship_intentions = {}
    
        # IM Priors
        self._intention_model_path = str(dp.im / "intention_model_from_code.xdsl")

        # Writing IM data to files
        self._intention_prediction_file = str(dp.intention_output / "intention_file.csv")
        self._trajectory_prediction_file = str(dp.intention_output / "trajectory_file.csv")

        with open(self._intention_prediction_file, 'w') as intentionFile:
            intentionFile.write("mmsi,x,y,time,colreg_compliant,good_seamanship,unmodeled_behaviour,has_turned_portwards,has_turned_starboardwards,change_in_speed,is_changing_course,CR_PS,CR_SS,HO,OT_en,OT_ing,priority_lower,priority_similar,priority_higher,risk_of_collision,current_risk_of_collision,start\n")
        with open(self._trajectory_prediction_file, 'w') as intentionFile:
            intentionFile.write("mmsi,time,traj_id,x,y,prob\n")

        self._t_prev = 0.0
        self._initialized = False
        self._t_run_im_last = 0.0
        self._t_run_psbmpc_last = 0.0
        self._t_upd_static_obstacle_last = 0.0
        self._speed_os_best = 1.0
        self._course_os_best = 0.0
        self._trajectory_os_best = []
        self._min_depth = 5
        self._min_distance_to_land =  10
        self._radius_of_coverage = 550
        self._angle_of_coverage_behind = 37.5
        self._obstacles = []
        self._obs_pred_hor_T = self._psbmpc_params.get_par_double(0)
        self._obs_pred_dt = self._psbmpc_params.get_par_double(1)
        # self._epsilon_rdp = self._psbmpc_params.get_par_double(20)
        self._epsilon_rdp = 25
        self._obs_pred_scen_Prob = {}
        self._did_intention_inference_run = {}
        self._grounding_hazards_in_enc = None
        self._relevant_grounding_hazards = None
        self._new_static_obstacle_data = True
        self._grounding_hazard_disks_all_ships = {}
        self._did_grounding_hazard_disks_all_ships_exist = {}
        self._n_pred_obstacle = {}

        self._period_psbmpc = 1.5
        self._use_im = self._psbmpc_params.get_par_bool(0)
        self._use_path_pruning_targetship = self._psbmpc_params.get_par_bool(2)

    def plan(
        self,
        t: float,
        waypoints: np.ndarray,
        speed_plan: np.ndarray,
        ownship_state: np.ndarray,
        do_list: list,
        enc: ENC,
        goal_state: Optional[np.ndarray] = None,
        w: Optional[stochasticity.DisturbanceData] = None,
        V_w: float = 0.0,
        wind_direction: np.ndarray = np.array([0, 0]),
        **kwargs
    ) -> np.ndarray:
        if not self._initialized:
            self._t_prev = t
            self._initialized = True

            self._grounding_hazards_in_enc = map_functions.extract_grounding_hazards_from_entire_enc(
                self._min_depth, self._min_distance_to_land, enc
            )
            ownship_state_cor = [ownship_state[1], ownship_state[0], math.degrees(ownship_state[2])]
            rel_grounding_hazards = map_functions.extract_grounding_hazards_from_relevant_sector_in_enc(
                self._grounding_hazards_in_enc, ownship_state_cor, self._radius_of_coverage, self._angle_of_coverage_behind, enc, False
            )
            gdf = gpd.GeoSeries(rel_grounding_hazards)
            simplified_geometries = gdf.simplify(self._epsilon_rdp, preserve_topology = True)
            self._relevant_grounding_hazards = map_functions.multi_polygon_to_list_of_ndarray_flip_x_y(simplified_geometries)
            self._new_static_obstacle_data = True

        # Update obstacle everytime either the IM or the PSBMPC runs
        if (t - self._t_run_psbmpc_last >= self._period_psbmpc): 
            # Format data to be compatible with IM
            # Making pairs of OS (0) and TS (1) for the IM, hence only 0 and 1 are needed for the mmsi_list 
            mmsi_list = imI.IM.IntVector()
            mmsi_list.append(0)
            mmsi_list.append(1)

            # Dict for keeping pairs of OS and TS states
            ship_states_dict = {} 

            # Ownship ID needs to be in the ship list
            os_id = 0
            x, y, psi, u, v, _ = ownship_state # last var is r (unused)
            cog = psi
            sog = np.linalg.norm(np.array([u, v]))
            os_ship_state = np.array([x, y, cog, sog])
            os_PSBMPC = np.array([x, y, cog, sog]) 

            for do in do_list:
                obs_id = do[0]
                x, y, Vx, Vy = do[1]
                cog = np.arctan2(Vy, Vx)
                sog = np.linalg.norm(np.array([Vx, Vy]))
                do_ship_state = np.array([x, y, cog, sog])
                
                ship_states = imI.IM.IntVector4dMap()
                ship_states[0] = os_ship_state
                ship_states[1] = do_ship_state

                # dict where every obs_id (obs ids -> 1, 2, 3, ..., n) is a key and the value 
                # is a ship_states dict where the values are a pair of OS (id = 0) and TS (id = 1) ids
                ship_states_dict[obs_id] = ship_states

                # Update dynamic obstacles
                obs_covar = psbmpcI.flatten(do[2])
                obs_len = do[3]
                obs_wid = do[4]
                A, B, C, D = obs_len/2, obs_len/2, obs_wid/2, obs_wid/2
                obs_aug_state = np.array([x, y, Vx, Vy, A, B, C, D, obs_id])

                in_tracked_obstacles = False
                for obstacle in self._obstacles:
                    if obstacle.get_ID() == obs_id:
                        obstacle.update_with_state_and_cov(
                            obs_aug_state, obs_covar, False, self._obs_pred_dt
                        )
                        in_tracked_obstacles = True
                        break

                # d_do_relevant > the estimated distance between the OS and the given TS
                if ((self._psbmpc_params.get_par_double(4) > math.sqrt((ownship_state[0] - do_ship_state[0])**2 + (ownship_state[1] - do_ship_state[1])**2))):
                    self._n_pred_obstacle[obs_id] = self._psbmpc_params.get_par_int(1)
                    if (self._use_path_pruning_targetship):
                        do_ship_state_cor = [do_ship_state[1], do_ship_state[0]] 
                        self._grounding_hazard_disks_all_ships[obs_id] = map_functions.extract_grounding_hazards_method_from_GPU_paper(
                            self._grounding_hazards_in_enc, self._radius_of_coverage, do_ship_state_cor 
                        )        
                else:
                    if obs_id in self._grounding_hazard_disks_all_ships:
                        self._grounding_hazard_disks_all_ships.pop(obs_id)
                    self._n_pred_obstacle[obs_id] = 1
                
                # Add dynamic obstacle if not added before
                if not in_tracked_obstacles:
                    self._obs_pred_scen_Prob[obs_id] = np.ones(self._n_pred_obstacle[obs_id])
                    self._obs_pred_scen_Prob[obs_id] = self._obs_pred_scen_Prob[obs_id]/np.sum(self._obs_pred_scen_Prob[obs_id])
                    self._obstacles.append(
                        psbmpcI.TrackedObstacle(obs_aug_state, obs_covar, self._obs_pred_scen_Prob[obs_id], False, self._obs_pred_hor_T, self._obs_pred_dt)
                    )
                    self._did_intention_inference_run[obs_id] = False 
        
            self._obstacles = self._obstacle_predictor(
                self._obstacles, 
                os_PSBMPC, 
                self._psbmpc_params, 
                psbmpcI.PathPredictionShape.SMOOTH
            )

        # Intention Model
        if (t - self._t_run_psbmpc_last >= self._period_psbmpc and self._use_im):
            # Prune the obstacle's paths
            # Executed as often as the IM is (to get updated Prs for each path, old Prs for old paths are not applicable to new paths)
            if self._use_path_pruning_targetship: 
                for obs_id in self._grounding_hazard_disks_all_ships:
                    if self._grounding_hazard_disks_all_ships[obs_id].is_empty: # Only prune if there is static obstacles nearby
                        continue
                    for obstacle in self._obstacles:
                        if obstacle.get_ID() == obs_id:
                            self.prune_obstacle_paths(obstacle)
                            break

            # Calculates intentions. If not initialized, initializes if distance is small enough and sog is large enough
            for do in do_list:
                ship_id = do[0]
                if ship_id != os_id:
                    dist = imI.Geometry.evaluateDistance(
                            ship_states_dict[ship_id][1][imI.Geometry.PX] - ship_states[os_id][imI.Geometry.PX], 
                            ship_states_dict[ship_id][1][imI.Geometry.PY] - ship_states[os_id][imI.Geometry.PY]
                    )
                    own_ship_sog = ship_states[os_id][3]
                    if ((dist < self._im_params.starting_distance) and (own_ship_sog > 0.1) and (self._did_intention_inference_run[ship_id] == False)): # init, should start running)
                        self._ship_intentions[ship_id] = imI.IM.IntentionModel(self._intention_model_path, self._im_params, 1, ship_states_dict[ship_id])
                        self._ship_intentions[ship_id].run_intention_inference(ship_states_dict[ship_id], mmsi_list, t)
                        x, y = ship_states_dict[ship_id][1][0:2]
                        self._did_intention_inference_run[ship_id] = True
                    elif ((ship_id in self._ship_intentions) and (dist < self._im_params.starting_distance) and ((self._did_intention_inference_run[ship_id] == True))): # already init, and should continue running)
                        self._ship_intentions[ship_id].run_intention_inference(ship_states_dict[ship_id], mmsi_list, t)
                        x, y = ship_states_dict[ship_id][1][0:2]
                        self._did_intention_inference_run[ship_id] = True
                    elif ((ship_id in self._ship_intentions) and (dist > self._im_params.starting_distance/10) and ((self._did_intention_inference_run[ship_id] == True))): # already init, and should stop running
                        self._did_intention_inference_run[ship_id] = False
                        self._ship_intentions.pop(ship_id)
                    else: # not ready to init
                        self._did_intention_inference_run[ship_id] = False

                    if self._did_intention_inference_run[ship_id]:
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
                                #print(f"Pr_CCEM^DO{ship_id - 1} set to {Pr_CCEM}")
                                #print(f"Pr_W GW^DO{ship_id - 1} set to {Pr_WGW}")
                                break  

            trajectory_candidates_dict = {}
            updated_trajectory_dict = {}
            for do in do_list:
                ship_id = do[0]
                updated_trajectory_dict[ship_id] = False
                pred_obs_trajectories_dict = {}
                trajectory_candidates_dict[ship_id] = {}
                if (ship_id != os_id) and (ship_id in self._ship_intentions):
                    for obstacle_i in self._obstacles:
                        if obstacle_i.get_ID() == ship_id:
                            pred_obs_trajectories_dict[ship_id] = obstacle_i.get_trajectories()
                            if len(pred_obs_trajectories_dict[ship_id]) != 1: # d_do_relevant will cause "n_ps" to be 1 if distance is too large
                                updated_trajectory_dict[ship_id] = True
                            break
                    
                    # Change state in predicted trajectories from [x, y, Vx, Vy] to [x, y, cog, sog]
                    if updated_trajectory_dict[ship_id]:
                        for i in range(len(pred_obs_trajectories_dict[ship_id])):
                            Vx_i = pred_obs_trajectories_dict[ship_id][i][2]
                            Vy_i = pred_obs_trajectories_dict[ship_id][i][3]

                            cog_i = np.arctan2(Vy_i, Vx_i)
                            sog_i = np.linalg.norm(np.array([Vx_i, Vy_i]))

                            pred_obs_trajectories_dict[ship_id][i][2] = cog_i
                            pred_obs_trajectories_dict[ship_id][i][3] = sog_i

                            trajectory_candidates_dict[ship_id][i] = pred_obs_trajectories_dict[ship_id][i]

                        # TODO: Change time_into_trajectory in parameters.h to be at the time step with the biggest deviation in cog. 
                        # Might be a tuning parameter
                        self._ship_intentions[ship_id].run_trajectory_inference(ship_states_dict[ship_id], mmsi_list, trajectory_candidates_dict[ship_id], self._obs_pred_dt)
                        # Comment this part out if you do not want to save to file
                        self._ship_intentions[ship_id].save_trajectories_to_file(self._trajectory_prediction_file, t, trajectory_candidates_dict[ship_id])

                        traj_probabilities_dict = {}
                        traj_probabilities_dict = self._ship_intentions[ship_id].get_traj_probabilities() 
                        traj_probabilities_list = np.array(list(traj_probabilities_dict.values()))

                        # Pr_s^ship_id is only set during situation (not before, not after)
                        # "len(traj_probabilities_list) = 1 before and after situation"
                        # len(traj_probabilities_list) = len(self._obs_pred_scen_Prob) during situation
                        if len(traj_probabilities_list) == self._n_pred_obstacle[ship_id]:
                            for obstacle_j in self._obstacles:
                                if obstacle_j.get_ID() == ship_id:
                                    self._obs_pred_scen_Prob[ship_id] = traj_probabilities_list
                                    if np.sum(self._obs_pred_scen_Prob[ship_id]) < 0.05: # if only small probabilities, then set to uniform
                                        self._obs_pred_scen_Prob[ship_id] = np.ones(self._n_pred_obstacle[ship_id])
                                        self._obs_pred_scen_Prob[ship_id] = self._obs_pred_scen_Prob[ship_id]/np.sum(self._obs_pred_scen_Prob[ship_id])
                                    obstacle_j.set_scenario_probabilities(self._obs_pred_scen_Prob[ship_id])
                                    #print(f"Pr_s^DO{ship_id - 1} set to {self._obs_pred_scen_Prob[ship_id]}")
                                    break
                    else:
                        for obstacle_k in self._obstacles:
                            if obstacle_k.get_ID() == ship_id:
                                self._obs_pred_scen_Prob[ship_id] = np.ones(self._n_pred_obstacle[ship_id])
                                self._obs_pred_scen_Prob[ship_id] = self._obs_pred_scen_Prob[ship_id]/np.sum(self._obs_pred_scen_Prob[ship_id])
                                obstacle_k.set_scenario_probabilities(self._obs_pred_scen_Prob[ship_id])
                                #print(f"Pr_s^DO{ship_id - 1} set to {self._obs_pred_scen_Prob[ship_id]}")
                                break

        references = self._los.compute_references(waypoints, speed_plan, None, ownship_state, t - self._t_prev)
        self._t_prev = t
        course_ref = references[2, 0]
        speed_ref = references[3, 0]
        if (t - self._t_run_psbmpc_last >= self._period_psbmpc):
            if t - self._t_upd_static_obstacle_last >= self._period_psbmpc:
                ownship_state_cor = [ownship_state[1], ownship_state[0], math.degrees(ownship_state[2])]
                rel_grounding_hazards = map_functions.extract_grounding_hazards_from_relevant_sector_in_enc(
                    self._grounding_hazards_in_enc, ownship_state_cor, self._radius_of_coverage, self._angle_of_coverage_behind, enc, False
                )
                gdf = gpd.GeoSeries(rel_grounding_hazards)
                simplified_geometries = gdf.simplify(self._epsilon_rdp, preserve_topology = True)
                self._relevant_grounding_hazards = map_functions.multi_polygon_to_list_of_ndarray_flip_x_y(simplified_geometries)
                self._new_static_obstacle_data = True
                self._t_upd_static_obstacle_last = t

            os_psbmpc_pred = self._psbmpc.calculate_optimal_offsets( 
                speed_ref, course_ref, waypoints, os_PSBMPC, V_w, wind_direction, self._relevant_grounding_hazards, \
                self._obstacles, self._new_static_obstacle_data, False
            )

            self._new_static_obstacle_data = False
            self._speed_os_best = os_psbmpc_pred.u_opt
            self._course_os_best = os_psbmpc_pred.chi_opt
            self._trajectory_os_best = os_psbmpc_pred.predicted_trajectory
            self._t_run_psbmpc_last = t
            #print(f"PSBMPC course output: {round(np.rad2deg(course_ref + self._course_os_best), 4)} | Best course offset: {np.rad2deg(self._course_os_best)} | Nominal course ref: {round(np.rad2deg(course_ref), 4)}")
            #print(f"PSBMPC speed output: {speed_ref * self._speed_os_best} | Best speed offset: {self._speed_os_best} | Nominal speed ref: {speed_ref}")
        references[2, 0] += self._course_os_best
        references[3, 0] = speed_ref * self._speed_os_best
        return references

    def prune_obstacle_paths(self, obstacle: psbmpcI.TrackedObstacle) -> None:
        """Prunes the obstacle's paths.

        Args:
            - obstacle (psbmpcI.TrackedObstacle): The obstacle to prune the paths of.
        """
        paths = obstacle.get_trajectories()
        paths_v = obstacle.get_mean_velocity_trajectories()
        obs_id = obstacle.get_ID()
        n_paths = np.shape(paths)[0]
        
        if n_paths == 1:
            self._update_obstacle_properties(obstacle, obs_id, paths, paths_v, n_paths)
            return 
        
        path_shortening_divisor = 2
        while path_shortening_divisor < 6: # Tuning parameter
            path_count = 0
            remove_paths_indices = []
            for path in paths: # Check if the paths at any index is inside any of the grounding hazards
                break_step_in_path_loop = False
                for k in range(np.shape(path)[1]//round(path_shortening_divisor)):
                    point = Point(path[1, k], path[0, k]) # Flipping of x and y (due to difference in seacharts and thecolavrepo's coordinate systems)
                    if break_step_in_path_loop == True:
                        break
                    for hazard in self._grounding_hazard_disks_all_ships[obs_id].geoms:
                        if hazard.contains(point):
                            remove_paths_indices.append(path_count)
                            break_step_in_path_loop = True
                            break
                path_count += 1
            path_shortening_divisor *= 1.25

        if remove_paths_indices:
            if len(remove_paths_indices) < n_paths:
                remaining_paths = np.delete(paths, remove_paths_indices, axis = 0)
                remaining_paths_v = obstacle.get_mean_velocity_trajectories()
                n_remaining_paths = np.shape(remaining_paths)[0]
                if np.shape(paths_v)[0] != 0: # Only used by the MROU prediction scheme (LOS is used)
                    remaining_paths_v = np.delete(paths_v, remove_paths_indices, axis = 0)
                self._update_obstacle_properties(obstacle, obs_id, remaining_paths, remaining_paths_v, n_remaining_paths)
            else: # If all paths are inside a grounding hazard (then paths cannot be removed)
                self._update_obstacle_properties(obstacle, obs_id, paths, paths_v, n_paths)
        else:
            self._update_obstacle_properties(obstacle, obs_id, paths, paths_v, n_paths)
        return
    
    def _update_obstacle_properties(self, obstacle: psbmpcI.TrackedObstacle, obs_id: int, paths: np.ndarray, paths_v: np.ndarray, n_paths: int) -> None:
        """Updates the obstacle's properties. Keeps the obstacle's properties in the simulator and the C++ code consistent.

        Args:
            - obstacle (psbmpcI.TrackedObstacle): The obstacle to update the properties of.
            - obs_id (int): The obstacle's ID.
            - paths (np.ndarray): The obstacle's paths (positions).
            - paths_v (np.ndarray): The obstacle's mean velocity paths.
            - n_paths (int): The number of paths.
        """
        # Set to correct value
        self._n_pred_obstacle[obs_id] = n_paths

        # Update paths (the other properties (set in this method) are set according to the paths)
        if np.shape(paths)[0] != 0:
            obstacle.set_trajectories(paths)
        if np.shape(paths_v)[0] != 0: # Only used by the MROU prediction scheme (LOS is used)
            obstacle.set_mean_velocity_trajectories(paths_v)
        
        # Set to correct length (temp uniform Prs, upd by IM)
        scen_prob = np.ones(n_paths)
        scen_prob = scen_prob/np.sum(scen_prob)
        obstacle.set_scenario_probabilities(scen_prob)

        # Set to correct value
        i = 0
        for obs in self._obstacles:
            if obs.get_ID() == obs_id:
                self._obstacle_predictor.set_n_ps_i(i, n_paths)
                break
            i += 1
        return

    def get_current_plan(self) -> np.ndarray:
        refs = np.zeros((9, 1))
        return refs

    def get_colav_data(self) -> dict:
        return {}

    def plot_results(self, ax_map: plt.Axes, enc: ENC, plt_handles: dict, **kwargs) -> dict:
        return plt_handles


class COLAVBuilder:
    @classmethod
    def construct_colav(cls, config: Optional[Config] = None) -> Optional[ICOLAV]:
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
        elif config and config.name == COLAVType.SBMPC_CPP:
            colav = SBMPCCPPWrapper(config)
        else:
            colav = None

        return colav
