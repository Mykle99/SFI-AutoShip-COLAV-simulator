"""
    trackers.py

    Summary:
        Contains class definitions for dynamic obstacle
        target trackers. Every tracker must adhere to the
        ITracker interface.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import colav_simulator.common.config_parsing as cp
import numpy as np
import scipy.linalg as la

#Create a simple import for the VIMMJIPDA package until it is correctly implemented as a submodule.
#TODO: Make decision on how submodule should work and implement it later
from colav_simulator.core.tracking.VIMMJIPDA.code.run import setup_manager
from colav_simulator.core.tracking.VIMMJIPDA.code.tracking.managers import Manager
from colav_simulator.core.tracking.VIMMJIPDA.code.tracking.constructs import State, Measurement, TrackState, Track
from colav_simulator.core.tracking.VIMMJIPDA.code.parameters import measurement_params



class ITracker(ABC):
    @abstractmethod
    def track(self, t: float, dt: float, true_do_states: list, ownship_state: np.ndarray) -> Tuple[list, list]:
        """Tracks/updates estimates on dynamic obstacles, based on sensor measurements
        generated from the input true dynamic obstacle states. Returns tracks and sensor measurements (if any)"""

    def get_track_information(self) -> Tuple[list, list]:
        """Returns the dynamic obstacle track information (ID, state, cov, length, width).
        Also, it returns the associated Normalized Innovation error Squared (NIS) values for
        the most recent update step for each track, and the track labels.

        Returns:
            Tuple[list, list]: List of tracks and list of NISes.
        """


@dataclass
class KFParams:
    """Class for holding KF parameters."""

    P_0: np.ndarray = field(default_factory=lambda: np.diag([49.0, 49.0, 0.1, 0.1]))
    q: float = 0.15

    def to_dict(self):
        output_dict = {"P_0": self.P_0.diagonal().tolist(), "q": self.q}
        return output_dict

    @classmethod
    def from_dict(cls, config_dict):
        return KFParams(P_0=np.diag(config_dict["P_0"]), q=config_dict["q"])
    

@dataclass
class VIMMJIPDAParams:
    """Class for holding VIMMJIPDA parameters."""
    IMM_off : bool = field(default_factory=lambda: False)
    single_target : bool = field(default_factory=lambda: False)
    visibility_off : bool = field(default_factory=lambda: False)

    def to_dict(self):
        output_dict = {"IMM_off": self.IMM_off, "single_target": self.single_target, "visibility_off": self.visibility_off}
        return output_dict
    
    @classmethod
    def from_dict(cls, config_dict):
        return VIMMJIPDAParams(IMM_off=config_dict["IMM_off"], single_target=config_dict["single_target"], visibility_off=config_dict["visibility_off"])
        

@dataclass
class Config:
    """Class for holding tracker configuration parameters."""
    #TODO: Add possibility to config VIMMJIPDA

    god_tracker: Optional[bool] = False
    kf: Optional[KFParams] = field(default_factory=lambda: KFParams())
    VIMMJIPDA: Optional[VIMMJIPDAParams] = field(default_factory=lambda: VIMMJIPDAParams())

    def to_dict(self) -> dict:
        output_dict = {}
        if self.kf is not None:
            output_dict["kf"] = self.kf.to_dict()
        if self.god_tracker is not None:
            output_dict["god_tracker"] = ""
        if self.VIMMJIPDA is not None:
            output_dict["VIMMJIPDA"] = self.VIMMJIPDA.to_dict()
        return output_dict

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = Config()

        if "kf" in config_dict:
            config.kf = cp.convert_settings_dict_to_dataclass(KFParams, config_dict["kf"])
            config.god_tracker = None
        elif "god_tracker" in config_dict:
            config.god_tracker = True
            config.kf = None
            config.VIMMJIPDA = None
        elif "VIMMJIPDA" in config_dict:
            config.god_tracker = False
            config.kf = None
            config.VIMMJIPDA = cp.convert_settings_dict_to_dataclass(VIMMJIPDAParams, config_dict["VIMMJIPDA"])

        return config


class TrackerBuilder:
    @classmethod
    def construct_tracker(cls, sensors: list, config: Optional[Config] = None) -> ITracker:
        """Builds a tracker from the configuration

        Args:
            sensors (list): Sensors used by the tracker.
            config (Optional[Config]): Tracker configuration. Defaults to None.

        Returns:
            ITracker: The tracker.
        """
        if config and config.kf:
            return KF(sensors, config.kf)
        elif config and config.god_tracker:
            return GodTracker(sensors)
        elif config and config.VIMMJIPDA:
            return VIMMJIPDA(sensors, config.VIMMJIPDA)
        else:
            return KF(sensors)


class GodTracker(ITracker):
    """This tracker is used to simulate perfect knowledge of dynamic obstacles."""

    def __init__(self, sensor_list: list) -> None:
        if sensor_list is None:
            raise ValueError("Sensor list must be provided.")

        self.sensors: list = sensor_list

        self._initialized: bool = False
        self._labels: list = []
        self._xs_upd: list = []
        self._P_upd: list = []
        self._length_upd: list = []
        self._width_upd: list = []

    def track(self, t: float, dt: float, true_do_states: list, ownship_state: np.ndarray) -> Tuple[list, list]:
        """Tracks/updates estimates on dynamic obstacles perfectly within the sensor range.

        Args:
            dt (float): Time since last update
            t (float): Current time (assumed >= 0)
            true_do_states (list): List of tuples of true dynamic obstacle indices and states (do_idx, [x, y, Vx, Vy], length, width) x n_do. Used for simulating sensor measurements.
            ownship_state (np.ndarray): Ownship state vector [x, y, Vx, Vy] used for simulating sensor measurements.

        Returns:
            Tuple[list, list]: List of ground truth dynamic obstacle tracks (ID, state, cov, length, width). Also, the sensor measurements list (not used).
        """
        if not self._initialized:
            for do_idx, do_state, do_length, do_width in true_do_states:
                self._labels.append(do_idx)
                self._xs_upd.append(do_state)
                self._P_upd.append(np.zeros((4, 4)))
                self._length_upd.append(do_length)
                self._width_upd.append(do_width)
            self._initialized = True

        # Only generate measurements for initialized tracks
        sensor_measurements = []
        for sensor in self.sensors:
            z = sensor.generate_measurements(t, true_do_states, ownship_state)
            sensor_measurements.append(z)

        tracks = []
        n_tracked_do = len(true_do_states)
        for i in range(n_tracked_do):
            self._xs_upd[i] = true_do_states[i][1]
            tracks.append(
                (
                    self._labels[i],
                    self._xs_upd[i],
                    self._P_upd[i],
                    self._length_upd[i],
                    self._width_upd[i],
                )
            )
        return tracks, sensor_measurements

    def get_track_information(self) -> Tuple[list, list]:
        tracks = []
        for i, label in enumerate(self._labels):
            tracks.append(
                (
                    label,
                    self._xs_upd[i],
                    self._P_upd[i],
                    self._length_upd[i],
                    self._width_upd[i],
                )
            )
        return tracks, [0.0 for _ in range(len(tracks))]


class KF(ITracker):
    """The KF class implements a linear Kalman filter based tracker."""

    def __init__(self, sensor_list: list, params: Optional[KFParams] = None) -> None:
        if sensor_list is None:
            raise ValueError("Sensor list must be provided.")

        if params is not None:
            self._params: KFParams = params
        else:
            self._params = KFParams()

        self._model = CVModel(self._params.q)

        self.sensors: list = sensor_list

        self._track_initialized: list = []
        self._track_terminated: list = []
        self._labels: list = []
        self._xs_p: list = []
        self._P_p: list = []
        self._xs_upd: list = []
        self._P_upd: list = []
        self._length_upd: list = []  # List of DO length estimates. Assumed known
        self._width_upd: list = []  # List of DO width estimates. Assumed known
        self._NIS: list = []

    def track(self, t: float, dt: float, true_do_states: list, ownship_state: np.ndarray) -> Tuple[list, list]:
        """Tracks/updates estimates on dynamic obstacles, based on sensor measurements
        generated from the input true dynamic obstacle states.

        NOTE: The KF assumes one generated measurement per dynamic obstacle per sensor.

        Args:
            dt (float): Time since last update
            t (float): Current time (assumed >= 0)
            true_do_states (list): List of tuples of true dynamic obstacle indices and states (do_idx, [x, y, Vx, Vy], length, width) x n_do. Used for simulating sensor measurements.
            ownship_state (np.ndarray): Ownship state vector [x, y, Vx, Vy] used for simulating sensor measurements.

        Returns:
            Tuple[list, list]: List of updated dynamic obstacle tracks (ID, state, cov, length, width). Also, a list the sensor measurements used.
        """
        max_sensor_range = max([sensor.max_range for sensor in self.sensors])
        for do_idx, do_state, do_length, do_width in true_do_states:
            dist_ownship_to_do = np.linalg.norm(do_state[:2] - ownship_state[:2])
            if do_idx not in self._labels and dist_ownship_to_do < max_sensor_range:
                # New track. TODO: Implement track initiation, e.g. n out of m based initiation.
                self._labels.append(do_idx)
                self._track_initialized.append(False)
                self._track_terminated.append(False)
                self._xs_upd.append(do_state)
                self._P_upd.append(self._params.P_0)
                self._xs_p.append(do_state)
                self._P_p.append(self._params.P_0)
                self._length_upd.append(do_length)
                self._width_upd.append(do_width)
                self._NIS.append(np.nan)
            elif do_idx in self._labels:
                self._track_initialized[self._labels.index(do_idx)] = True

        n_tracked_do = len(self._xs_upd)
        # # TODO: Implement track termination for when covariance is too large.
        # for i in range(n_tracked_do):
        #     if np.sqrt(self._P_upd[i][0, 0]) > 50.0 or np.sqrt(self._P_upd[i][1, 1]) > 50.0:
        #         self._track_terminated[i] = True

        # Only generate measurements for initialized tracks
        sensor_measurements = []
        for sensor in self.sensors:
            z = sensor.generate_measurements(t, true_do_states, ownship_state)
            sensor_measurements.append(z)
        tracks = []
        for i in range(n_tracked_do):
            if self._track_initialized[i] and not self._track_terminated[i]:
                self._xs_p[i], self._P_p[i] = self.predict(self._xs_upd[i], self._P_upd[i], dt)
                self._xs_upd[i] = self._xs_p[i]
                self._P_upd[i] = self._P_p[i]

                if sensor_measurements:
                    for sensor_id in range(len(self.sensors)):
                        z = sensor_measurements[sensor_id][i]
                        self._xs_upd[i], self._P_upd[i], NIS_i = self.update(
                            self._xs_upd[i], self._P_upd[i], z, sensor_id
                        )

                        if not np.isnan(NIS_i):
                            self._NIS[i] = NIS_i
            tracks.append(
                (
                    self._labels[i],
                    self._xs_upd[i],
                    self._P_upd[i],
                    self._length_upd[i],
                    self._width_upd[i],
                )
            )

        # print(f"xs_p: {self._xs_p}, xs_upd: {self._xs_upd}")
        # print(f"P_p: {self._P_p}")
        # print(f"P_upd: {self._P_upd}")
        return tracks, sensor_measurements

    def predict(self, xs_upd: np.ndarray, P_upd: np.ndarray, dt: float):
        F = self._model.F(dt)
        Q = self._model.Q(dt)

        x_pred = self._model.f(xs_upd, dt)
        P_pred = F @ P_upd @ F.T + Q

        return x_pred, P_pred

    def innovation(self, xs_p: np.ndarray, P_p: np.ndarray, z: np.ndarray, sensor_id: int):
        zbar = self.sensors[sensor_id].h(xs_p)
        v = z - zbar

        H = self.sensors[sensor_id].H(xs_p)
        R = self.sensors[sensor_id].R(xs_p)
        S = H @ P_p @ H.T + R

        return v, S

    def update(
        self, xs_p: np.ndarray, P_p: np.ndarray, z: np.ndarray, sensor_id: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if any(np.isnan(z)):
            return xs_p, P_p, np.nan

        v, S = self.innovation(xs_p, P_p, z, sensor_id)
        H = self.sensors[sensor_id].H(xs_p)

        K = P_p @ H.T @ la.inv(S)
        x_upd = xs_p + K @ v
        P_upd = P_p - K @ H @ P_p

        return x_upd, P_upd, NIS(v, S)

    def get_track_information(self) -> Tuple[list, list]:
        tracks = []
        for i, label in enumerate(self._labels):
            tracks.append(
                (
                    label,
                    self._xs_upd[i],
                    self._P_upd[i],
                    self._length_upd[i],
                    self._width_upd[i],
                )
            )
        return tracks, self._NIS


def NIS(v: np.ndarray, S: np.ndarray) -> float:
    return v.T @ la.inv(S) @ v


class CVModel:
    """The CVModel class implements a constant velocity model."""

    def __init__(self, q: float) -> None:
        self._q: float = q

    def f(self, xs: np.ndarray, dt: float) -> np.ndarray:
        """Returns the r.h.s of the prediction model state transition function.

        Args:
            xs (np.ndarray): State vector [x, y, Vx, Vy]
            dt (float): Time step

        Returns:
            np.ndarray: New state vector dt seconds ahead

        """
        return xs + np.array([xs[2] * dt, xs[3] * dt, 0.0, 0.0])

    def F(self, dt: float) -> np.ndarray:
        """Returns the Jacobian of the prediction model state transition function.

        Args:
            xs (np.ndarray): xs (np.ndarray): State vector [x, y, Vx, Vy]
            dt (float): Time step

        Returns:
            np.ndarray: Jacobian of the prediction model state transition function
        """
        return np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def Q(self, dt: float) -> np.ndarray:
        """Returns the process noise covariance matrix.

        Args:
            dt (float): Time step

        Returns:
            np.ndarray: Process noise covariance matrix
        """
        return (
            np.array(
                [
                    [dt**3 / 3.0, 0.0, dt**2 / 2.0, 0.0],
                    [0.0, dt**3 / 3.0, 0.0, dt**2 / 2.0],
                    [dt**2 / 2.0, 0.0, dt, 0.0],
                    [0.0, dt**2 / 2.0, 0.0, dt],
                ]
            )
            * self._q
        )







class VIMMJIPDA(ITracker):
    """The VIMMJIPDA class implements the VIMMJIPDA (Visibility Interacting Multiple Models Joint Integrated Probabilistic Data Association) tracker by
    Audun Gulliksstad Hem, Edmund Førland Brekke and Lars-Christian Ness Tokle, introduced in the article:
    "Multitarget Tracking With Multiple Models and Visibility: Derivation and Verification on Maritime Radar Data"

    NOTE: It is possible to configure this tracker by turning of the functionality for: Interacting Multiple Models, Visibility and Multi-Target turning
    the tracker into an IPDA tracker
    """

    def __init__(self, sensor_list: list, params: Optional[VIMMJIPDAParams] = None) -> None:
        #TODO Add functionality to input sensors. (Give error message for not using radar???)

        if sensor_list is None:
            raise ValueError("Sensor list must be provided.")


        if params is not None:
            self._params: VIMMJIPDAParams = params
        else:
            self._params = VIMMJIPDAParams()


        self.sensors: list = sensor_list

        self._track_initialized: list = []
        self._track_terminated: list = []
        self._labels: list = [] # List of DO IDs and labels
        self._means: list = []
        self._covs: list = []
        self._length_upd: list = []  # List of DO length estimates. Assumed known
        self._width_upd: list = []  # List of DO width estimates. Assumed known
        self._NIS: list = []


        self._manager: Manager = setup_manager(self._params.IMM_off, self._params.single_target, self._params.visibility_off)

    def track(self, t: float, dt: float, true_do_states: list, ownship_state: np.ndarray) -> Tuple[list, list]:
        """Tracks/updates estimates on dynamic obstacles, based on sensor measurements
        generated from the input true dynamic obstacle states.

        Args:
            dt (float): Time since last update
            t (float): Current time (assumed >= 0)
            true_do_states (list): List of tuples of true dynamic obstacle indices and states (do_idx, [x, y, Vx, Vy], length, width) x n_do. Used for simulating sensor measurements.
            ownship_state (np.ndarray): Ownship state vector [x, y, Vx, Vy] used for simulating sensor measurements.

        Returns:
            Tuple[list, list]: List of updated dynamic obstacle tracks (ID, state, cov, length, width). Also, a list the sensor measurements used.
        """
        # Update tracker variables based on true states
        max_sensor_range = max([sensor.max_range for sensor in self.sensors]) #Find largest range among the sensors
        for do_idx, do_state, do_length, do_width in true_do_states: # Loop through every DO, and get their ID, state, lenght and width
            dist_ownship_to_do = np.linalg.norm(do_state[:2] - ownship_state[:2]) #Calculate the distance between ownship and DO
            if do_idx not in self._labels and dist_ownship_to_do < max_sensor_range: #Check if DO is not already detected and is closer than max sensor range
                # New track. TODO: Implement track initiation, e.g. n out of m based initiation.
                self._labels.append(do_idx) #Add DO-ID to tracker Labels
                self._track_initialized.append(False) #? Why are these false still?ownship
                self._track_terminated.append(False) #? Why is this false still?
                self._means.append(np.array([0,0,0,0]))
                self._covs.append(np.eye(5))
                self._length_upd.append(do_length) #Include the length of object into tracker
                self._width_upd.append(do_width) #Include the width of object into tracker
                self._NIS.append(np.nan) #Don't include NIS yet
            elif do_idx in self._labels:
                self._track_initialized[self._labels.index(do_idx)] = True #Set the target as initialized

        # print(self._params)

        
        sensor_measurements = []
        for sensor in self.sensors:
            z = sensor.generate_measurements(t, true_do_states, ownship_state)
            print("z = : ", z , "type: ", type(z))
            sensor_measurements.append(z)
            meas_covariance_NE = sensor._params.R
            clutter = sensor.generate_clutter(t, ownship_state)
            print("clutter = : ", clutter , "type: ", type(clutter))
            # print(sensor._params.to_dict())
        # meas_covariance_NE[0][0] = 10 # To see that the covariance comes out correct
        
        # TODO: Set this value via 
        meas_covariance_XY = np.asarray([[meas_covariance_NE[1][1], 0], [0, meas_covariance_NE[0][0]]])
        # print(meas_covariance_XY)
            
        # TODO: Specify whether the Tracker and the measurements should use the same Cov or not. (Edmund mentioned this) 
        # TODO: Add functionality for several sensors

        # print(sensor_measurements)
        # Apply changes to measurements here so that they will fit into the VIMMJIPDA
        # The measurements already have added noise
        """ The ownship position needs to be a construct.State object.
        This comes on the form Construct.State(Mean, Covariance, timestamp, ID)
        Here, mean is the position on the form: (E, V_E, N, V_N, 0). Where E = East, N = North, V = Velocity
        Covariance = np.Identity(4)
        Timestamp = t
        Id can be skipped

        """
        ownship_mean = np.asarray([ownship_state[1], ownship_state[3], ownship_state[0], ownship_state[2], 0])
        # print(ownship_mean)
        ownship_cov = np.identity(5)

        ownship_pos = State(ownship_mean, ownship_cov, t)


        """
        The measurement set needs to be a set containing construct.Measurement object
        This comes on the form Construct.Measurement(measurement, measurement_params['cart_cov'],  float(timestamp))
        Measurement is the position in xy-coordinates on the form (E,N)
        measurement_params['cart_cov'] is given in parameters
        timestamp is given
        """

        #TODO: Look into, might need if measurements
        sensor_measurement = set() #Look at import_data.py to see how to transform data
        # print(type(sensor_measurement))

        new_meas = False # Create a variable to see if we are on a timestep that matches with sensor measurement rates
        for sensor in sensor_measurements:
            for meas in sensor:
                # print(meas, type(meas), " Time: ", t)
                # for do_idx, do_state, do_length, do_width in true_do_states:
                #     print(do_state[0], do_state[1])
                if not np.isnan(meas[0]) and not np.isnan(meas[1]):
                    values = np.asarray([meas[1],meas[0]])
                    # TODO: Add Functionality to choose if Filter knows the measurement Cov or not
                    # sensor_measurement.add(Measurement(values, measurement_params['cart_cov'],  t)) # Choose this if Tracker should not know meas cov
                    sensor_measurement.add(Measurement(values, meas_covariance_XY,  t)) # Choose this if Tracker should know meas cov
        
        # Check if this is a timestep with measurements
        for sensor in self.sensors:
            # Loop through all DO
            for i, (_, xs, length, width) in enumerate(true_do_states):
                if ((t - sensor._prev_meas_time[i]) % (1 / sensor._params.measurement_rate) == 0):
                    new_meas = True
            
        # Run the VIMMJIPDA Tracker at the same rate as sensor measurement rates
        if new_meas:
            self._manager.step(sensor_measurement, float(t), ownship=ownship_pos)


        tracks = []
        # for track in self._manager.tracks:
        #     print("Timestep: ", t , " ",  track)

        for track in self._manager.tracks:
            if track.index > len(self._means):
                self._means.append(np.array([0,0,0,0]))
                print("Test1")
            if track.index > len(self._covs):
                self._covs.append(np.eye(4))
                print("Test2")
            if track.index > len(self._length_upd):
                self._length_upd.append(true_do_states[0][2])
            if track.index > len(self._width_upd):
                self._width_upd.append(true_do_states[0][3])


            # print(type(track))
            mean_xy, cov_xy = track.states.get_mean_covariance_array()
            # print(track.states.__len__())
            # print(track.states.leaves.get_mean_covariance_array())
            #print('\n mean: \n', mean_xy, type(mean_xy))
            #print('\n cov: \n', cov_xy, type(cov_xy))
            # print(cov_xy)
            mean_NE = np.array([mean_xy[0][2], mean_xy[0][0], mean_xy[0][3], mean_xy[0][1]])
            # TODO: Transform cov matrix to NE coordinates 
            # TODO: Create func to transform from XY to NE
            cov_NE = np.array([
                        [cov_xy[0][2][2], cov_xy[0][0][2], cov_xy[0][2][3] , cov_xy[0][2][1]],
                        [cov_xy[0][0][2], cov_xy[0][0][0], cov_xy[0][0][3], cov_xy[0][0][1]],
                        [cov_xy[0][2][3], cov_xy[0][0][3], cov_xy[0][3][3], cov_xy[0][3][1]],
                        [cov_xy[0][1][2], cov_xy[0][0][1], cov_xy[0][3][1], cov_xy[0][1][1]]
                ])
            # print(track.index)
            self._means[track.index - 1] = mean_NE
            self._covs[track.index - 1] = cov_NE
            # print(mean_NE, type(mean_NE), 'mean \n')

            
            # print('cov xy',cov_xy, type(cov_xy), '\n')
            # print('cov NE',cov_NE, type(cov_NE), '\n')
        # print(true_do_states, 'true states')
        # print(self._labels, 'labels')
        # print(self._labels, 'labels')
        #TODO: Move this into loop for more tracks than 1
            
        # print(self._labels, 'labels')  
        #TODO: Move this into loop for more tracks than 1
            
            tracks.append(
                (
                    track.index,
                    self._means[track.index - 1],
                    self._covs[track.index - 1],
                    self._length_upd[track.index - 1],
                    self._width_upd[track.index - 1]
                )
            )
        
        
        #Return tracks and sensor_measurements
        return tracks, sensor_measurements




    def get_track_information(self) -> Tuple[list, list]:
        tracks = []
        for track in self._manager.tracks:
            tracks.append(
                (
                    track.index,
                    self._means[track.index-1],
                    self._covs[track.index-1],
                    self._length_upd[track.index-1],
                    self._width_upd[track.index-1]
                )
            )
        return tracks, self._NIS
