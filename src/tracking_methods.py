"""
tracking_methods.py
===================
Core tracking algorithms and motion prediction for Polytrack.

This module provides:
  - ExtendedKalmanFilter  – non-linear Kalman filter (position + velocity state)
  - KalmanFilter          – standard linear Kalman filter
  - TrackingMethods       – mixin that bundles prediction, assignment, and
                            compressed-video frame-mapping utilities

The class hierarchy is deliberately kept flat so that InsectTracker and
FlowerTracker can compose these capabilities through multiple inheritance
without carrying heavy per-instance state here.
"""

import os
import numpy as np
import csv
from scipy.optimize import linear_sum_assignment
import logging

LOGGER = logging.getLogger()


class ExtendedKalmanFilter:
    """
    Non-linear Extended Kalman Filter for 2-D position tracking.

    State vector: [x, y, vx, vy]
    Measurement:  [x, y]

    Although the state-transition model used here is linear (constant
    velocity), the EKF formulation is retained so that a non-linear
    transition can be substituted without changing the interface.
    """

    def __init__(self, initial_state, initial_covariance, process_noise, observation_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise = process_noise
        self.observation_noise = observation_noise
    
    def predict(self, dt):
        # State transition function (non-linear)
        x, y, vx, vy = self.state
        predicted_state = np.array([
            x + vx * dt,
            y + vy * dt,
            vx,
            vy
        ])
        
        # State transition Jacobian (partial derivatives of state transition function)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Predict next state using non-linear state transition
        self.state = predicted_state
        self.covariance = np.dot(F, np.dot(self.covariance, F.T)) + self.process_noise
    
    def update(self, measurement):
        # Measurement function (non-linear)
        x, y, vx, vy = self.state
        predicted_measurement = np.array([x, y])
        
        # Measurement Jacobian (partial derivatives of measurement function)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Kalman gain calculation
        S = np.dot(H, np.dot(self.covariance, H.T)) + self.observation_noise
        K = np.dot(self.covariance, np.dot(H.T, np.linalg.inv(S)))
        
        # Update state estimate using non-linear measurement update
        self.state = self.state + np.dot(K, (measurement - predicted_measurement))
        self.covariance = np.dot((np.eye(self.covariance.shape[0]) - np.dot(K, H)), self.covariance)



class KalmanFilter:
    """
    Standard linear Kalman Filter for 2-D position tracking.

    State vector: [x, y, vx, vy]
    Measurement:  [x, y]
    """

    def __init__(self, initial_state, initial_covariance, process_noise_covariance, observation_noise_covariance):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise_covariance = process_noise_covariance
        self.observation_noise_covariance = observation_noise_covariance

    def predict(self, F):
        """Propagate state and covariance forward one time step using transition matrix F."""
        self.state = np.dot(F, self.state)
        self.covariance = np.dot(F, np.dot(self.covariance, F.T)) + self.process_noise_covariance

    def update(self, measurement, H):
        """Correct the predicted state with a new measurement via the Kalman gain."""
        S = np.dot(H, np.dot(self.covariance, H.T)) + self.observation_noise_covariance
        K = np.dot(self.covariance, np.dot(H.T, np.linalg.inv(S)))
        self.state = self.state + np.dot(K, (measurement - np.dot(H, self.state)))
        self.covariance = np.dot((np.eye(self.covariance.shape[0]) - np.dot(K, H)), self.covariance)



class TrackingMethods(KalmanFilter, ExtendedKalmanFilter):
    """
    Mixin providing prediction and assignment algorithms used by both
    InsectTracker and FlowerTracker.

    Supported prediction methods (set via config):
      - "ConstantVelocity"  (default) – simple linear extrapolation
      - "Kalman"            – linear Kalman filter
      - "ExtendedKalman"    – extended Kalman filter

    Supported assignment methods:
      - "HungarianMethod"   (default) – optimal linear sum assignment
      - "ABP"               – greedy association by proximity
    """

    def __init__(self, prediction_method: str) -> None:
        # prediction_method arrives as a list from YAML config; take the first entry.
        try:
            self.prediction_method = prediction_method[0]
        except (TypeError, IndexError, AttributeError):
            # Malformed config — fall back to a safe default rather than crashing.
            LOGGER.warning(
                "Could not read prediction_method from config; defaulting to 'ConstantVelocity'."
            )
            self.prediction_method = "ConstantVelocity"

        # actual_nframe tracks the de-compressed frame counter used throughout
        # the pipeline. It must be initialised here so that map_frame_number()
        # can safely increment it (self.actual_nframe += 1) before the first
        # successful frame-number lookup in a compressed video.
        self.actual_nframe: int = 0

        LOGGER.info(f"Prediction method: {self.prediction_method}")
    
    def calculate_distance(self, x: float, y: float, px: float, py: float) -> float:

        # Optimized calculation using vectorized operations
        squared_distance = np.square(x - px) + np.square(y - py)

        # Return distance as float for improved accuracy (if needed)
        return np.sqrt(squared_distance)
    
    def Hungarian_method(self, detections, predictions, cost_threshold=None):
        num_detections, num_predictions = len(detections), len(predictions)
        if num_detections == 0 or num_predictions == 0:
            return []

        cost_matrix = np.zeros((num_predictions, num_detections))
        for p in np.arange(num_predictions):
            for d in np.arange(num_detections):
                cost_matrix[p][d] = self.calculate_distance(
                    predictions[p][1],
                    predictions[p][2],
                    detections[d][0],
                    detections[d][1],
                )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = []
        for pred_idx, det_idx in zip(row_ind, col_ind):
            if cost_threshold is None or cost_matrix[pred_idx][det_idx] <= cost_threshold:
                assignments.append((det_idx, pred_idx))

        return assignments
    
    def hungarian_assignment(self,detections, predictions, cost_threshold=0.0):
        """
        Assigns detections to predicted tracks using the Hungarian algorithm.

        Args:
            detections (list): List of detection coordinates (2D numpy array).
            predictions (list): List of predicted track coordinates (2D numpy array).
            cost_threshold (float): Cost threshold to filter assignments.

        Returns:
            list: List of tuples representing assignments (detection_index, prediction_index).
        """

        # Calculate pairwise distances (or costs) between detections and predictions
        num_detections = len(detections)
        num_predictions = len(predictions)
        cost_matrix = np.full((num_detections, num_predictions),0)

        for i in range(num_detections):
            for j in range(num_predictions):
                # Calculate Euclidean distance between detection[i] and prediction[j]
                cost_matrix[i][j] = np.linalg.norm(np.array([detections[i][0],detections[i][1]]) - np.array([predictions[j][1],predictions[j][2]]))

        # Use the Hungarian algorithm to find the optimal assignments
        detection_indices, prediction_indices = linear_sum_assignment(cost_matrix)

        # Filter assignments based on cost threshold
        assignments = []
        for det_idx, pred_idx in zip(detection_indices, prediction_indices):
            if cost_matrix[det_idx][pred_idx] <= cost_threshold:
                assignments.append((det_idx, pred_idx))

        return assignments
    
    def assign_by_proximity(self, detections, predictions, cost_threshold=0.0):
        """
        Assigns detections to predicted tracks using association by proximity.

        Args:
            detections (list): List of detection coordinates (2D numpy array).
            predictions (list): List of predicted track coordinates (2D numpy array).
            cost_threshold (float): Cost threshold to filter assignments.

        Returns:
            list: List of tuples representing assignments (detection_index, prediction_index).
        """

        num_detections = len(detections)
        num_predictions = len(predictions)

        if num_detections == 0 or num_predictions == 0:
            return []

        cost_matrix = np.full((num_detections, num_predictions), np.inf)

        LOGGER.debug(f"Number of detections: {num_detections}, Number of predictions: {num_predictions}")

        # Calculate pairwise distances (or costs) between detections and predictions
        for i in range(num_detections):
            for j in range(num_predictions):
                # Calculate Euclidean distance between detection[i] and prediction[j]
                cost_matrix[i][j] = np.linalg.norm(
                    np.array([detections[i][0], detections[i][1]]) - np.array([predictions[j][1], predictions[j][2]])
                )

        assignments = []
        unmatched_detections = list(range(num_detections))
        unmatched_predictions = list(range(num_predictions))

        while True:
            # Check if all elements in the cost_matrix are np.inf
            if np.all(cost_matrix == np.inf):
                break

            # Find the minimum value in the cost_matrix
            min_dist_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            min_dist = cost_matrix[min_dist_idx]

            # If the minimum distance is greater than the threshold, stop the assignment process
            if min_dist > cost_threshold:
                break

            det_idx, pred_idx = min_dist_idx
            assignments.append((det_idx, pred_idx))

            # Remove the assigned detection and prediction from further consideration
            unmatched_detections.remove(det_idx)
            unmatched_predictions.remove(pred_idx)

            # Invalidate this match to prevent it from being reused
            cost_matrix[det_idx, :] = np.inf
            cost_matrix[:, pred_idx] = np.inf

        LOGGER.debug(f"Assignments: {assignments}")
        LOGGER.debug(f"Unmatched detections: {unmatched_detections}")
        LOGGER.debug(f"Unmatched predictions: {unmatched_predictions}")

        return assignments
    


    
    def map_frame_number(self, nframe: int, compressed_video:bool) -> int:

        if compressed_video:
            if nframe in self.video_frame_num:
                _frame_number_pos = self.video_frame_num.index(nframe)
                self.actual_nframe = self.actual_frame_num[_frame_number_pos]
            else:
                self.actual_nframe += 1

        else:
            self.actual_nframe = nframe

        return self.actual_nframe
    

    def get_compression_details(self, video_filepath: str, info_filename: str) -> tuple:
        """
        Parse the CSV sidecar that maps compressed-video frame indices to their
        original (uncompressed) frame numbers.

        Expected CSV columns (no header row is consumed):
          col 0 – video frame number  (int)
          col 1 – actual frame number (int)
          col 2 – full-frame flag     (int, may be empty)

        Args:
            video_filepath: Path to the video file *or* the directory that
                            contains the sidecar when info_filename is given.
            info_filename:  Name of the sidecar CSV.  Pass an empty string or
                            None to use the default "<video_stem>_video_info.csv"
                            naming convention.

        Returns:
            (video_frame_number_list, actual_frame_number_list, full_frame_number_list)

        Raises:
            FileNotFoundError: If the sidecar CSV cannot be found.
            ValueError: If the CSV is empty or cannot be parsed.
        """
        if not info_filename:
            info_filename = None

        if info_filename is not None:
            compression_details_file = os.path.join(
                video_filepath, os.path.splitext(info_filename)[0]
            )
        else:
            compression_details_file = os.path.splitext(video_filepath)[0] + "_video_info.csv"

        if not os.path.isfile(compression_details_file):
            raise FileNotFoundError(
                f"Compression sidecar not found: '{compression_details_file}'. "
                "Check that source.compression_info points to the correct file, "
                "or set compressed_video: False in the config."
            )

        video_frame_number_list: list[int] = []
        actual_frame_number_list: list[int] = []
        full_frame_number_list: list[int] = []

        try:
            with open(compression_details_file, "r", encoding="utf-8") as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)  # skip header row
                for row in csv_reader:
                    video_frame_number_list.append(int(row[0]))
                    actual_frame_number_list.append(int(row[1]))
                    if len(row) > 2 and row[2] != "":
                        full_frame_number_list.append(int(row[2]))
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse compression sidecar '{compression_details_file}': {exc}"
            ) from exc

        if not video_frame_number_list:
            raise ValueError(
                f"Compression sidecar '{compression_details_file}' contains no data rows."
            )

        LOGGER.info(
            f"Loaded compression details: {len(video_frame_number_list)} frames, "
            f"{len(full_frame_number_list)} full frames."
        )
        return video_frame_number_list, actual_frame_number_list, full_frame_number_list
    

    def predict_next(self, for_predictions):
        predicted = []

        if self.prediction_method == 'Kalman':

            # Kalman filter setup parameters
            initial_state = np.array([0, 0, 0, 0])  # Initial state: [x, y, vx, vy]
            initial_covariance = np.eye(4) * 1000  # Initial covariance matrix
            process_noise_covariance = np.eye(4) * 0.1  # Process noise covariance (increased for robustness)
            observation_noise_covariance = np.eye(2) * 30  # Observation noise covariance (increased for robustness)
            
            # State transition matrix (constant velocity model)
            F = np.array([[1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
            
            # Measurement matrix (to extract position from state)
            H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
            
            # Create a Kalman filter instance
            kalman_filter = KalmanFilter(initial_state, initial_covariance, process_noise_covariance, observation_noise_covariance)
            
            for insect in for_predictions:
                insect_num = insect[0]
                x0, y0 = float(insect[1]), float(insect[2])  # Position at t-1
                x1, y1 = float(insect[3]), float(insect[4])  # Position at t-2
                
                # Calculate initial velocity (assuming constant velocity model)
                vx = x0 - x1
                vy = y0 - y1
                
                # Set initial state for the Kalman filter
                kalman_filter.state = np.array([x0, y0, vx, vy])
                
                # Predict next state using Kalman filter (1 time step prediction)
                kalman_filter.predict(F)
                
                # Get predicted position from the updated state estimate
                predicted_x, predicted_y = kalman_filter.state[0], kalman_filter.state[1]
                
                # Append predicted result to output list
                predicted.append([insect_num, predicted_x, predicted_y])
        

        elif self.prediction_method == 'ExtendedKalman':
            
            # Extended Kalman filter setup parameters
            initial_state = np.array([0, 0, 0, 0])  # Initial state: [x, y, vx, vy]
            initial_covariance = np.eye(4) * 1000  # Initial covariance matrix
            process_noise = np.eye(4) * 0.01  # Process noise covariance
            
            # Increase process noise for velocity to adapt to changes
            process_noise[2, 2] = 0.1  # Variance of velocity in x-direction
            process_noise[3, 3] = 0.1  # Variance of velocity in y-direction
            
            observation_noise = np.eye(2) * 0.1  # Observation noise covariance
            
            # Create an Extended Kalman filter instance
            ekf = ExtendedKalmanFilter(initial_state, initial_covariance, process_noise, observation_noise)
            
            for insect in for_predictions:
                insect_num = insect[0]
                x0, y0 = float(insect[1]), float(insect[2])  # Position at t-1
                x1, y1 = float(insect[3]), float(insect[4])  # Position at t-2
                
                # Calculate initial velocity (assuming constant velocity model)
                vx = x0 - x1
                vy = y0 - y1
                
                # Set initial state for the Extended Kalman filter [x, y, vx, vy]
                ekf.state = np.array([x0, y0, vx, vy])
                
                # Time step (assuming constant time interval between measurements)
                dt = 1.0  # Adjust this based on the time interval between measurements
                
                # Predict next state using Extended Kalman filter
                ekf.predict(dt)
                
                # Get predicted position from the updated state estimate
                predicted_x, predicted_y = ekf.state[0], ekf.state[1]
                
                # Append predicted result to output list
                predicted.append([insect_num, predicted_x, predicted_y])

        else:

            for _insect in for_predictions:
                _insect_num = _insect[0]
                _x0 = float(_insect[1])
                _y0 = float(_insect[2])
                _x1 = float(_insect[3])
                _y1 = float(_insect[4])
                
                    
                Dk1 = np.transpose([_x0, _y0])
                Dk2 = np.transpose([_x1, _y1])
                A = [[2,0,-1,0],  [0,2,0,-1]]
                Dkc = np.concatenate((Dk1,Dk2))
                
        #         print(Dk1,Dk2,Dkc)
                Pk = np.dot(A,Dkc.T)
                
                predicted.append([_insect_num, Pk[0], Pk[1]])

        return predicted
