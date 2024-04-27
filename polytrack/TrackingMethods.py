import os
import numpy as np
import csv
from scipy.optimize import linear_sum_assignment
from scipy.linalg import block_diag
import logging

LOGGER = logging.getLogger()



class ExtendedKalmanFilter:
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
    def __init__(self, initial_state, initial_covariance, process_noise_covariance, observation_noise_covariance):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise_covariance = process_noise_covariance
        self.observation_noise_covariance = observation_noise_covariance
    
    def predict(self, F):
        # Predict the next state using the state transition matrix F
        self.state = np.dot(F, self.state)
        # Predict the next covariance
        self.covariance = np.dot(F, np.dot(self.covariance, F.T)) + self.process_noise_covariance
    
    def update(self, measurement, H):
        # Kalman gain calculation
        K = np.dot(self.covariance, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(self.covariance, H.T)) + self.observation_noise_covariance)))
        # Update the state estimate
        self.state = self.state + np.dot(K, (measurement - np.dot(H, self.state)))
        # Update the covariance matrix
        self.covariance = np.dot((np.eye(self.covariance.shape[0]) - np.dot(K, H)), self.covariance)



class TrackingMethods(KalmanFilter, ExtendedKalmanFilter):

    def __init__(self,
                 prediction_method: str) -> None:
        
        self.prediction_method = prediction_method
        LOGGER.info(f"Prediction method: {self.prediction_method}")

        pass
    
    def calculate_distance(self, x: float, y: float, px: float, py: float) -> float:

        # Optimized calculation using vectorized operations
        squared_distance = np.square(x - px) + np.square(y - py)

        # Return distance as float for improved accuracy (if needed)
        return np.sqrt(squared_distance)
    
    def Hungarian_method(self, _detections, _predictions):
        num_detections, num_predictions = len(_detections), len(_predictions)
        mat_shape = max(num_detections, num_predictions)
        hun_matrix = np.full((mat_shape, mat_shape),0)
        for p in np.arange(num_predictions):
            for d in np.arange(num_detections):
                hun_matrix[p][d] = self.calculate_distance(_predictions[p][1],_predictions[p][2],_detections[d][0],_detections[d][1])
        
        row_ind, col_ind = linear_sum_assignment(hun_matrix)

        return col_ind
    
    def assign_detections_to_tracks(self,detections, predictions, cost_threshold=0.0):
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
    

    def get_compression_details(self,
                                video_filepath: str,
                                info_filename: str) -> tuple:
                
        if info_filename == '': info_filename = None

        if info_filename is not None:
            compression_details_file = os.path.join(video_filepath, os.path.splitext(info_filename)[0])
        else:
            compression_details_file = os.path.splitext(video_filepath)[0] +'_video_info.csv'


        with open(compression_details_file, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)

            video_frame_number_list = []
            actual_frame_number_list = []
            full_frame_number_list = []

            next(csv_reader)  # Skip the first row

            for row in csv_reader:
                video_frame_number_list.append(int(row[0]))
                actual_frame_number_list.append(int(row[1]))
                if row[2] != '':
                    full_frame_number_list.append(int(row[2]))

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

