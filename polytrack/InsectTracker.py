import os, sys
import cv2
import numpy as np
import math
from ultralytics import YOLO
import csv
from scipy.optimize import linear_sum_assignment
from scipy.linalg import block_diag
from polytrack.config import pt_cfg
from polytrack.utilities import Utilities

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



class Tracking_Methods(KalmanFilter, ExtendedKalmanFilter):

    def __init__(self,
                 prediction_method: str) -> None:
        
        self.prediction_method = prediction_method

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




class DL_Detector():

    def __init__(self,
                insect_detector: YOLO,
                insect_iou_threshold: float,
                dl_detection_confidence: float
                ) -> None:
        self.insect_detector = YOLO(insect_detector)
        # self.insect_classes = insect_classes
        self.insect_classes = [0,2,3]
        self.insect_iou_threshold = insect_iou_threshold
        self.dl_detection_confidence =dl_detection_confidence

        return None
    
    def _decode_DL_results(self, 
                           _results: np.ndarray) -> np.ndarray:
        
        # Extract the classes, confidence scores, and bounding boxes from the results
        _results_cpu = _results[0].boxes.cpu()
        classes = _results_cpu.cls
        conf = _results_cpu.conf
        boxes = _results_cpu.xyxy

        # Create array in the format [xmin, ymin, xmax, ymax, class, confidence]
        detections = np.zeros((len(classes), 6))
        detections[:, 0] = boxes[:, 0]
        detections[:, 1] = boxes[:, 1]
        detections[:, 2] = boxes[:, 2]
        detections[:, 3] = boxes[:, 3]
        detections[:, 4] = classes
        detections[:, 5] = conf

        return detections
    
    def __calculate_cog(self, 
                        _results: np.ndarray) -> np.ndarray:
        
        _insect_detection = np.zeros(shape=(0,5))

        for result in _results:
            mid_x = int((result[0] + result[2])/2)
            mid_y = int((result[1] + result[3])/2)
            area = int(abs((result[0] - result[2])*(result[1] - result[3])))
            _insect_detection = np.vstack([_insect_detection,(mid_x, mid_y, area, result[4], result[5])])

        return _insect_detection
        
    

    def run_dl_detector(self, 
                        frame: np.ndarray) -> np.ndarray:
        

        results = self.insect_detector.predict(source=frame, 
                                                conf=self.dl_detection_confidence, 
                                                show=False, 
                                                verbose = False, 
                                                iou = self.insect_iou_threshold, 
                                                classes = self.insect_classes)
        
        detections = self._decode_DL_results(results)
        processed_detections = self.__calculate_cog(detections)

        return processed_detections
    



class FGBG_Detector(Tracking_Methods):

    prev_frame = None
    last_full_frame = None

    def __init__(self,
                 min_blob_area: int,
                 max_blob_area: int,
                 downscale_factor: int,
                 dilate_kernel_size: int,
                 movement_threshold: int,
                 compressed_video: bool,
                 video_filepath: str,
                 info_filename:str,
                 prediction_method:str) -> None:
        
        super().__init__(prediction_method = prediction_method)
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.downscale_factor = downscale_factor
        self.dilate_kernel_size = dilate_kernel_size
        self.movement_threshold = movement_threshold
        self.compressed_video = compressed_video
        self.video_filepath = video_filepath
        self.info_filename = info_filename

        if self.compressed_video:
            self.video_frame_num, self.actual_frame_num, self.full_frame_num = self.get_compression_details(self.video_filepath, self.info_filename)

        downscaled_kernel_size = int(dilate_kernel_size / downscale_factor)
        self.dilation_kernel = np.ones((downscaled_kernel_size, downscaled_kernel_size))
        self.fx = self.fy = 1 / self.downscale_factor

        return None
    
    def preprocess_frame(self, 
                         frame: np.ndarray,
                         nframe: int) -> np.ndarray:

        if self.downscale_factor == 1:
            downscaled_frame = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
        else:
            downscaled_frame = cv2.resize(cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY), dsize=None, fx=self.fx, fy=self.fy)
            

        if nframe in self.full_frame_num:
            self.last_full_frame = downscaled_frame
            preprocessed_frame = downscaled_frame
        else:
            diff_frame = cv2.absdiff(downscaled_frame, self.last_full_frame)
            preprocessed_frame = cv2.add(downscaled_frame, diff_frame)

        return preprocessed_frame
    
    def detect_foreground_blobs(self,
                                frame: np.ndarray) -> np.ndarray:
        
        if self.prev_frame is None:
            self.prev_frame = frame
        
        # Compute pixel difference between consecutive frames 
        diff_frame = cv2.absdiff(frame, self.prev_frame)

        # Convert to grayscale
        dilated_frame = cv2.dilate(diff_frame, kernel=self.dilation_kernel)

        # Cut off pixels that did not have "enough" movement. This is now a 2D array of just 1s and 0s
        _, threshed_diff = cv2.threshold(src=dilated_frame, thresh=self.movement_threshold, maxval=255, type=cv2.THRESH_BINARY)

        frame_mask = cv2.medianBlur(cv2.dilate(threshed_diff, kernel=self.dilation_kernel), 31)

        self.prev_frame = frame

        return frame_mask
    

    def process_foreground_blobs(self, 
                                 contours: np.ndarray)-> np.ndarray:
        
        _possible_insects = np.zeros(shape=(0,3))

        for c in contours:
            cog_coord, _box_dims, _ = cv2.minAreaRect(c)
            cog_coord = [int(num * self.downscale_factor) for num in cog_coord]
            _box_dims = [int(num * self.downscale_factor) for num in _box_dims]
        
            _area = _box_dims[0] * _box_dims[1]

            if (_area > self.min_blob_area) and (_area<self.max_blob_area):
                _possible_insects = np.vstack([_possible_insects,(int(cog_coord[0]),int(cog_coord[1]),int(_area))])
            else:
                pass
        
        return _possible_insects
    

    def run_fgbg_detector(self,
                        frame: np.ndarray, 
                        nframe) -> np.ndarray:
        
        if self.preprocess_frame:
            frame = self.preprocess_frame(frame, nframe)

        foreground_blobs = self.detect_foreground_blobs(frame)

        contours, _ = cv2.findContours(foreground_blobs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detections = self.process_foreground_blobs(contours)

        return detections

class InsectTracker(DL_Detector, FGBG_Detector):

    def __init__(self,
                 insect_detector: YOLO,
                 insect_iou_threshold: float,
                 dl_detection_confidence: float,
                 min_blob_area: int,
                 max_blob_area: int,
                 downscale_factor: int,
                 dilate_kernel_size: int,
                 movement_threshold: int,
                 compressed_video: bool,
                 max_interframe_travel: int,
                 video_filepath: str,
                 info_filename: str,
                 iou_threshold: float,
                 model_insects_large: YOLO,
                 prediction_method: str) -> None:
        
        
        DL_Detector.__init__(self,
                             insect_detector = insect_detector,
                             insect_iou_threshold = insect_iou_threshold,
                             dl_detection_confidence = dl_detection_confidence)
        
        FGBG_Detector.__init__(self,        
                                min_blob_area = min_blob_area,
                                max_blob_area = max_blob_area,
                                downscale_factor = downscale_factor,
                                dilate_kernel_size = dilate_kernel_size,
                                movement_threshold = movement_threshold,
                                compressed_video = compressed_video,
                                video_filepath = video_filepath,
                                info_filename = info_filename,
                                prediction_method = prediction_method)

        self.predictions = []
        self.max_interframe_travel = max_interframe_travel
        self.compressed_video = compressed_video
        self.iou_threshold = iou_threshold
        self.model_insects_large = YOLO(model_insects_large)
        self.tracks = []
        self.insect_list = []
        
        return None
    

    def run_tracker(self, 
                    frame: np.ndarray,
                    nframe: int,
                    predictions: np.ndarray) -> np.ndarray:
        
        self.predictions = predictions
        
        fg_detections = self.run_fgbg_detector(frame, nframe)

        if len(fg_detections) <= len(self.predictions):
            fgbg_associated_detections, fgbg_missing_insects, fgbg_unassociated_detections = self.process_detections(fg_detections, 
                                                                                                                     self.predictions, 
                                                                                                                     dl_detections= False)
        else:
            fgbg_associated_detections, fgbg_unassociated_detections = [], []
            fgbg_missing_insects = [i[0] for i in self.predictions]


        if bool(fgbg_missing_insects) or bool(fgbg_unassociated_detections) or (len(fg_detections) > len(self.predictions)) or (len(fg_detections) == 0 and self.compressed_video) :
            dl_predictions = np.zeros(shape=(0,3))
            for pred in np.arange(len(fgbg_missing_insects)):
                dl_predictions = np.vstack([dl_predictions,([row for row in self.predictions if fgbg_missing_insects[pred] == row[0]])])

            if (nframe not in self.full_frame_num):
                dl_detections = self.run_dl_detector(frame)
            else:
                dl_detections = []

            dl_associated_detections, dl_missing_insects, potential_new_insects = self.process_detections(dl_detections, dl_predictions, dl_detections=True)

            if potential_new_insects.any():
                new_insects = self.verify_new_insects(frame, potential_new_insects, fgbg_associated_detections, fg_detections)
            else:
                new_insects = []

        else:
            dl_associated_detections, dl_missing_insects, new_insects = [], [], []


        return (fgbg_associated_detections, dl_associated_detections, dl_missing_insects, new_insects)
    

    def compound_detections(self,
                            fgbg_associated_detections: np.ndarray,
                            dl_associated_detections: np.ndarray,
                            new_insects: np.ndarray) -> np.ndarray:
        
        detections = np.zeros(shape=(0,3))

        for _insect in fgbg_associated_detections:
           detections = np.vstack([detections, _insect[0:3]])

        for _insect in dl_associated_detections:
            detections = np.vstack([detections, _insect[0:3]])

        for _insect in new_insects:
            if len(self.insect_list) > 0:
                new_insect_id = max(self.insect_list) + 1
            else:
                new_insect_id = 1
                self.insect_list.append(new_insect_id)

            _insect = np.insert(_insect, 0, new_insect_id)
            detections = np.vstack([detections, _insect[0:3]])
              
        return detections

        

    def verify_new_insects(self,
                            frame: np.ndarray,
                            potential_new_insects: np.ndarray,
                            fgbg_associated_detections: np.ndarray,
                            fg_detections: np.ndarray) -> list:
          
          if fgbg_associated_detections is None:
                fgbg_associated_detections = []
    
          if fg_detections is None:
                fg_detections = []
    
          potential_new_insects = self.remove_duplicate_detections(potential_new_insects, fgbg_associated_detections)
    
          low_confidence = []
          for dl_detection in np.arange(len(potential_new_insects)):
                mid_x = int(potential_new_insects[dl_detection][0])
                mid_y = int(potential_new_insects[dl_detection][1])
                insect_type = int(potential_new_insects[dl_detection][3])
    
                x0 = max(0, int(mid_x - 160))
                y0 = max(0, int(mid_y - 160))
                x1 = min(int(mid_x + 160), 1920)
                y1 = min(int(mid_y + 160), 1080)
    
                croped_frame = frame[y0:y1, x0:x1]
    
                black_frame = np.zeros((640,640,3), np.uint8)
                black_frame[200:200+croped_frame.shape[0], 200:200+croped_frame.shape[1]] = croped_frame
    
                crop = cv2.flip(black_frame, -1)

                dl_detection_confidences=[0.7,0.9,0.5, 0.5]
    
                confidence = dl_detection_confidences[insect_type]
    
                new_insect_results = self.model_insects_large.predict(source=crop, conf=confidence, show=False, verbose = False, iou = 0.5, classes = [insect_type], augment = True, imgsz = (640,640))
    
                new_insect_detections = self._decode_DL_results(new_insect_results)
    
                if len(new_insect_detections) == 0:
                    low_confidence.append(dl_detection)
                else:
                    pass
    
          new_insects = np.delete(potential_new_insects, low_confidence, axis=0)
    
          return new_insects
    


    def process_detections(self,
                            detections: np.array,
                            predictions: np.array,
                            dl_detections: bool) -> tuple:
          
        if dl_detections:
            max_interframe_travel_distance = self.max_interframe_travel[0]
            unassociated_array_length = 5
        else:
            max_interframe_travel_distance = self.max_interframe_travel[1]
            unassociated_array_length = 3
              
        missing_detections = []
        unassociated_detections = np.zeros(shape=(0,unassociated_array_length))
        associated_detections = np.zeros(shape=(0,6))
    
        assignments = self.Hungarian_method(detections, predictions)
        tracking_numbers = [i[0] for i in predictions]
        num_of_objects_tracked = len(tracking_numbers)
          
        for _unassociated in (assignments[num_of_objects_tracked:]):
            unassociated_detections = np.vstack([unassociated_detections,(detections[_unassociated])])       
                                  
          
        for _object in np.arange(num_of_objects_tracked):
            _object_num = assignments[_object]
    
            if (_object_num < len(detections)):
                _center_x, _center_y, _area, _species, _confidence = self.decode_detections(detections, _object_num)
                _distance_error = self.calculate_distance(_center_x,_center_y, predictions[_object][1], predictions[_object][2])
                if(_distance_error > max_interframe_travel_distance):
                    missing_detections.append(predictions[_object][0])
                else:
                    associated_detections = np.vstack([associated_detections,(int(predictions[_object][0]),_center_x, _center_y, _area, _species, _confidence)])
            else:
                missing_detections.append(predictions[_object][0])


        return associated_detections, missing_detections, unassociated_detections
    
    def decode_detections(self, 
                          detections: np.ndarray, 
                          insect_num: int):
        
        _center_x = int(detections[insect_num][0])
        _center_y = int(detections[insect_num][1])
        _area = int(detections[insect_num][2])

        if len(detections[insect_num]) > 3:
            _species = detections[insect_num][3]
            _confidence = detections[insect_num][4]
        else:
            _species = 0
            _confidence = 0

        return _center_x, _center_y, _area, _species, _confidence
    

    def remove_duplicate_detections(self, _dl_detections: np.ndarray, _bs_associated_detections: np.ndarray) -> np.ndarray:    
        _duplicate_detections = []
        for _bs_detection in _bs_associated_detections:
            _bs_bounding_box = [_bs_detection[1], _bs_detection[2], _bs_detection[3]]
            for _dl_detection in np.arange(len(_dl_detections)):
                _dl_bounding_box = [_dl_detections[_dl_detection][0], _dl_detections[_dl_detection][1], _dl_detections[_dl_detection][2]]

                # Calculate the intersection over union considering the bounding box of the detections

                _iou = self.calculate_iou(_bs_bounding_box, _dl_bounding_box)
                if _iou > self.iou_threshold:
                    _duplicate_detections.append(_dl_detection)
                else:
                    pass
        
        _dl_detections_cleaned = np.delete(_dl_detections, _duplicate_detections, axis=0)

        return _dl_detections_cleaned
    

    def calculate_iou(self, bs_bounding_box: np.array, dl_bounding_box: np.array) -> float:
        """Calculates the intersection over union of two bounding boxes.

        Args:
            bbox1: A list of four floats, representing the top-left and bottom-right
            coordinates of the first bounding box.
            bbox2: A list of four floats, representing the top-left and bottom-right
            coordinates of the second bounding box.

        Returns:
            A float, representing the intersection over union of the two bounding boxes.
        """
        # Calculate the length of a side using the area
        bs_box_side = int(np.sqrt(dl_bounding_box[2])/2)

        # define bbox1
        bbox1 = np.zeros(shape=(4))
        bbox2 = np.zeros(shape=(4))

        # Calculate the top-left and bottom-right coordinates of the first bounding box.        
        bbox1[0] = int(bs_bounding_box[0]) - bs_box_side
        bbox1[2] = int(bs_bounding_box[0]) + bs_box_side
        bbox1[1] = int(bs_bounding_box[1]) - bs_box_side
        bbox1[3] = int(bs_bounding_box[1]) + bs_box_side

        # Calculate the top-left and bottom-right coordinates of the second bounding box.
        bbox2[0] = int(dl_bounding_box[0]) - bs_box_side
        bbox2[2] = int(dl_bounding_box[0]) + bs_box_side
        bbox2[1] = int(dl_bounding_box[1]) - bs_box_side
        bbox2[3] = int(dl_bounding_box[1]) + bs_box_side


        # Calculate the intersection area of the two bounding boxes.
        intersection_area = np.maximum(0, min(bbox2[2], bbox1[2]) - max(bbox2[0], bbox1[0])) * \
                            np.maximum(0, min(bbox2[3], bbox1[3]) - max(bbox2[1], bbox1[1]))

        # Calculate the union area of the two bounding boxes.
        union_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) + \
                    (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) - \
                    intersection_area

        # Calculate the intersection over union.
        iou = intersection_area / union_area

        return iou



    