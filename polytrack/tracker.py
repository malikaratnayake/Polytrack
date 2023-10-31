import os, sys
# import time
import cv2
import numpy as np
import itertools as it
import math
from PIL import Image
from ultralytics import YOLO
import csv
from scipy.optimize import linear_sum_assignment
import json

from polytrack.general import cal_dist


model = YOLO('./data/yolov8_models/yolov8s_best.pt')
class_names = model.names

class Config:
    """
    A class to store configuration parameters for the PolyTrack object tracker.

    Attributes:
        yolov8_confidence (float): Confidence threshold for YOLOv8 object detection.
        new_insect_confidence (float): Confidence threshold for detecting new insects.
        yolov8_iou_threshold (float): Intersection over union threshold for YOLOv8 object detection.
        bs_dl_iou_threshold (float): Intersection over union threshold for background subtraction and deep learning object detection.
        bs_max_interframe_distance (int): Maximum distance between object centroids for background subtraction object tracking.
        dl_max_interframe_distance (int): Maximum distance between object centroids for deep learning object tracking.
        knn_min_blob_area (int): Minimum blob area for KNN object tracking.
        knn_max_blob_area (int): Maximum blob area for KNN object tracking.
    """
    def __init__(
        self,
        yolov8_confidence: float,
        new_insect_confidence: float,
        yolov8_iou_threshold: float,
        bs_dl_iou_threshold: float,
        bs_max_interframe_distance: int,
        dl_max_interframe_distance: int,
        knn_min_blob_area: int,
        knn_max_blob_area: int
    ) -> None:
        """
        Initializes a Config object with the specified parameters.

        Args:
            yolov8_confidence (float): Confidence threshold for YOLOv8 object detection.
            new_insect_confidence (float): Confidence threshold for detecting new insects.
            yolov8_iou_threshold (float): Intersection over union threshold for YOLOv8 object detection.
            bs_dl_iou_threshold (float): Intersection over union threshold for background subtraction and deep learning object detection.
            bs_max_interframe_distance (int): Maximum distance between object centroids for background subtraction object tracking.
            dl_max_interframe_distance (int): Maximum distance between object centroids for deep learning object tracking.
            knn_min_blob_area (int): Minimum blob area for KNN object tracking.
            knn_max_blob_area (int): Maximum blob area for KNN object tracking.
        """
        self.yolov8_confidence = yolov8_confidence
        self.new_insect_confidence = new_insect_confidence
        self.yolov8_iou_threshold = yolov8_iou_threshold
        self.bs_dl_iou_threshold = bs_dl_iou_threshold
        self.bs_max_interframe_distance = bs_max_interframe_distance
        self.dl_max_interframe_distance = dl_max_interframe_distance
        self.knn_min_blob_area = knn_min_blob_area
        self.knn_max_blob_area = knn_max_blob_area


with open('./polytrack/config.json', "r") as f:
    __config_dict = json.load(f)
    CONFIG = Config(**__config_dict)


class DL_Detections():
    yolov8_confidence = CONFIG.yolov8_confidence
    iou_threshold = CONFIG.yolov8_iou_threshold

    def __init__(self) -> None:
        # Config.__init__(self, './polytrack/config.json')
        self.flower_class = self.__get_classes_to_detect(detect_flowers=True)
        self.insect_classes = self.__get_classes_to_detect(detect_flowers=False)

        return None

    def __run_deep_learning(self, _frame, detect_flowers: bool) -> np.ndarray:
            """
            Runs deep learning model on the input frame to detect flowers or insects.

            Args:
                _frame (np.ndarray): Input frame to run the model on.
                detect_flowers (bool): If True, detects flowers. If False, detects insects.

            Returns:
                np.ndarray: Array of detections in the format [xmin, ymin, xmax, ymax, class, confidence].
            """

            # More info: https://docs.ultralytics.com/modes/predict/#inference-arguments

            # Set the classes to detect based on the input parameter
            if detect_flowers:
                classes_to_detect = self.flower_class
            else:
                classes_to_detect = self.insect_classes

            # Run the model on the input frame with the specified parameters
            results = model.predict(source=_frame, conf=self.yolov8_confidence, show=False, verbose = False, iou = self.iou_threshold, classes = classes_to_detect)

            # Extract the classes, confidence scores, and bounding boxes from the results
            classes = results[0].boxes.cls
            conf = results[0].boxes.conf
            boxes = results[0].boxes.xyxy

            # Create array in the format [xmin, ymin, xmax, ymax, class, confidence]
            detections = np.zeros((len(classes), 6))
            detections[:, 0] = boxes[:, 0]
            detections[:, 1] = boxes[:, 1]
            detections[:, 2] = boxes[:, 2]
            detections[:, 3] = boxes[:, 3]
            detections[:, 4] = classes
            detections[:, 5] = conf

            return detections

    def __get_classes_to_detect(self, detect_flowers: bool) -> list:
            """
            Returns a list of classes to detect based on the value of detect_flowers.

            Args:
                detect_flowers (bool): A boolean value indicating whether or not to detect flowers.

            Returns:
                list: A list of classes to detect.
            """

            _class_list = []

            if detect_flowers:
                # _class_list = [key for key in class_names]
                _class_list = [1]
            else:
                for key, value in class_names.items():
                    if value != 'flower':
                        _class_list.append(key)
            
            return _class_list
    

    def get_deep_learning_detection(self, _frame, detect_flowers: bool) -> np.ndarray:

        _detections = self.__run_deep_learning(_frame, detect_flowers)

        if detect_flowers:
            processed_detections = self.__process_flower_results(_detections)
        else:
            processed_detections = self.__process_insect_results(_detections)

        return processed_detections
    
    def __process_flower_results(self, _results: np.ndarray) -> np.ndarray:
        # Get the center of gravity of the detected flower and radius of the bounding circle
        _flower_detection = np.zeros(shape=(0,5)) #(create an array to store data x,y,area, conf, type)

        for result in _results:
            mid_x = int((result[0] + result[2])/2)
            mid_y = int((result[1] + result[3])/2)
            radius = int(cal_dist(result[0], result[1], mid_x, mid_y)*math.cos(math.radians(45)))
            _flower_detection = np.vstack([_flower_detection,(float(mid_x), float(mid_y), float(radius), result[4], result[5])])

        return _flower_detection
    
    def __process_insect_results(self, _results: np.ndarray) -> np.ndarray:
        _insect_detection = np.zeros(shape=(0,5))

        for result in _results:
            mid_x = int((result[0] + result[2])/2)
            mid_y = int((result[1] + result[3])/2)
            area = int(abs((result[0] - result[2])*(result[1] - result[3])))
            _insect_detection = np.vstack([_insect_detection,(mid_x, mid_y, area, result[4], result[5])])

        return _insect_detection
    


class BS_Detections:
    # fgbg = cv2.createBackgroundSubtractorKNN()
    min_area = CONFIG.knn_min_blob_area
    max_area = CONFIG.knn_max_blob_area
    new_insect_confidence = CONFIG.new_insect_confidence

    def __init__(self) -> None:

        self.fgbg = cv2.createBackgroundSubtractorKNN()

        return None 
    
    def get_bs_detection(self, _frame) -> np.ndarray:

        _foreground_contours = self.__run_bs(_frame)

        _detections = self.__process_bs_results(_foreground_contours, self.min_area, self.max_area)

        return _detections
    
    def __run_bs(self, _frame) -> np.ndarray:

        _fgmask = self.fgbg.apply(_frame)

        _median = cv2.medianBlur(_fgmask,9)
        _kernel = np.ones((5,5),np.uint8)
        _processed_frame = cv2.erode(_median,_kernel,iterations = 1)
    
        _contours, _ = cv2.findContours(_processed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return _contours
    
    def __process_bs_results(self, contours: np.ndarray, 
                             min_area: int,
                             max_area: int) -> np.ndarray:
        
        _possible_insects = np.zeros(shape=(0,3))

        for c in contours:
            center_coord, _box_dims, _ = cv2.minAreaRect(c)
            _area = _box_dims[0] * _box_dims[1]

            if (_area > min_area) and (_area<max_area):
                _possible_insects = np.vstack([_possible_insects,(int(center_coord[0]),int(center_coord[1]),int(_area))])
            else:
                pass
        
        return _possible_insects
    

class InsectTracker(DL_Detections, BS_Detections):

    bs_max_interframe_distance = CONFIG.bs_max_interframe_distance
    dl_max_interframe_distance = CONFIG.dl_max_interframe_distance
    new_insect_condidence = CONFIG.new_insect_confidence
    bs_dl_iou_threshold = CONFIG.bs_dl_iou_threshold

    def __init__(self, video_info_filepath: str) -> None:
        DL_Detections.__init__(self)
        BS_Detections.__init__(self)

        # super().__init__()
        self.last_full_frame = None
        self.last_bs_associated_detections = None
        self.video_info_filepath = video_info_filepath
        self.video_frame_num, self.actual_frame_num, self.full_frame_num = self.get_video_info(self.video_info_filepath)
        self.actual_nframe = self.actual_frame_num[0]


    @staticmethod
    def get_video_info(_video_info_filepath:str) -> tuple:

        csv_file = str(_video_info_filepath) + 'video_info.csv'

        with open(csv_file, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)

            _video_frame_number_list = []
            _actual_frame_number_list = []
            _full_frame_number_list = []


            for row in csv_reader:
                _video_frame_number_list.append(int(row[0]))
                _actual_frame_number_list.append(int(row[1]))
                if row[2] != '':
                    _full_frame_number_list.append(int(row[2]))

        return _video_frame_number_list, _actual_frame_number_list, _full_frame_number_list
    
    @staticmethod
    def calculate_distance(x: float, y:float, px: float, py:float) -> int:

        edx = float(x) - float(px)
        edy = float(y) - float(py)
        error = np.sqrt(edx**2+edy**2)
    
        return int(error)
    
    # @staticmethod
    def map_frame_number(self, nframe: int) -> int:

        if nframe in self.video_frame_num:
            _frame_number_pos = self.video_frame_num.index(nframe)
            self.actual_nframe = self.actual_frame_num[_frame_number_pos]
        else:
            self.actual_nframe += 1

        return self.actual_nframe
    
    def preprocess_frame(self, _frame, _nframe: int):
        
        if _nframe in self.full_frame_num:
            self.last_full_frame = _frame
            _bs_frame = _frame
        else:
            _bs_frame = cv2.add(cv2.absdiff(_frame, self.last_full_frame), _frame)

        return _bs_frame

    def track(self, _frame, _nframe: int, predictions: np.array) -> np.ndarray:

        _bs_frame = self.preprocess_frame(_frame, _nframe)
        
        _bs_detections = self.get_bs_detection(_bs_frame)

        if len(_bs_detections) <= len(predictions):
            bs_associated_detections, bs_missing_insects, bs_unassociated_detections = self.__process_detections(_bs_detections, predictions, False)
            
            if bs_associated_detections.any():
                self.last_bs_associated_detections = bs_associated_detections
            else:
                pass

        else:
            bs_associated_detections, bs_unassociated_detections = [], []
            bs_missing_insects = [i[0] for i in predictions]

        run_deep_leaning = self.__verify_bs_detections(_bs_detections, bs_missing_insects, bs_unassociated_detections, predictions, _nframe)

        if run_deep_leaning:
            dl_predictions = np.zeros(shape=(0,3))
            for pred in np.arange(len(bs_missing_insects)):
                dl_predictions = np.vstack([dl_predictions,([row for row in predictions if bs_missing_insects[pred] == row[0]])])

            _dl_detections = self.get_deep_learning_detection(_frame, detect_flowers=False)

            
            dl_associated_detections, dl_missing_insects, potential_new_insects = self.__process_detections(_dl_detections, dl_predictions, True)

            if potential_new_insects.any():
                new_insects = self.__verify_new_insects(potential_new_insects, self.last_bs_associated_detections)
            else:
                new_insects = []

        else:
            dl_associated_detections, dl_missing_insects, new_insects = [], [], []

        return bs_associated_detections, dl_associated_detections, dl_missing_insects, new_insects
    
    def __remove_duplicate_detections(self, _dl_detections: np.ndarray, _bs_associated_detections: np.ndarray) -> np.ndarray:    
        _duplicate_detections = []
        for _bs_detection in _bs_associated_detections:
            _bs_bounding_box = [_bs_detection[1], _bs_detection[2], _bs_detection[3]]
            for _dl_detection in np.arange(len(_dl_detections)):
                _dl_bounding_box = [_dl_detections[_dl_detection][0], _dl_detections[_dl_detection][1], _dl_detections[_dl_detection][2]]

                # Calculate the intersection over union considering the bounding box of the detections

                _iou = self.calculate_iou(_bs_bounding_box, _dl_bounding_box)
                if _iou > self.bs_dl_iou_threshold:
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




    def __verify_new_insects(self, potential_new_insects: np.ndarray, bs_associated_detections: np.ndarray) -> list:

        if bs_associated_detections is None:
            bs_associated_detections = []
    
        potential_new_insects = self.__remove_duplicate_detections(potential_new_insects, bs_associated_detections)
        
        _low_confidence = []
        for _dl_detection in np.arange(len(potential_new_insects)):
            if potential_new_insects[_dl_detection][4] < self.new_insect_confidence:
                _low_confidence.append(_dl_detection)
            else:
               pass
        _new_insects = np.delete(potential_new_insects, _low_confidence, axis=0)

        return _new_insects


    
    def __verify_bs_detections(self, _bs_detections: np.ndarray, _missing_insects: list, _unassociated_detections: np.ndarray, _predictions: np.array, _nframe: int) -> bool:

        # If there are missing insects or unassociated detections or bs_detections, run deep learning
        if (len(_missing_insects) > 0) or (len(_unassociated_detections) > 0) or (len(_bs_detections) > len(_predictions)) or (_nframe in self.full_frame_num):
            run_deep_learning = True
        else:
            run_deep_learning = False

        return run_deep_learning
    
    

    def __process_detections(self, detections: np.array,
                              predictions: np.array,
                              dl_detected: bool) -> tuple:
    
        _max_distance_error, unassociated_array_length = self.__get_tracking_parameters(dl_detected)
        _missing_detections = []
        _unassociated_detections = np.zeros(shape=(0,unassociated_array_length))
        _associated_detections = np.zeros(shape=(0,6))

        _assignments = self.Hungarian_method(detections, predictions)
        _tracking_numbers = [i[0] for i in predictions]
        _num_of_objects_tracked = len(_tracking_numbers)
        
        for _unassociated in (_assignments[_num_of_objects_tracked:]):
            _unassociated_detections = np.vstack([_unassociated_detections,(detections[_unassociated])])       
                                
        
        for _object in np.arange(_num_of_objects_tracked):
            _object_num = _assignments[_object]

            if (_object_num < len(detections)):
                _center_x, _center_y, _area, _species, _confidence = self.__decode_detections(detections, _object_num)
                _distance_error = self.calculate_distance(_center_x,_center_y, predictions[_object][1], predictions[_object][2])
                if(_distance_error > _max_distance_error):
                    _missing_detections.append(predictions[_object][0])
                else:
                    _associated_detections = np.vstack([_associated_detections,(int(predictions[_object][0]),_center_x, _center_y, _area, _species, _confidence)])
            else:
                _missing_detections.append(predictions[_object][0])


        return _associated_detections, _missing_detections, _unassociated_detections
    

    def __decode_detections(self, detections: np.ndarray, insect_num: int):
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


    
    def Hungarian_method(self,_detections, _predictions):
        num_detections, num_predictions = len(_detections), len(_predictions)
        mat_shape = max(num_detections, num_predictions)
        hun_matrix = np.full((mat_shape, mat_shape),0)
        for p in np.arange(num_predictions):
            for d in np.arange(num_detections):
                hun_matrix[p][d] = cal_dist(_predictions[p][1],_predictions[p][2],_detections[d][0],_detections[d][1])
        
        row_ind, col_ind = linear_sum_assignment(hun_matrix)

        return col_ind  
    
    def __get_tracking_parameters(self, dl_detected: bool):



        if dl_detected:
            max_distance =  self.dl_max_interframe_distance
            unassocitaed_length = 5
        else:
            max_distance =  self.bs_max_interframe_distance
            unassocitaed_length = 3

        return max_distance, unassocitaed_length

    

    def cal_threshold_dist(self, _max_dist_dl,bs_mode):
        if bs_mode:
            threshold_dist = _max_dist_dl*1.5
        else:
            threshold_dist = _max_dist_dl*2
        
        return threshold_dist

    def low_confident_ass(self, _detections, _predictions,_max_dist_dl,_dist,bs_mode):
        
        threshold_dist = self.cal_threshold_dist(_max_dist_dl,bs_mode)
        
        if (len(_detections) == len(_predictions)) and (_dist <= threshold_dist) and pt_cfg.POLYTRACK.NEW_INSECT_MODE:
            pt_cfg.POLYTRACK.NEW_INSECT_MODE = False
            can_associate = True
        else:
            can_associate = False
        
        return can_associate
        





    
        
        
    





