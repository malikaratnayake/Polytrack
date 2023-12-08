import os, sys
import cv2
import numpy as np
import math
from ultralytics import YOLO
import csv
from scipy.optimize import linear_sum_assignment
from polytrack.config import pt_cfg
from polytrack.utilities import Utilities
import random
# from polytrack.general import cal_dist


model_flowers = YOLO(pt_cfg.POLYTRACK.FLOWER_MODEL)
model_insects = YOLO(pt_cfg.POLYTRACK.INSECT_MODEL)
if pt_cfg.POLYTRACK.INSECT_MODEL_LARGE is not None:
    model_insects_large = YOLO(pt_cfg.POLYTRACK.INSECT_MODEL_LARGE)
else:
    model_insects_large = model_insects

class_names = model_insects.names
TrackUtilities = Utilities()

class DL_Detections():
    yolov8_confidence = pt_cfg.POLYTRACK.YOLOV8_CONFIDENCE
    iou_threshold = pt_cfg.POLYTRACK.DL_IOU_THRESHOLD

    def __init__(self) -> None:
        # Config.__init__(self, './polytrack/config.json')
        self.flower_class = self.__get_classes_to_detect(detect_flowers=True)
        self.insect_classes = self.__get_classes_to_detect(detect_flowers=False)

        return None

    def __run_deep_learning(self, _frame, audit_frame,detect_flowers: bool) -> np.ndarray:
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
                yolov8_model = model_flowers

            elif audit_frame:
                classes_to_detect = self.insect_classes
                yolov8_model = model_insects_large

            else:
                classes_to_detect = self.insect_classes
                yolov8_model = model_insects
                

            results = yolov8_model.predict(source=_frame, conf=self.yolov8_confidence, show=False, verbose = False, iou = self.iou_threshold, classes = classes_to_detect)
            detections = self._decode_DL_results(results)

            return detections
    
    def _decode_DL_results(self, _results: np.ndarray) -> np.ndarray:
        
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
    

    def get_deep_learning_detection(self, _frame, audit_frame, detect_flowers: bool) -> np.ndarray:

        _detections = self.__run_deep_learning(_frame, audit_frame ,detect_flowers)

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
            radius = int(TrackUtilities.cal_dist(result[0], result[1], mid_x, mid_y)*math.cos(math.radians(45)))
            _flower_detection = np.vstack([_flower_detection,(int(mid_x), int(mid_y), int(radius), result[4], result[5])])

        return _flower_detection
    
    def __process_insect_results(self, _results: np.ndarray) -> np.ndarray:
        _insect_detection = np.zeros(shape=(0,5))

        for result in _results:
            # print("result", result)
            mid_x = int((result[0] + result[2])/2)
            mid_y = int((result[1] + result[3])/2)
            area = int(abs((result[0] - result[2])*(result[1] - result[3])))
            _insect_detection = np.vstack([_insect_detection,(mid_x, mid_y, area, result[4], result[5])])

        # print("insect detection: ", _insect_detection)

        return _insect_detection
    


class BS_Detections:
    # fgbg = cv2.createBackgroundSubtractorKNN()
    min_area = pt_cfg.POLYTRACK.MIN_INSECT_AREA
    max_area = pt_cfg.POLYTRACK.MAX_INSECT_AREA
    new_insect_confidence = pt_cfg.POLYTRACK.NEW_INSECT_CONFIDENCE
    

    def __init__(self) -> None:

        self.fgbg = cv2.createBackgroundSubtractorKNN()
        self.prev_frame = None

        return None 
    
    def reset_bg_model(self):
        self.fgbg = cv2.createBackgroundSubtractorKNN()
        return None
    
    def get_bs_detection(self, _frame) -> np.ndarray:

        _foreground_contours = self.__run_bs(_frame)

        _detections = self.__process_bs_results(_foreground_contours, self.min_area, self.max_area)

        return _detections
    
    def process_fgbg_output(self, _fgmask) -> np.ndarray:
        _median = cv2.medianBlur(_fgmask,9)
        _kernel = np.ones((5,5),np.uint8)
        
        _eroded_frame = cv2.erode(_median,_kernel,iterations = 1)
        _, threshed_diff = cv2.threshold(src=_eroded_frame, thresh=200 , maxval=255, type=cv2.THRESH_BINARY)
        # Dilate
        _processed_frame = cv2.dilate(threshed_diff,_kernel,iterations = 1)

        # cv2.imshow('frame',_processed_frame)

        return _processed_frame
    
    def calculate_diff(self, _frame):
        _bg_frame =  cv2.cvtColor(_frame,  cv2.COLOR_BGR2GRAY)

        if self.prev_frame is not None:
            diff = cv2.absdiff(_bg_frame, self.prev_frame)
            # Convert to grayscale
            


            # # Cut off pixels that did not have "enough" movement. This is now a 2D array of just 1s and 0s
            _, threshed_diff = cv2.threshold(src=diff, thresh=100 , maxval=255, type=cv2.THRESH_BINARY)
            gray_frame = cv2.dilate(threshed_diff, kernel=np.ones((5, 5)))

            # mask = cv2.medianBlur(cv2.dilate(threshed_diff, kernel=self.dilation_kernel), 9)

        else:
            gray_frame = None

        self.prev_frame = _bg_frame


        return gray_frame



    
    def __run_bs(self, _frame) -> np.ndarray:

        # _fgmask = self.calculate_diff(_frame)

        # if _fgmask is None:
        _fgmask = self.fgbg.apply(_frame, learningRate=0.1)

        _processed_frame = self.process_fgbg_output(_fgmask)
        
        # cv2.imshow('frame',_processed_frame)

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

    bs_max_interframe_distance = pt_cfg.POLYTRACK.MAX_DIST_BS
    dl_max_interframe_distance = pt_cfg.POLYTRACK.MAX_DIST_DL
    new_insect_condidence = pt_cfg.POLYTRACK.NEW_INSECT_CONFIDENCE
    bs_dl_iou_threshold = pt_cfg.POLYTRACK.BS_DL_IOU_THRESHOLD

    def __init__(self) -> None:
        DL_Detections.__init__(self)
        BS_Detections.__init__(self)

        # super().__init__()
        self.last_full_frame = None
        self.last_bs_associated_detections = None
        self.compressed_video = pt_cfg.POLYTRACK.COMPRESSED_VIDEO
        # if self.compressed_video:
        self.video_frame_num, self.actual_frame_num, self.full_frame_num = None, None, None
        self.actual_nframe = None
        # else:
        #     pass

    def reset(self):
        self.__init__()

        return None


    def get_video_info(self,_video_info_filepath:str, _video_name: str) -> tuple:

        # try:

        _video_info_file = os.path.join(_video_info_filepath, os.path.splitext(_video_name)[0])

        csv_file = str(_video_info_file) + '_video_info.csv'

        with open(csv_file, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)

            _video_frame_number_list = []
            _actual_frame_number_list = []
            _full_frame_number_list = []

            next(csv_reader)  # Skip the first row

            for row in csv_reader:
                _video_frame_number_list.append(int(row[0]))
                _actual_frame_number_list.append(int(row[1]))
                if row[2] != '':
                    _full_frame_number_list.append(int(row[2]))

        self.video_frame_num, self.actual_frame_num, self.full_frame_num = _video_frame_number_list, _actual_frame_number_list, _full_frame_number_list
        self.actual_nframe = _actual_frame_number_list[0]

        return _video_frame_number_list, _actual_frame_number_list, _full_frame_number_list
        
        # except FileNotFoundError:
        #     print('Video info file not found. Please check the path and try again.')
        #     sys.exit(1)
    
    @staticmethod
    def calculate_distance(x: float, y:float, px: float, py:float) -> int:

        edx = float(x) - float(px)
        edy = float(y) - float(py)
        error = np.sqrt(edx**2+edy**2)
    
        return int(error)
    
    # @staticmethod
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
    
    def preprocess_frame(self, __frame, _nframe: int):

        if _nframe in self.full_frame_num:
            self.last_full_frame = __frame
            _bs_frame = __frame
        else:
            _bs_frame = cv2.add(cv2.absdiff(__frame, self.last_full_frame), __frame)

        return _bs_frame

    def track(self, _compressed_video, _frame, _nframe, audit_frame ,predictions):

        _dl_frame = _frame.copy()
        
        if _compressed_video:
            _bs_frame = self.preprocess_frame(_frame, _nframe)
        else:
            _bs_frame = _frame

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
                
            _dl_detections = self.get_deep_learning_detection(_dl_frame, audit_frame ,detect_flowers=False)
            
            dl_associated_detections, dl_missing_insects, potential_new_insects = self.__process_detections(_dl_detections, dl_predictions, True)

            if potential_new_insects.any():
                new_insects = self.__verify_new_insects(_frame, potential_new_insects, self.last_bs_associated_detections)
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
    

    @staticmethod
    def predict_next(_for_predictions):
        
        _predicted = []
        for _insect in _for_predictions:
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
            
            _predicted.append([_insect_num, Pk[0], Pk[1]])
            
        
        return _predicted




    def __verify_new_insects(self, _frame, potential_new_insects: np.ndarray, bs_associated_detections: np.ndarray) -> list:

        if bs_associated_detections is None:
            bs_associated_detections = []

        potential_new_insects = self.__remove_duplicate_detections(potential_new_insects, bs_associated_detections)
        
        _low_confidence = []
        for _dl_detection in np.arange(len(potential_new_insects)):            

            #Get the midpoint of the bounding box
            _mid_x = int(potential_new_insects[_dl_detection][0])
            _mid_y = int(potential_new_insects[_dl_detection][1])
            _insect_type = int(potential_new_insects[_dl_detection][3])

            #Get the coordinates of top left and bottom right of the bounding box
            _x0 = max(0, int(_mid_x - 160))
            _y0 = max(0, int(_mid_y - 160))
            _x1 = min(int(_mid_x + 160), 1920)
            _y1 = min(int(_mid_y + 160), 1080)
        

            #Crop the bounding box from the frame
            _croped_frame = _frame[_y0:_y1, _x0:_x1]

            # Expand the cropped bounding box to a 640x640 frame with the cropped bounding box in the center
            _black_frame = np.zeros((640,640,3), np.uint8)
            
            # Place the cropped frame in the coordintes 200,200 of the black frame
            _black_frame[200:200+_croped_frame.shape[0], 200:200+_croped_frame.shape[1]] = _croped_frame

            #Flip the frame horizontally and vertically
            _crop = cv2.flip(_black_frame, -1)

            _confidence = self.new_insect_confidence[_insect_type]
            

            _new_insect_results = model_insects_large.predict(source=_crop, conf=_confidence, show=False, verbose = False, iou = 0.5, classes = [_insect_type], augment = True, imgsz = (640,640))

            _new_insect_detections = self._decode_DL_results(_new_insect_results)


            # if potential_new_insects[_dl_detection][4] < self.new_insect_confidence:
            if len(_new_insect_detections) == 0:
                _low_confidence.append(_dl_detection)
            else:
               pass

        _new_insects = np.delete(potential_new_insects, _low_confidence, axis=0)


        return _new_insects


    
    def __verify_bs_detections(self, _bs_detections: np.ndarray, _missing_insects: list, _unassociated_detections: np.ndarray, _predictions: np.array, _nframe: int) -> bool:

        # If there are missing insects or unassociated detections or bs_detections, run deep learning
        if (len(_missing_insects) > 0) or (len(_unassociated_detections) > 0) or (len(_bs_detections) > len(_predictions)) or (len(_bs_detections) == 0 and self.compressed_video ) or (self.compressed_video and (_nframe in self.full_frame_num)):
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
                hun_matrix[p][d] = TrackUtilities.cal_dist(_predictions[p][1],_predictions[p][2],_detections[d][0],_detections[d][1])
        
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
    
class FlowerTracker(InsectTracker):
    def __init__(self) -> None:
        super().__init__()

    def associate_detections_DL(self, _detections, _predictions, _max_dist_dl):
        _missing = [] 
        _assignments = self.Hungarian_method(_detections, _predictions)
        _insects = [i[0] for i in _predictions]
        
        _not_associated = np.zeros(shape=(0,5))
        for _nass in (_assignments[len(_insects):]):
            _not_associated = np.vstack([_not_associated,(_detections[_nass])])
                                
        
        _associations_DL = np.zeros(shape=(0,6))
        for ass in np.arange(len(_insects)):
            _record = _assignments[ass]

            if (_record <= len(_detections)-1):
                _xc, _yc, _area, _lable, _conf = _detections[_assignments[ass]][0],_detections[_assignments[ass]][1],_detections[_assignments[ass]][2],_detections[_assignments[ass]][3],_detections[_assignments[ass]][4]
                _dist = TrackUtilities.cal_dist(_xc,_yc,_predictions[ass][1],_predictions[ass][2])
                if(_dist>_max_dist_dl) and not self.low_confident_ass(_detections, _predictions, _max_dist_dl,_dist, False):
                    _missing.append(_predictions[ass][0])
                else:
                    _associations_DL = np.vstack([_associations_DL,(_predictions[ass][0],_detections[_assignments[ass]][0],_detections[_assignments[ass]][1],_detections[_assignments[ass]][2],_detections[_assignments[ass]][3],_detections[_assignments[ass]][4])])
                    
            else:
                _missing.append(_predictions[ass][0])

        return _associations_DL, _missing, _not_associated
    
    def track_flowers(self, _nframe, frame, _flower_details):

        flower_positions_dl = sorted(self.get_deep_learning_detection(frame, False,True), key=lambda x: float(x[0]))

        associations_DL, missing, not_associated  = self.associate_detections_DL(flower_positions_dl, _flower_details, pt_cfg.POLYTRACK.FLOWER_MOVEMENT_THRESHOLD)

        flower_info = (_nframe, associations_DL, missing, not_associated)

        # record_flower_positions(_nframe, associations_DL, missing, not_associated)

        return flower_info
        
class LowResMode(BS_Detections):

    def __init__(self) -> None:
        BS_Detections.__init__(self)

        self.fgbg_lowres = cv2.createBackgroundSubtractorKNN()

        return None

    def check_idle(self, _nframe: int, _predicted_position: np.array, _compressed_video: bool):
        if ((_nframe >pt_cfg.POLYTRACK.INITIAL_FRAMES) and (bool(_predicted_position) == False) and not pt_cfg.POLYTRACK.IDLE_OUT) and not _compressed_video:
            _idle = True

        else:
            _idle=False
            
        return _idle
    
    def prepare_to_track(self,nframe, vid, idle, new_insect, video_start_frame):

        if idle and (len(new_insect)>0):
            nframe = max((nframe - pt_cfg.POLYTRACK.BACKTRACK_FRAMES), video_start_frame)
            reset_frame = nframe - video_start_frame
            vid.set(1, reset_frame)
            idle = False
            new_insect = []
            pt_cfg.POLYTRACK.IDLE_OUT = True

        else:
            pass

        return nframe, idle, new_insect
    
    def process_frame(self,_frame, _compressed_video, _idle):

        if _idle and not _compressed_video:
            width, height = _frame.shape[1], _frame.shape[0]
            idle_width, idle_height = pt_cfg.POLYTRACK.LOWERES_FRAME_WIDTH, pt_cfg.POLYTRACK.LOWERES_FRAME_HEIGHT
            lores_frame = cv2.resize(_frame, (idle_width, idle_height))
            _dim_factor = (width*height)/(idle_width*idle_height)

            _contours = self.__detect_changes(lores_frame)
            _possible_insects = self.__filter_contours(_contours, _dim_factor)

            if len(_possible_insects) > 0:
                return True
            else:
                return False
        else:
            return True

    
    def __detect_changes(self, _frame):
        
        fgmask_lowres = self.fgbg_lowres.apply(_frame)
        pt_cfg.POLYTRACK.LR_MODE = True

        fgmask_lowres_processed = self.process_fgbg_output(fgmask_lowres)
        
        _contours, _ = cv2.findContours(fgmask_lowres_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return _contours
    
    def __filter_contours(self, _contours, _dim_factor):
        _insects = np.zeros(shape=(0,3))
    
        for c in _contours:
            (_x,_y), (_w, _h), _ = cv2.minAreaRect(c)
            _area = _w*_h*_dim_factor

            if (_area > pt_cfg.POLYTRACK.MIN_INSECT_AREA) and (_area<pt_cfg.POLYTRACK.MAX_INSECT_AREA):
                _insects = np.vstack([_insects,(_x,_y,_area)])
            else:
                pass
                
        return _insects

    






    
        
        
    





