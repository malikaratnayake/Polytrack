import os, sys
# import time
import cv2
import numpy as np
import itertools as it
import math
from PIL import Image
from ultralytics import YOLO

from polytrack.general import cal_dist
from polytrack.config import pt_cfg


model = YOLO('./data/yolov8_models/yolov8s_best.pt')
class_names = model.names


class DL_Detections:
    def __init__(self) -> None:
        self.confidence = pt_cfg.POLYTRACK.DL_SCORE_THRESHOLD
        self.iou_threshold = 0.7
        self.flower_class = self.__get_classes_to_detect(detect_flowers=True)
        self.insect_classes = self.__get_classes_to_detect(detect_flowers=False)

        
        return None

    def __run_deep_learning(self, _frame, detect_flowes: bool) -> np.ndarray:

        # More info: https://docs.ultralytics.com/modes/predict/#inference-arguments

        if detect_flowes:
            classes_to_detect = self.flower_class
        else:
            classes_to_detect = self.insect_classes

        results = model.predict(source=_frame, conf=pt_cfg.POLYTRACK.DL_SCORE_THRESHOLD, show=False, verbose = False, iou = 0.25, classes = classes_to_detect)


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
            area = abs((result[0] - result[2])*(result[1] - result[3]))
            _insect_detection = np.vstack([_insect_detection,(float(mid_x), float(mid_y), float(area), result[4], result[5])])

        return _insect_detection
    


        







    





def dl_detections_process(output):
    classes = pt_cfg.POLYTRACK.TRACKING_INSECTS
    # allowed_classes = pt_cfg.POLYTRACK.TRACKING_INSECTS
    # num_classes = len(classes)
    _dl_detections = np.zeros(shape=(0,6)) 
    # out_boxes, out_scores, out_classes, num_boxes = bboxes

    

    for i in range(len(output)):
        # if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = output[i][0:4]
        score = output[i][5]
        class_ind = int(output[i][4])
        # print(class_ind, classes[class_ind])
        class_name = classes[class_ind]

        # if class_name not in allowed_classes:
        #     continue
        # else:
        _dl_detections = np.vstack([_dl_detections,(coor[0], coor[1], coor[2], coor[3], class_name, score)])


    return _dl_detections


def map_darkspots(__frame, _dark_spots):
    for spot in _dark_spots:
        __frame = cv2.circle(__frame, (int(spot[0]), int(spot[1])), int(pt_cfg.POLYTRACK.DL_DARK_SPOTS_RADIUS), (100,100,100), -1)

    return __frame

# # Write the create_mosaic function to create a mosic image using the classes detected by Deep Learning
# def create_mosaic(_frame, _detections):
#     _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
#     _frame = Image.fromarray(_frame)
#     _frame = np.array(_frame)
#     _frame = cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR)
#     _frame = map_darkspots(_frame, pt_cfg.POLYTRACK.RECORDED_DARK_SPOTS)

#     for _detection in _detections:
#         _x_TL = int(float(_detection[0]))
#         _y_TL = int(float(_detection[1]))
#         _x_BR = int(float(_detection[2]))
#         _y_BR = int(float(_detection[3]))
#         _class = _detection[4]
#         _conf = _detection[5]

#         _frame = cv2.rectangle(_frame, (_x_TL, _y_TL), (_x_BR, _y_BR), (0, 255, 0), 2)
#         _frame = cv2.putText(_frame, str(_class) + ' ' + str(round(_conf, 2)), (_x_TL, _y_TL - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return _frame

def run_DL(_frame):

    # More info: https://docs.ultralytics.com/modes/predict/#inference-arguments

    results = model.predict(source=_frame, conf=pt_cfg.POLYTRACK.DL_SCORE_THRESHOLD, show=False, verbose = False, iou = 0.7)


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

    #use the bounding box coordinates to create a mosaic image and save it to the folder
    # mosaic = create_mosaic(_frame, detections)
    # cv2.imwrite(str(pt_cfg.POLYTRACK.OUTPUT) +'mosaic.jpg', mosaic)
    # print('Mosaic image saved')

    # # Show the mosaic image
    # if pt_cfg.POLYTRACK.SHOW_VIDEO_OUTPUT:
    #     cv2.imshow('Mosaic', mosaic)
    #     cv2.waitKey(1)

    


 
   




    _detections = dl_detections_process(detections)

    return _detections

    #Calculate the area covered by the insect
def cal_bodyArea_DL(_x_TL,_y_TL,_x_BR,_y_BR): 
    _body_area = abs((_x_BR-_x_TL)*(_y_BR-_y_TL))
    
    return _body_area


#Extract the data from result and calculate the center of gravity of the insect. Uses the top left and bottom right coordinates
def cal_CoG_DL(result): 
    _x_DL, _y_DL, _body_area, _radius  = 0, 0, 0, 0
    _x_TL  = int(float(result[0]))
    _y_TL = int(float(result[1]))
    _x_BR = int(float(result[2]))
    _y_BR = int(float(result[3]))
    _x_DL = int(round((_x_TL+_x_BR)/2))
    _y_DL = int(round((_y_TL+_y_BR)/2))


    
    _radius = round(cal_dist(_x_TL, _y_TL,_x_DL,_y_DL)*math.cos(math.radians(45)))
    
    _body_area = cal_bodyArea_DL(_x_TL,_y_TL,_x_BR,_y_BR)

    return _x_DL,_y_DL, _body_area, _radius


#Detect insects in frame using Deep Learning
def detect_deep_learning(_frame, flowers = False):
    _results = run_DL(_frame)
    #print(flowers)

    _deep_learning_detections = process_DL_results(_results, flowers)

    if (len(_deep_learning_detections)>1) : 
        _deep_learning_detections = verify_insects_DL(_deep_learning_detections)
    else:
        pass


    return _deep_learning_detections
    


def process_DL_results(_results, flowers):
    _logDL = np.zeros(shape=(0,5)) #(create an array to store data x,y,area, conf, type)

    for result in _results: # Go through the detected results
        confidence = result[5]
        _species = result[4]

        if not flowers:
            if ((_species != 'flower')): # Filter out detections which do not meet the threshold
                _x_DL, _y_DL, _body_area, _ = cal_CoG_DL(result) #Calculate the center of gravity
                
                _logDL = np.vstack([_logDL,(float(_x_DL), float(_y_DL), float(_body_area),_species,confidence)])
                
            else:
                pass
        else:
            if ((_species == 'flower')): # Filter out detections which do not meet the threshold
                _x_DL, _y_DL, _ , _radius = cal_CoG_DL(result) #Calculate the center of gravity
                
                _logDL = np.vstack([_logDL,(float(_x_DL), float(_y_DL), float(_radius),_species,confidence)])
            
            else:
                pass
                   
    
    return _logDL


# Calculate the distance between two coordinates
def cal_euclidean_DL(_insects_inFrame,_pair):
    _dx = float(_insects_inFrame[_pair[0]][0]) - float(_insects_inFrame[_pair[1]][0])
    _dy = float(_insects_inFrame[_pair[0]][1]) - float(_insects_inFrame[_pair[1]][1])
    _dist = np.sqrt(_dx**2+_dy**2)
    
    return _dist   

#Verify that there are no duplicate detections (The distance between two CoG are >= 20 pixels)
def verify_insects_DL(_insects_inFrame):
    _conflict_pairs = []
    _combinations = it.combinations(np.arange(len(_insects_inFrame)), 2)
    
    for pair in _combinations:
        _distance = cal_euclidean_DL(_insects_inFrame,pair)
        if (_distance<15):
            _conflict_pairs.append(pair)

    if (_conflict_pairs): _insects_inFrame = evaluvate_conflict(_conflict_pairs, _insects_inFrame)
    
    return _insects_inFrame

#Evaluvate the confidence levels in DL and remove the least confidence detections
def evaluvate_conflict(_conflict_pairs, _insects_inFrame):
    to_be_removed = []
    for pairs in _conflict_pairs:
        conf_0 = _insects_inFrame[pairs[0]][4]
        conf_1 = _insects_inFrame[pairs[1]][4]
        
        if (conf_0>=conf_1):to_be_removed.append(pairs[1])
      
        else: to_be_removed.append(pairs[0])
    
    to_be_removed = list(dict.fromkeys(to_be_removed)) #Remove duplicates
    _insects_inFrame = np.delete(_insects_inFrame, to_be_removed, 0)

    return _insects_inFrame

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
