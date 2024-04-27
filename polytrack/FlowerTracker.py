import numpy as np
from ultralytics import YOLO
import math
from polytrack.TrackingMethods import TrackingMethods
import logging
LOGGER = logging.getLogger()


class DL_Flower_Detector():

    def __init__(self,
                flower_detector: str,
                flower_iou_threshold: float,
                flower_detection_confidence: float,
                flower_classes: np.ndarray) -> None:
        self.flower_detector = YOLO(flower_detector)
        self.flower_iou_threshold = flower_iou_threshold
        self.flower_detection_confidence = flower_detection_confidence
        self.flower_classes = flower_classes

        return None
    
    def _decode_flower_detections(self, 
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
        
        _flower_detection = np.zeros(shape=(0,5))

        for result in _results:
            min_x = result[0]
            min_y = result[1]
            max_x = result[2]
            max_y = result[3]

            # Calculate center of the minimum enclosing circle
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
    
            # Calculate radius of the minimum enclosing circle
            dx = max_x - cx
            dy = max_y - cy
            radius = math.sqrt(dx**2 + dy**2) / 2

            _flower_detection = np.vstack([_flower_detection,(cx, cy, radius, result[4], result[5])])

        return _flower_detection
        
    

    def run_flower_detector(self, 
                        frame: np.ndarray) -> np.ndarray:
        

        results = self.flower_detector.predict(source=frame, 
                                                conf=self.flower_detection_confidence, 
                                                show=False, 
                                                verbose = False, 
                                                iou = self.flower_iou_threshold, 
                                                classes = self.flower_classes)
        
        detections = self._decode_flower_detections(results)
        processed_detections = self.__calculate_cog(detections)

        return processed_detections
    

class FlowerTracker(DL_Flower_Detector, TrackingMethods):

    def __init__(self,
                flower_detector: str,
                flower_iou_threshold: float,
                flower_detection_confidence: float,
                flower_classes: np.ndarray,
                prediction_method: str) -> None:
        
        TrackingMethods.__init__(self,
                                 prediction_method = prediction_method)
        
        DL_Flower_Detector.__init__(self,
                             flower_detector = flower_detector,
                             flower_iou_threshold = flower_iou_threshold,
                             flower_detection_confidence = flower_detection_confidence,
                             flower_classes = flower_classes)
        
        self.flower_predictions = []
        

        return None
    
    def run_flower_tracker(self, 
                    frame: np.ndarray,
                    flower_predictions: np.ndarray) -> np.ndarray:
        
        self.flower_predictions = flower_predictions

        dl_flower_detections = self.run_flower_detector(frame)

        associated_flower_detections, missing_flowers, new_flower_detections= self.process_flower_detections(dl_flower_detections, self.flower_predictions)

        return associated_flower_detections, missing_flowers, new_flower_detections

             
    def process_flower_detections(self,
                            detections: np.array,
                            predictions: np.array) -> tuple:
          
        max_interframe_travel_distance = 10
        unassociated_array_length = 5

              
        missing_flowers = []
        new_flower_detections = np.zeros(shape=(0,unassociated_array_length))
        associated_flower_detections = np.zeros(shape=(0,6))
    
        assignments = self.Hungarian_method(detections, predictions)
        tracking_numbers = [i[0] for i in predictions]
        num_of_flowers_tracked = len(tracking_numbers)
          
        for _unassociated in (assignments[num_of_flowers_tracked:]):
            new_flower_detections = np.vstack([new_flower_detections,(detections[_unassociated])])       
                                  
          
        for _flower in np.arange(num_of_flowers_tracked):
            _flower_num = assignments[_flower]
    
            if (_flower_num < len(detections)):
                _center_x, _center_y, _radius, _species, _confidence = self.decode_detections(detections, _flower_num)
                _distance_error = self.calculate_distance(_center_x,_center_y, predictions[_flower][1], predictions[_flower][2])
                if(_distance_error > max_interframe_travel_distance):
                    missing_flowers.append(predictions[_flower][0])
                else:
                    associated_flower_detections = np.vstack([associated_flower_detections,(int(predictions[_flower][0]),_center_x, _center_y, _radius, _species, _confidence)])
            else:
                missing_flowers.append(predictions[_flower][0])


        return associated_flower_detections, missing_flowers, new_flower_detections
    
    def decode_detections(self, 
                          detections: np.ndarray, 
                          flower_num: int):
        
        _center_x = int(detections[flower_num][0])
        _center_y = int(detections[flower_num][1])
        _radius = int(detections[flower_num][2])
        _species = detections[flower_num][3]
        _confidence = detections[flower_num][4]

        return _center_x, _center_y, _radius, _species, _confidence
    
  
