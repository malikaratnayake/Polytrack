import numpy as np
from ultralytics import YOLO
import math
from tracking_methods import TrackingMethods
import logging
LOGGER = logging.getLogger()


class DL_Flower_Detector():

    def __init__(self,
                flower_detector: str,
                flower_iou_threshold: float,
                flower_detection_confidence: float,
                flower_classes: np.ndarray,
                flower_image_size: list | None,
                device: str,
                use_fp16: bool) -> None:
        self.device = device
        self.flower_detector = YOLO(flower_detector)
        self.flower_detector.to(self.device)
        self.flower_iou_threshold = flower_iou_threshold
        self.flower_detection_confidence = flower_detection_confidence
        self.flower_classes = flower_classes
        self.flower_image_size = flower_image_size
        self.use_fp16 = bool(use_fp16) and str(self.device).startswith("cuda")

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

    def _overlap_ratio_xyxy(self, a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        min_area = min(area_a, area_b)
        if min_area <= 0.0:
            return 0.0
        return inter / min_area

    def _merge_overlaps(self, detections: np.ndarray, overlap_thresh: float) -> np.ndarray:
        if detections.size == 0:
            return detections
        merged = detections.astype(float).tolist()
        changed = True
        while changed:
            changed = False
            new_merged = []
            while merged:
                base = merged.pop(0)
                bx1, by1, bx2, by2, bclass, bconf = base
                i = 0
                while i < len(merged):
                    other = merged[i]
                    if self._overlap_ratio_xyxy(np.array(base[:4]), np.array(other[:4])) >= overlap_thresh:
                        bx1 = min(bx1, other[0])
                        by1 = min(by1, other[1])
                        bx2 = max(bx2, other[2])
                        by2 = max(by2, other[3])
                        if other[5] > bconf:
                            bclass = other[4]
                            bconf = other[5]
                        base = [bx1, by1, bx2, by2, bclass, bconf]
                        merged.pop(i)
                        changed = True
                    else:
                        i += 1
                new_merged.append(base)
            merged = new_merged
        return np.array(merged, dtype=float)
    
    def __calculate_cog(self, 
                        _results: np.ndarray) -> np.ndarray:
        
        _flower_detection = np.zeros(shape=(0,5))
        overlap_thresh = max(0.0, 1.0 - self.flower_iou_threshold)
        _results = self._merge_overlaps(_results, overlap_thresh)

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
        predict_kwargs = {
            "source": frame,
            "conf": self.flower_detection_confidence,
            "show": False,
            "verbose": False,
            "iou": self.flower_iou_threshold,
            "classes": self.flower_classes,
            "half": self.use_fp16,
            "device": self.device,
        }
        if self.flower_image_size and len(self.flower_image_size) >= 2:
            # Config stores [width, height]; Ultralytics expects (height, width).
            predict_kwargs["imgsz"] = (self.flower_image_size[1], self.flower_image_size[0])

        results = self.flower_detector.predict(**predict_kwargs)
        
        detections = self._decode_flower_detections(results)
        processed_detections = self.__calculate_cog(detections)

        return processed_detections
    

class FlowerTracker(DL_Flower_Detector, TrackingMethods):

    def __init__(self,
                config: dict,
                device: str) -> None:
        
        TrackingMethods.__init__(self,
                                 prediction_method = config.prediction_method)
        
        DL_Flower_Detector.__init__(self,
                             flower_detector = config.detector_properties.model,
                             flower_iou_threshold = config.detector_properties.iou_threshold,
                             flower_detection_confidence = config.detector_properties.detection_confidence,
                             flower_classes = config.classes,
                             flower_image_size = getattr(config.detector_properties, "image_size", None),
                             device=device,
                             use_fp16 = getattr(config.detector_properties, "use_fp16", False))
        
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
    
        assignments = self.Hungarian_method(detections, predictions, cost_threshold=max_interframe_travel_distance)
        assigned_det_indices = [a[0] for a in assignments]
        assigned_pred_indices = [a[1] for a in assignments]

        for det_idx, pred_idx in assignments:
            _center_x, _center_y, _radius, _species, _confidence = self.decode_detections(detections, det_idx)
            associated_flower_detections = np.vstack([
                associated_flower_detections,
                (int(predictions[pred_idx][0]), _center_x, _center_y, _radius, _species, _confidence),
            ])

        if len(detections) > 0:
            unassociated_indices = np.setdiff1d(np.arange(len(detections)), assigned_det_indices)
            for det_idx in unassociated_indices:
                new_flower_detections = np.vstack([new_flower_detections, detections[det_idx]])

        if len(predictions) > 0:
            missing_pred_indices = np.setdiff1d(np.arange(len(predictions)), assigned_pred_indices)
            for pred_idx in missing_pred_indices:
                missing_flowers.append(predictions[pred_idx][0])


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
    
  
