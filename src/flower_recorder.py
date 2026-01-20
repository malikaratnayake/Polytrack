import numpy as np
import pandas as pd
import os
import logging
import math

LOGGER = logging.getLogger()


class FlowerRecorder():

    def __init__(self,
                    config: dict,
                    directory_config: dict,
                    video_resolution: tuple[int, int] | None = None) -> None:
        
        self.flower_tracks = []

        self.output_directory = directory_config.output
        self.flower_border = config.border_extension
        self.last_update_frame = 0
        self.last_recorded_flower_positions = []
        self.flowers_list = []
        detector_props = getattr(config, "detector_properties", None)
        self.iou_threshold = getattr(detector_props, "iou_threshold", 0.0) if detector_props is not None else 0.0
        self.frame_width = None
        self.frame_height = None
        if video_resolution is not None:
            self.frame_width, self.frame_height = video_resolution
        self.last_union_area = None
        self.last_union_ratio = None

        return None
    
    def record_flowers(self,
                     mapped_frame_num: int, 
                     associated_flower_detections: np.ndarray, 
                     missing_flowers: np.ndarray, 
                    new_flower_detections:np.ndarray) -> np.ndarray:
        
        self.record_flower_detections(mapped_frame_num, associated_flower_detections, missing_flowers, new_flower_detections)

        flower_detections_for_predictions = self.get_flower_positions_for_predictions(mapped_frame_num)

        self.last_recorded_flower_positions = self.get_last_recorded_flower_positions()
        self._update_union_area_ratio()
        
        return flower_detections_for_predictions, self.last_recorded_flower_positions

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

    def _merge_overlaps(self, boxes: np.ndarray, overlap_thresh: float) -> np.ndarray:
        if boxes.size == 0:
            return boxes
        merged = boxes.astype(float).tolist()
        changed = True
        while changed:
            changed = False
            new_merged = []
            while merged:
                base = merged.pop(0)
                bx1, by1, bx2, by2 = base
                i = 0
                while i < len(merged):
                    other = merged[i]
                    if self._overlap_ratio_xyxy(np.array(base), np.array(other)) >= overlap_thresh:
                        bx1 = min(bx1, other[0])
                        by1 = min(by1, other[1])
                        bx2 = max(bx2, other[2])
                        by2 = max(by2, other[3])
                        base = [bx1, by1, bx2, by2]
                        merged.pop(i)
                        changed = True
                    else:
                        i += 1
                new_merged.append(base)
            merged = new_merged
        return np.array(merged, dtype=float)

    def _clip_boxes_to_frame(self, boxes: np.ndarray, width: int, height: int) -> np.ndarray:
        if boxes.size == 0:
            return boxes
        clipped = boxes.astype(float).copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(width))
        clipped[:, 2] = np.clip(clipped[:, 2], 0.0, float(width))
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(height))
        clipped[:, 3] = np.clip(clipped[:, 3], 0.0, float(height))
        valid = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
        return clipped[valid]

    def _union_area_xyxy(self, boxes: np.ndarray) -> float:
        if boxes.size == 0:
            return 0.0
        x_coords = np.unique(np.concatenate([boxes[:, 0], boxes[:, 2]]))
        x_coords.sort()
        area = 0.0
        for i in range(len(x_coords) - 1):
            x_left = x_coords[i]
            x_right = x_coords[i + 1]
            width = x_right - x_left
            if width <= 0.0:
                continue
            mask = (boxes[:, 0] < x_right) & (boxes[:, 2] > x_left)
            if not np.any(mask):
                continue
            y_intervals = boxes[mask][:, [1, 3]]
            y_intervals = y_intervals[np.argsort(y_intervals[:, 0])]
            current_start, current_end = y_intervals[0]
            y_total = 0.0
            for y1, y2 in y_intervals[1:]:
                if y1 <= current_end:
                    current_end = max(current_end, y2)
                else:
                    y_total += max(0.0, current_end - current_start)
                    current_start, current_end = y1, y2
            y_total += max(0.0, current_end - current_start)
            area += width * y_total
        return area

    def _flower_positions_to_boxes(self) -> np.ndarray:
        if len(self.last_recorded_flower_positions) == 0:
            return np.empty((0, 4), dtype=float)
        boxes = []
        for flower in self.last_recorded_flower_positions:
            cx = float(flower[1])
            cy = float(flower[2])
            radius = float(flower[3])
            x1 = cx - radius
            y1 = cy - radius
            x2 = cx + radius
            y2 = cy + radius
            boxes.append([x1, y1, x2, y2])
        return np.array(boxes, dtype=float)

    def _flower_tracks_to_boxes(self, use_padding: bool = False) -> np.ndarray:
        if len(self.flower_tracks) == 0:
            return np.empty((0, 4), dtype=float)
        boxes = []
        for flower in self.flower_tracks:
            cx = float(flower[2][-1][1])
            cy = float(flower[2][-1][2])
            radius = float(flower[2][-1][3])
            if use_padding:
                radius = radius * float(self.flower_border)
            x1 = cx - radius
            y1 = cy - radius
            x2 = cx + radius
            y2 = cy + radius
            boxes.append([x1, y1, x2, y2])
        return np.array(boxes, dtype=float)

    def _update_union_area_ratio(self) -> None:
        if self.frame_width is None or self.frame_height is None:
            self.last_union_area = None
            self.last_union_ratio = None
            return
        boxes = self._flower_positions_to_boxes()
        overlap_thresh = max(0.0, 1.0 - float(self.iou_threshold))
        merged = self._merge_overlaps(boxes, overlap_thresh)
        clipped = self._clip_boxes_to_frame(merged, self.frame_width, self.frame_height)
        union_area = self._union_area_xyxy(clipped)
        frame_area = float(self.frame_width * self.frame_height)
        ratio = (union_area / frame_area) if frame_area > 0.0 else 0.0
        self.last_union_area = union_area
        self.last_union_ratio = ratio
    
    def get_last_recorded_flower_positions(self):

        last_recorded_flowers = []

        for flower in self.flower_tracks:
            last_recorded_flowers.append([flower[0], flower[2][-1][1], flower[2][-1][2], flower[2][-1][3]])
    
        return last_recorded_flowers
    
    def monitor_flower_visits(self,
                            insect_positions: np.ndarray) -> np.ndarray:
        
        insect_flower_visits = np.zeros(shape=(0,2))
        
        if len(self.last_recorded_flower_positions) > 0:
            for insect in insect_positions:
                x0, y0 = int(insect[1]), int(insect[2])
                for flower in self.last_recorded_flower_positions:
                    cx, cy, radius = int(flower[1]), int(flower[2]), int(flower[3])
                    if self.is_point_inside_circle(x0, y0, cx, cy, radius, self.flower_border):
                        insect_flower_visits = np.vstack([insect_flower_visits, [insect[0], flower[0]]])

        return insect_flower_visits
    

    def record_flower_visitations(self,
                                insect_flower_visits: np.ndarray,
                                mapped_frame_num: int,
                                insect_tracks: list) -> None:
        
        for visit in insect_flower_visits:
            insect_num = int(visit[0])
            flower_num = int(visit[1])

            insect_position = int(next((index for index, record in enumerate(insect_tracks) if record[0] == insect_num), None))
            insect_track_record = insect_tracks[insect_position][3]
            associated_detection_position = int(next((index for index, record in enumerate(insect_track_record) if record[0] == mapped_frame_num), None))
            insect_tracks[insect_position][3][associated_detection_position][3] = flower_num

        return None
                        

    def is_point_inside_circle(self, x0, y0, cx, cy, r, d):
        distance_to_center = math.sqrt((x0 - cx)**2 + (y0 - cy)**2)
        
        effective_radius = r * d
        
        if distance_to_center < effective_radius:
            return True  # The point is inside the circle
        else:
            return False  # The point is outside the circle
    

    def record_flower_detections(self, 
                             mapped_frame_num: int, 
                             associated_flower_detections: np.ndarray,
                             missing_flowers: np.ndarray,
                             new_flower_detections: np.ndarray) -> np.ndarray:

        for detection in associated_flower_detections:
            _flower_num = int((detection[0]))
            _cx = int((detection[1]))
            _cy = int((detection[2]))
            _radius = int((detection[3]))

            flower_position = int(next((index for index, record in enumerate(self.flower_tracks) if record[0] == _flower_num), None))
            flower_record = [mapped_frame_num, _cx, _cy, _radius]
            self.flower_tracks[flower_position][2].append(flower_record)
            
        for detection in new_flower_detections:

            if len(self.flower_tracks) == 0:
                _flower_num = 0
            else:
                _flower_num = self.flower_tracks[-1][0]+1

            _cx = int((detection[0]))
            _cy = int((detection[1]))
            _radius = int((detection[2]))
            _species = detection[3]

            flower_record_new = [_flower_num ,_species, [[mapped_frame_num, _cx, _cy, _radius]]]
            self.flower_tracks.append(flower_record_new)
            self.flowers_list.append(_flower_num)
            

        return None
    

    def get_flower_positions_for_predictions(self, 
                                             mapped_frame_num: int) -> np.ndarray:
    
        current_flower_positions = np.empty([0,5])

        for flower in self.flower_tracks:
            _cx0 = flower[2][-1][1]
            _cy0 = flower[2][-1][2]

            if not np.isnan(_cx0):

                _past_detections = len(flower[2])
                
                if(_past_detections>=2):
                    _cx1 = float(flower[2][-2][1])
                    _cy1 = float(flower[2][-2][2])

                    
                    if np.isnan(_cx1):
                        _cx1 = _cx0 
                        _cy1 = _cy0
                        
                    else:
                        _cx1 = int(_cx1)
                        _cy1 = int(_cy1)
                        
                else:
                    _cx1=_cx0 
                    _cy1=_cy0
   
                current_flower_positions = np.vstack([current_flower_positions,(flower[0],_cx0,_cy0,_cx1,_cy1)])
            
            else:
                pass

        
        return current_flower_positions
    

    def save_flower_tracks(self) -> None:

        flower_tracks_filepath = os.path.join(self.output_directory, os.path.basename(self.output_directory)+'_flower_tracks.csv')

        with open(flower_tracks_filepath, 'w') as f:
            f.write('flower_num, species, recorded_positions\n')
            for flower in self.flower_tracks:
                f.write(f"{flower[0]},{flower[1]},{flower[2]}\n")

        LOGGER.info(f'Completed tracking flowers. Flower tracks saved: {flower_tracks_filepath}')

        flower_positions_filepath = os.path.join(self.output_directory, os.path.basename(self.output_directory)+'_flower_positions.csv')

        with open(flower_positions_filepath, 'w') as f:
            f.write('flower_num, species, cx, cy, radius\n')
            for flower in self.flower_tracks:
                f.write(f"{flower[0]},{flower[1]},{flower[2][-1][1]},{flower[2][-1][2]},{flower[2][-1][3]}\n")

        LOGGER.info(f'Completed tracking flowers. Flower positions saved: {flower_positions_filepath}')

        coverage_filename = os.path.join(
            self.output_directory,
            os.path.basename(self.output_directory) + "_flower_coverage.txt",
        )
        frame_dims = (
            f"{self.frame_width}x{self.frame_height}"
            if self.frame_width is not None and self.frame_height is not None
            else "Unknown"
        )
        frame_area = (
            float(self.frame_width * self.frame_height)
            if self.frame_width is not None and self.frame_height is not None
            else 0.0
        )
        flower_count = len(self.flower_tracks)
        overlap_thresh = max(0.0, 1.0 - float(self.iou_threshold))
        boxes = self._flower_tracks_to_boxes()
        merged = self._merge_overlaps(boxes, overlap_thresh)
        clipped = self._clip_boxes_to_frame(merged, self.frame_width, self.frame_height)
        bbox_area = self._union_area_xyxy(clipped)
        coverage_ratio = (bbox_area / frame_area) if frame_area > 0.0 else 0.0
        padded_boxes = self._flower_tracks_to_boxes(use_padding=True)
        padded_merged = self._merge_overlaps(padded_boxes, overlap_thresh)
        padded_clipped = self._clip_boxes_to_frame(padded_merged, self.frame_width, self.frame_height)
        padded_bbox_area = self._union_area_xyxy(padded_clipped)
        padded_coverage_ratio = (padded_bbox_area / frame_area) if frame_area > 0.0 else 0.0
        with open(coverage_filename, "w") as f:
            f.write(f"FRAME_DIMENSIONS: {frame_dims}\n")
            f.write(f"FRAME_AREA: {frame_area:.0f}\n")
            f.write(f"FLOWER_NUMBER: {flower_count}\n")
            f.write(f"FLOWER_BOUNDINGBOX_AREA: {bbox_area:.0f}\n")
            f.write(f"FLOWER_COVERAGE_RATIO: {coverage_ratio:.6f}\n")
            f.write(f"BOUNDINGBOX_PADDING: {float(self.flower_border):.3f}\n")
            f.write(f"FLOWER_BOUNDINGBOX_AREA_WITH_PADDING: {padded_bbox_area:.0f}\n")
            f.write(f"FLOWER_COVERAGE_RATIO_WITH_PADDING: {padded_coverage_ratio:.6f}\n")

        LOGGER.info(f'Flower coverage saved: {coverage_filename}')
       
        return None

        
