import numpy as np
import pandas as pd
import os
import logging
import math

LOGGER = logging.getLogger()


class FlowerRecorder():

    def __init__(self,
                    output_directory: str,
                    flower_border: int) -> None:
        
        self.flower_tracks = []

        self.output_directory = output_directory
        self.flower_border = flower_border
        self.last_update_frame = 0
        self.last_recorded_flower_positions = []
        self.flowers_list = []

        return None
    
    def record_flowers(self,
                     mapped_frame_num: int, 
                     associated_flower_detections: np.ndarray, 
                     missing_flowers: np.ndarray, 
                     new_flower_detections:np.ndarray) -> np.ndarray:
        
        self.record_flower_detections(mapped_frame_num, associated_flower_detections, missing_flowers, new_flower_detections)

        flower_detections_for_predictions = self.get_flower_positions_for_predictions(mapped_frame_num)

        self.last_recorded_flower_positions = self.get_last_recorded_flower_positions()
        
        return flower_detections_for_predictions, self.last_recorded_flower_positions
    
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
       
        return None

        