import cv2
import numpy as np
import pandas as pd
import os
import logging
import math


class FlowerRecorder():

    def __init__(self,
                    output_directory: str,
                    flower_border: int) -> None:
        
        self.flower_tracks = pd.DataFrame(columns=['nframe', 'flower_num', 'cx', 'cy', 'radius', 'species', 'confidence'])
        self.output_directory = output_directory
        self.flower_border = flower_border
        self.last_update_frame = 0
        self.last_recorded_flower_positions = []

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
    
        last_recorded_flowers = self.flower_tracks[['flower_num', 'cx','cy','radius']].loc[self.flower_tracks['nframe'] == self.flower_tracks.nframe.max()].values.tolist()

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
                                insect_tracks: pd.DataFrame) -> None:
        
        for visit in insect_flower_visits:
            insect_num = int(visit[0])
            flower_num = int(visit[1])
            
            insect_row = insect_tracks.loc[(insect_tracks['insect_num'] == insect_num) & (insect_tracks['nframe'] == mapped_frame_num)].index[0]
            insect_tracks.at[insect_row, 'flower'] = flower_num
            
        
        return None
                        

    def is_point_inside_circle(self, x0, y0, cx, cy, r, d):
        distance_to_center = math.sqrt((x0 - cx)**2 + (y0 - cy)**2)
        
        effective_radius = r + d
        
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
            _species = detection[4]
            _confidence = (detection[5])
            
            flower_record_DL = [mapped_frame_num, _flower_num, _cx, _cy, _radius, _species, _confidence]
            self.flower_tracks.loc[len(self.flower_tracks)] = flower_record_DL

        for detection in new_flower_detections:
            
            _flower_num = self.flower_tracks['flower_num'].max()+1
            
            if np.isnan(_flower_num): _flower_num=0

            _cx = int((detection[0]))
            _cy = int((detection[1]))
            _radius = int((detection[2]))
            _species = detection[3]
            _confidence = detection[4]
            
            flower_record_DL = [mapped_frame_num, _flower_num, _cx, _cy, _radius, _species, _confidence]
            self.flower_tracks.loc[len(self.flower_tracks)] = flower_record_DL

        for missed_flower in missing_flowers:

            _flower_num = missed_flower
            last_pos_details = self.flower_tracks.loc[self.flower_tracks['flower_num'] == _flower_num].iloc[-1].values.tolist()

            _cx = int(float(last_pos_details[2]))
            _cy = int(float(last_pos_details[3]))
            _radius = int(float(last_pos_details[4]))
            _species = last_pos_details[5]
            _confidence = last_pos_details[6]
            
            flower_record = [mapped_frame_num, _flower_num, _cx, _cy, _radius, _species, _confidence]
            self.flower_tracks.loc[len(self.flower_tracks)] = flower_record

        return None
    

    def get_flower_positions_for_predictions(self, 
                                             mapped_frame_num: int) -> np.ndarray:
    
        current_flower_positions = np.empty([0,5])
        tracked_flowers = list(set(self.flower_tracks.loc[(self.flower_tracks['nframe'] == mapped_frame_num)]['flower_num'].values.tolist()))
        
        for flower in tracked_flowers:
            _cx0 = self.flower_tracks.loc[self.flower_tracks['flower_num'] == flower].iloc[-1]['cx']
            _cy0 = self.flower_tracks.loc[self.flower_tracks['flower_num'] == flower].iloc[-1]['cy']

            if not np.isnan(_cx0):

                _past_detections = len(self.flower_tracks.loc[self.flower_tracks['flower_num'] == flower])
                
                if(_past_detections>=2):
                    _cx1 = float(self.flower_tracks.loc[self.flower_tracks['flower_num'] == flower].iloc[-2]['cx'])
                    _cy1 = float(self.flower_tracks.loc[self.flower_tracks['flower_num'] == flower].iloc[-2]['cy'])

                    
                    if np.isnan(_cx1):
                        _cx1 = _cx0 
                        _cy1 = _cy0
                        
                    else:
                        _cx1 = int(_cx1)
                        _cy1 = int(_cy1)
                        
                else:
                    _cx1=_cx0 
                    _cy1=_cy0
   
                current_flower_positions = np.vstack([current_flower_positions,(flower,_cx0,_cy0,_cx1,_cy1)])
            
            else:
                pass

        
        return current_flower_positions
    

    def save_flower_tracks(self) -> None:
        
        self.flower_tracks.to_csv(os.path.join(self.output_directory, os.path.basename(self.output_directory))+'_flower_tracks.csv', sep=',')

        flower_positions = self.flower_tracks.loc[self.flower_tracks['nframe'] == self.flower_tracks.nframe.max()]

        flower_positions.to_csv(os.path.join(self.output_directory, os.path.basename(self.output_directory))+'_flower_positions.csv', sep=',')
        
        return None
        

        