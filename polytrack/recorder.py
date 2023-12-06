import cv2
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
from polytrack.config import pt_cfg
# from polytrack.general import cal_abs_time, assign_insect_num,cal_dist, assign_datapoint_name, check_sight_coordinates
from polytrack.utilities import Utilities
from operator import itemgetter


TrackUtilities = Utilities()


class VideoWriter:
    width = pt_cfg.POLYTRACK.FRAME_WIDTH
    height = pt_cfg.POLYTRACK.FRAME_HEIGHT
    fps = pt_cfg.POLYTRACK.FPS
    output_video = pt_cfg.POLYTRACK.SHOW_TRACK_FRAME or pt_cfg.POLYTRACK.SHOW_VIDEO_OUTPUT or pt_cfg.POLYTRACK.SAVE_TRACK_FRAME or pt_cfg.POLYTRACK.SAVE_VIDEO_OUTPUT
    # track_frame = np.zeros((1080,1920,3), np.uint8)
    # if pt_cfg.POLYTRACK.SAVE_VIDEO_OUTPUT: self.out_video = self.setup_video_save(pt_cfg.POLYTRACK.OUTPUT)
    # if pt_cfg.POLYTRACK.SAVE_TRACK_FRAME: self.out_track = self.setup_track_save(pt_cfg.POLYTRACK.OUTPUT)

    def __init__(self):
        # self.width = pt_cfg.POLYTRACK.FRAME_WIDTH
        # self.height = pt_cfg.POLYTRACK.FRAME_HEIGHT
        # self.fps = pt_cfg.POLYTRACK.FPS
        # self.output_video = pt_cfg.POLYTRACK.SHOW_TRACK_FRAME or pt_cfg.POLYTRACK.SHOW_VIDEO_OUTPUT or pt_cfg.POLYTRACK.SAVE_TRACK_FRAME or pt_cfg.POLYTRACK.SAVE_VIDEO_OUTPUT
        self.track_frame = np.zeros((1080,1920,3), np.uint8)
        if pt_cfg.POLYTRACK.SAVE_VIDEO_OUTPUT: self.out_video = self.setup_video_save(pt_cfg.POLYTRACK.OUTPUT)
        if pt_cfg.POLYTRACK.SAVE_TRACK_FRAME: self.out_track = self.setup_track_save(pt_cfg.POLYTRACK.OUTPUT)

        return None
    
    def setup_video_save(self, output_directory: str):
        codec = cv2.VideoWriter_fourcc(*pt_cfg.POLYTRACK.VIDEO_WRITER)
        out_video = cv2.VideoWriter(str(output_directory)+'video_'+str(TrackUtilities.assign_datapoint_name())+'.avi', codec, self.fps, (self.width, self.height))

        return out_video
    
    def setup_track_save(self, output_directory: str):
        codec = cv2.VideoWriter_fourcc(*pt_cfg.POLYTRACK.VIDEO_WRITER)
        out_track = cv2.VideoWriter(str(output_directory)+'track.avi', codec, self.fps, (self.width, self.height))

        return out_track
    
    def process_output_video(self, frame, track_frame,details_frame, _nframe, idle):

        track_frame, display_frame = self.prepare_output_video(frame, track_frame, details_frame, _nframe)

        if pt_cfg.POLYTRACK.SHOW_TRACK_FRAME:
            cv2.imshow("PolyTrack - Insect Tracks only", cv2.resize(track_frame, (pt_cfg.POLYTRACK.VIDEO_OUTPUT_WIDTH, pt_cfg.POLYTRACK.VIDEO_OUTPUT_HEIGHT)))
        
        if pt_cfg.POLYTRACK.SHOW_VIDEO_OUTPUT:
            cv2.imshow("PolyTrack - Insect Tracks", cv2.resize(display_frame, (pt_cfg.POLYTRACK.VIDEO_OUTPUT_WIDTH, pt_cfg.POLYTRACK.VIDEO_OUTPUT_HEIGHT)))

        if pt_cfg.POLYTRACK.SAVE_VIDEO_OUTPUT and not idle and not pt_cfg.POLYTRACK.IDLE_OUT:
            #pt_cfg.POLYTRACK.SAVE_VIDEO_OUTPUT and not idle and not pt_cfg.POLYTRACK.IDLE_OUT
            self.out_video.write(display_frame)
            
        if pt_cfg.POLYTRACK.SAVE_TRACK_FRAME and not idle:
            self.out_track.write(track_frame)

        return None
    
    def prepare_output_video(self, frame, track_frame, details_frame, _nframe):

        details_frame = cv2.putText(details_frame, 'Frame: ' +str(_nframe) + '| Time' +str(TrackUtilities.cal_abs_time(_nframe, pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS)), (20, 20), cv2.FONT_HERSHEY_DUPLEX , 0.8, (255,255,255), 1, cv2.LINE_AA)
        track_frame = cv2.add(details_frame,track_frame)
        display_frame = cv2.add(frame, track_frame)

        return track_frame, display_frame
    
    def track_colour(self, _insect_num):
            
        if (_insect_num <= 5):
            _colour_code = (_insect_num*5)%6
        else:
            _colour_code = _insect_num%6
        
    
        if(_colour_code ==0): _colour = (255,0,0)
        elif(_colour_code ==1): _colour = (0,255,0)
        elif(_colour_code ==2): _colour = (0,0,255)
        elif(_colour_code ==3): _colour = (0,255,255)
        elif(_colour_code ==4): _colour = (255,0,255)
        else: _colour = (255,255,0)
        
        return _colour
    
    def mark_flower_positions(self, _flower_info):

        _center_x, _center_y, _radius = _flower_info

        cv2.circle(self.track_frame, (_center_x, _center_y), _radius, (0,0,255), 4)

        return None
    
class FlowerRecorder:

    latest_flower_details = []
    flower_tracks = None
    flowers = None
    flower_entry_exit = None

    def __init__(self):
        # super().__init__()  # Initiate parent classes
        self.flower_entry_exit = pd.DataFrame(columns = ['nframe', 'flower','insect_num','entry_time','exit_time'])
        self.flowers = pd.DataFrame(columns = ['flower_num', 'x0','y0','radius','species','confidence'])
        self.flower_tracks = pd.DataFrame(columns = ['nframe','flower_num', 'x0','y0','radius','species','confidence'])

        return None
    
    def get_flowers(self):

        flower_details = self.flowers[['flower_num', 'x0','y0','radius']].values

        return flower_details

    def update_flower_analysis(self,_insect_track, _insectname):
        
        self.flowers[str(_insectname)+'_time'] = np.nan
        self.flowers[str(_insectname)+'_visits'] = np.nan

        for flower in np.arange(0, len(self.flowers),1):
            self.flowers.iloc[flower, self.flowers.columns.get_loc(str(_insectname)+'_time')] = len(_insect_track[_insect_track['flower'] == flower])
            self.flowers.iloc[flower, self.flowers.columns.get_loc(str(_insectname)+'_visits')] =  _insect_track['visit_num'][_insect_track['flower'] == flower].max()


        return None

    def save_flowers(self):
        try:
            self.flowers.insert(self.flowers.columns.get_loc('y0')+1,'y_adj', ((1080-self.flowers['y0'])) if pt_cfg.POLYTRACK.FACING_NORTH else self.flowers['y0'])
            self.flowers.insert(self.flowers.columns.get_loc('confidence')+1,'Total_time',self.flowers[[col for col in self.flowers.columns if col.endswith('_time')]].sum(axis=1))
            self.flowers.insert(self.flowers.columns.get_loc('Total_time')+1,'Total_visits', self.flowers[[col for col in self.flowers.columns if col.endswith('_visits')]].sum(axis=1))
            self.flowers.to_csv(str(pt_cfg.POLYTRACK.OUTPUT)+'flowers_'+str(TrackUtilities.assign_datapoint_name())+'.csv', sep=',')
            self.flower_tracks.to_csv(str(pt_cfg.POLYTRACK.OUTPUT)+'flowes_tracks_'+str(TrackUtilities.assign_datapoint_name())+'.csv', sep=',')
        
        except:
            pass

        return None
    

    def record_flower_positions(self, _flower_info):

        _nframe, _associations_DL, _missing, _not_associated =_flower_info

        for ass_d in _associations_DL:
            _flower_num = int(float(ass_d[0]))
            _x = int(float(ass_d[1]))
            _y = int(float(ass_d[2]))
            _radius = int(float(ass_d[3]))
            _species = ass_d[4]
            _confidence = ass_d[5]

            flower_record = [_nframe, _flower_num, _x, _y, _radius, _species, _confidence]
            self.flower_tracks.loc[len(self.flower_tracks)] = flower_record

    
        for miss in _missing:
            _flower_num = miss
            last_pos_details = self.flower_tracks.loc[self.flower_tracks['flower_num'] == _flower_num].iloc[-1].values.tolist()

            _x = int(float(last_pos_details[2]))
            _y = int(float(last_pos_details[3]))
            _radius = int(float(last_pos_details[4]))
            _species = last_pos_details[5]
            _confidence = last_pos_details[6]
            
            flower_record = [_nframe, _flower_num, _x, _y, _radius, _species, _confidence]
            self.flower_tracks.loc[len(self.flower_tracks)] = flower_record

        _not_associated = sorted(_not_associated, key=lambda x: float(x[0]))

        for nass in _not_associated:
            _flower_num = self.flower_tracks['flower_num'].max()+1
            if np.isnan(_flower_num): _flower_num=0

            _x = int(float(nass[0]))
            _y = int(float(nass[1]))
            _radius = int(float(nass[2]))
            _species = nass[3]
            _confidence = nass[4]
            
            flower_record = [_nframe, _flower_num, _x, _y, _radius, _species, _confidence]
            self.flower_tracks.loc[len(self.flower_tracks)] = flower_record

        return self.get_flower_details()
    
    def get_flower_details(self):
    
        _recorded_flowers = self.flower_tracks[['flower_num', 'x0','y0','radius']].loc[self.flower_tracks['nframe'] == self.flower_tracks.nframe.max()].values.tolist()

        return _recorded_flowers
    

    def check_on_flower(self, _coordinates, _flowers):
        _x = _coordinates[0]
        _y = _coordinates[1]
        foraging_flowers = []

        for flower in _flowers:
            dist_from_c = TrackUtilities.cal_dist(_x,_y,flower[1],flower[2])
            if dist_from_c <=  flower[3]*pt_cfg.POLYTRACK.FLOWER_RADIUS_THRESHOLD:
                foraging_flowers.insert(len(foraging_flowers),[flower[0],dist_from_c])
            else:
                pass


        current_flower = self.evaluate_flowers(foraging_flowers)

    
        return current_flower


    def evaluate_flowers(self, _foraging_flowers):
        if (_foraging_flowers):
            if (len(_foraging_flowers) == 1):
                _current_flower = _foraging_flowers[0][0]
            else:
                _current_flower = sorted(_foraging_flowers, key=itemgetter(1))[0][0]
        else:
            _current_flower = np.nan

        return _current_flower
        
    
    

    
class Recorder(VideoWriter, FlowerRecorder):

    def __init__(self)-> None:
        VideoWriter.__init__(self)  # Initiate parent classes
        FlowerRecorder.__init__(self)

        self.insect_tracks = pd.DataFrame(columns=['nframe', 'insect_num', 'x0', 'y0', 'area', 'species', 'confidence', 'status', 'model', 'flower', 'visit_num'])
        if pt_cfg.POLYTRACK.SIGHTING_TIMES:
            self.insects_sightings = pd.DataFrame(columns=['species', 'insect_num', 'start_time', 'end_time'])
        if pt_cfg.POLYTRACK.INSECT_VERIFICATION:
            self.dropped_insects = []
        self.edge_pixels = pt_cfg.POLYTRACK.EDGE_PIXELS
        self.width, self.height, self.fps = pt_cfg.POLYTRACK.FRAME_WIDTH, pt_cfg.POLYTRACK.FRAME_HEIGHT, pt_cfg.POLYTRACK.FPS
        self.max_occlusions = pt_cfg.POLYTRACK.MAX_OCCLUSIONS
        self.max_occlusions_edge = pt_cfg.POLYTRACK.MAX_OCCLUSIONS_EDGE
        if pt_cfg.POLYTRACK.RECORD_ENTRY_EXIT_FLOWER: 
            self.flower_entry_exit = pd.DataFrame(columns = ['nframe', 'flower','insect_num','entry_time','exit_time'])

        # if pt_cfg.POLYTRACK.DL_DARK_SPOTS: dark_spots = []
        return None
    
    def record_track(self, frame,_nframe, _associated_det_BS, _associated_det_DL, _missing, _new_insect, latest_flower_positions , idle):


        #Record Data

        if self.output_video: 
            details_frame = np.zeros((1080,1920,3), np.uint8)
        else:
            details_frame = None

        
        self.record_BS_detections(_nframe, details_frame, _associated_det_BS, latest_flower_positions)
        self.record_DL_detections(_nframe, details_frame, _associated_det_DL, latest_flower_positions)
        self.mark_flower_positions(latest_flower_positions)
        self.record_missing(_nframe, _missing)
        self.record_new_insect(frame,_nframe, _new_insect, latest_flower_positions)
        
        if self.output_video: self.process_output_video(frame, self.track_frame, details_frame, _nframe, idle)
        
        _for_predictions = self.get_data_predictions(_nframe)

        if pt_cfg.POLYTRACK.INSECT_VERIFICATION and (_nframe % pt_cfg.POLYTRACK.INSECT_VERIFICATION_INTERVAL == 0):
            _for_predictions = self.verify_insects(_for_predictions, latest_flower_positions)
        
        return _for_predictions
    
    def mark_flower_positions(self, flower_info):

        if flower_info is not None:

            for flower in flower_info:
                _flower_name, _center_x, _center_y, _radius = flower
                _flower_name = str(_flower_name)
                _center_x = int(_center_x)
                _center_y = int(_center_y)
                _radius = int(_radius)

                cv2.circle(self.track_frame, (_center_x, _center_y), _radius, (0,0,255), 4)
                cv2.putText(self.track_frame, 'F' +_flower_name, (_center_x+_radius, _center_y), cv2.FONT_HERSHEY_DUPLEX , 0.7, (0,255,255), 1, cv2.LINE_AA)

        else:
            pass

        return None
        


    def record_BS_detections(self, _nframe, details_frame, _associated_det_BS, latest_flower_positions):
        
        for record in _associated_det_BS:
            _insect_num = int(float(record[0]))
            _x = int(float(record[1]))
            _y = int(float(record[2]))
            _area = int(float(record[3]))
            _species = self.insect_tracks['species'][self.insect_tracks.loc[self.insect_tracks['insect_num'] == _insect_num, 'species'].last_valid_index()]
            _confidence = np.nan
            _status = 'In'
            _model = 'BS'
            _flower, _visit_number = self.update_analysis(_nframe, _insect_num, [_x,_y], self.insect_tracks, latest_flower_positions)
            
            
            if self.output_video:
                cv2.circle(self.track_frame, (_x, _y), 3, self.track_colour(_insect_num), 2)
                cv2.putText(details_frame, str(pt_cfg.POLYTRACK.TRACKING_INSECTS[int(_species)])+' ' + str(_insect_num)+' BS', (_x+20, _y+20), cv2.FONT_HERSHEY_DUPLEX , 0.7, self.track_colour(_insect_num), 1, cv2.LINE_AA) 

            
            insect_record_BS = [_nframe, _insect_num, _x, _y, _area, _species, _confidence, _status, _model,_flower, _visit_number]
            self.insect_tracks.loc[len(self.insect_tracks)] = insect_record_BS

        return None
            

    def update_analysis(self, _nframe, _insect_num, _coordinates,_insect_tracks, latest_flower_positions):

        if pt_cfg.POLYTRACK.ANALYSE_POLLINATION:

            if (_nframe % pt_cfg.POLYTRACK.ANALYSIS_UPDATE_FREQUENCY == 0):
                _flower_current = self.check_on_flower(_coordinates, latest_flower_positions)
                _visit_number = self.update_visit_num(_flower_current, _insect_num,_insect_tracks)
                if pt_cfg.POLYTRACK.RECORD_ENTRY_EXIT_FLOWER: self.record_entry_exit(_nframe, _flower_current,_insect_tracks, _insect_num)
            
            
            else:
                _flower_current = _insect_tracks.loc[_insect_tracks['insect_num'] == _insect_num].iloc[-1]['flower']
                _visit_number = _insect_tracks.loc[_insect_tracks['insect_num'] == _insect_num].iloc[-1]['visit_num']

        else:
            _flower_current, _visit_number = np.nan, np.nan

        return _flower_current, _visit_number
    

    def verify_insects(self, for_predictions, latest_flower_positions):

        _verified_predictions = np.empty([0,5])
        dark_spots = pt_cfg.POLYTRACK.RECORDED_DARK_SPOTS

        for prediction in for_predictions:
            insect_num = prediction[0]
            total_records = len(self.insect_tracks[(self.insect_tracks.insect_num == insect_num)])
            bs_ratio = len(self.insect_tracks[(self.insect_tracks.insect_num == insect_num) & (self.insect_tracks.model == 'BS')])/total_records
            not_on_flower = np.isnan(self.check_on_flower([prediction[1], prediction[2]],latest_flower_positions))
            #print(total_records, bs_ratio, not_on_flower)


            if (total_records >= pt_cfg.POLYTRACK.INSECT_VERIFICATION_MIN_FRAMES) and (bs_ratio < pt_cfg.POLYTRACK.INSECT_VERIFICATION_MIN_BS) and not_on_flower:
                last_detections = self.get_last_detections(insect_num, pt_cfg.POLYTRACK.INSECT_VERIFICATION_LAST_FRAMES)
                if self.cal_cum_distance(last_detections) <= pt_cfg.POLYTRACK.INSECT_VERIFICATION_THRESHOLD_CUM_DISTANCE:
                    self.dropped_insects.append(insect_num)

                    if pt_cfg.POLYTRACK.DL_DARK_SPOTS: dark_spots.append(self.get_dark_spot(prediction))
                else:
                    _verified_predictions = np.vstack([_verified_predictions,(prediction)])
            else:
                _verified_predictions = np.vstack([_verified_predictions,(prediction)])

        pt_cfg.POLYTRACK.RECORDED_DARK_SPOTS = dark_spots
        #print('dark_spots', dark_spots)

        
        return _verified_predictions
    

    def get_last_detections(self, _insect_num, _num_frames):
        _last_detections = self.insect_tracks[(self.insect_tracks['insect_num'] == _insect_num) & (self.insect_tracks['status'] == 'In' )][self.insect_tracks.columns[2:4]][-min(len(self.insect_tracks[(self.insect_tracks['insect_num'] == _insect_num)]),_num_frames):].values

        return _last_detections
    

    def cal_cum_distance(self, _last_detections):
        cum_distance = 0
        for _d in np.arange(0,len(_last_detections)-1,1):
            cum_distance += TrackUtilities.cal_dist(_last_detections[_d][0],_last_detections[_d][1],_last_detections[_d+1][0],_last_detections[_d+1][1])

        return cum_distance
    
    def get_dark_spot(self, _prediction):
        x0 = self.insect_tracks['x0'][self.insect_tracks[(self.insect_tracks.insect_num == _prediction[0])].first_valid_index()]
        y0 = self.insect_tracks['y0'][self.insect_tracks[(self.insect_tracks.insect_num == _prediction[0])].first_valid_index()]

        if self.output_video: cv2.circle(self.track_frame, (int(x0), int(y0)), int(pt_cfg.POLYTRACK.DL_DARK_SPOTS_RADIUS), (255,255,255), 0)

        return [x0,y0]
    

    def record_DL_detections(self, _nframe, details_frame, _associated_det_DL, latest_flower_positions):

        for record in _associated_det_DL:
            _insect_num = int(float(record[0]))
            _x = int(float(record[1]))
            _y = int(float(record[2]))
            _area = int(float(record[3]))
            _species = self.insect_tracks['species'][self.insect_tracks.loc[self.insect_tracks['insect_num'] == _insect_num, 'species'].last_valid_index()]
            _confidence = float(record[5])
            _status = 'In'
            _model = 'DL'
            _flower, _visit_number = self.update_analysis(_nframe, _insect_num, [_x,_y], self.insect_tracks, latest_flower_positions)
            
            if self.output_video:
                cv2.circle(self.track_frame, (_x, _y), 3, self.track_colour(_insect_num), 2)
                cv2.putText(details_frame, str(pt_cfg.POLYTRACK.TRACKING_INSECTS[int(_species)])+' ' + str(_insect_num)+' DL' + str(round(int(_confidence),2)), (_x+20, _y+20), cv2.FONT_HERSHEY_DUPLEX , 0.7, self.track_colour(_insect_num), 1, cv2.LINE_AA) 

            
            insect_record_DL = [_nframe, _insect_num, _x, _y, _area, _species, _confidence, _status, _model,_flower, _visit_number]
            self.insect_tracks.loc[len(self.insect_tracks)] = insect_record_DL

        return None
    

    def record_missing(self,_nframe, _missing):
        
        for record in _missing:
            _insect_num = int(float(record))
            _x = np.nan
            _y = np.nan
            _area = np.nan
            _species = self.insect_tracks['species'][self.insect_tracks.loc[self.insect_tracks['insect_num'] == _insect_num, 'species'].last_valid_index()]
            _confidence = np.nan
            _status = self.evaluate_missing(_nframe, _insect_num)
            _model = np.nan
            _flower, _visit_number = np.nan, np.nan
            
            insect_record_missing = [_nframe, _insect_num, _x, _y, _area, _species, _confidence, _status, _model,_flower, _visit_number]
            self.insect_tracks.loc[len(self.insect_tracks)] = insect_record_missing 


        return None
    
    def record_new_insect(self, _frame, _nframe, _new_insect, latest_flower_positions):

        if len(_new_insect)>0 : pt_cfg.POLYTRACK.IDLE_OUT = False
    
        for record in _new_insect:
            
            if TrackUtilities.check_sight_coordinates(record): #verify whether detection is not false positive
                
                # _insect_num = insect_tracks['insect_num'].max()+1
                current_time = TrackUtilities.cal_abs_time(_nframe, pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS)
                _insect_num = TrackUtilities.assign_insect_num(current_time, pt_cfg.POLYTRACK.INSECT_COUNT)
                
                # if np.isnan(_insect_num): _insect_num=0
                    
                _x = int(float(record[0]))
                _y = int(float(record[1]))
                _area = int(float(record[2]))
                _species = record[3]
                _confidence = float(record[4])
                _status = 'In'
                _model = 'DL'
                _flower = self.check_on_flower([_x,_y], latest_flower_positions)
                _visit_number = np.nan if np.isnan(_flower) else 1

                
                # _, _ = update_analysis(_nframe, _insect_num, [_x,_y], insect_tracks)
                self.manual_verification(_frame,_insect_num, _x, _y,pt_cfg.POLYTRACK.TRACKING_INSECTS[int(_species)],_confidence)
                
                if self.output_video:
                    cv2.circle(self.track_frame, (int(float(_x)), int(float(_y))), 4, self.track_colour(_insect_num), 4)
                    cv2.putText(self.track_frame, str(pt_cfg.POLYTRACK.TRACKING_INSECTS[int(_species)])+' ' + str(_insect_num), (_x+20, _y+20), cv2.FONT_HERSHEY_DUPLEX , 0.7, self.track_colour(_insect_num), 1, cv2.LINE_AA) 

                insect_record_new = [_nframe, _insect_num, _x, _y, _area, _species, _confidence, _status, _model,_flower, _visit_number]
                self.insect_tracks.loc[len(self.insect_tracks)] = insect_record_new

                if pt_cfg.POLYTRACK.SIGHTING_TIMES:
                    new_insect = [_species, _insect_num, current_time, np.nan]
                    self.insects_sightings.loc[len(self.insects_sightings)] = new_insect  

                if pt_cfg.POLYTRACK.RECORD_ENTRY_EXIT_FLOWER: self.record_entry_exit(_nframe, _flower,self.insect_tracks, _insect_num, new_insect=True)

            else:
                pass   
    

    def manual_verification(self, _frame, _insect_num, _x, _y, _species, _confidence):
        _insect_image = _frame[max(_y-50,1):min(_y+50,1079), max(_x-50,1):min(_x+50,1919)]
        _filename= str(_species)+'_'+str(_insect_num)+'_img.png'
        cv2.imwrite(str(pt_cfg.POLYTRACK.OUTPUT)+str(_filename), _insect_image)

        return None
    

    def evaluate_missing(self, _nframe, _insect_num):

        #check whether it has left the frame (last appearence in the edge and being missing for more than 15 frames)
        _last_edge_det = self.last_det_check(_insect_num)
        _missing_frames = _nframe - self.insect_tracks['nframe'][self.insect_tracks.loc[self.insect_tracks['insect_num'] == _insect_num, 'x0'].last_valid_index()]

        # print(_last_edge_det, _missing_frames, _nframe, insect_tracks['nframe'][insect_tracks.loc[insect_tracks['insect_num'] == _insect_num, 'x0'].last_valid_index()], pt_cfg.POLYTRACK.NOISY)
        
        if ((_last_edge_det==True) and (_missing_frames>self.max_occlusions_edge)) and not pt_cfg.POLYTRACK.NOISY:
            _status ='out'
            self.save_track(_insect_num)
            # print('out')
            
        elif(_missing_frames>self.max_occlusions) and not pt_cfg.POLYTRACK.NOISY:
            _status ='out'
            self.save_track(_insect_num)
            # print('out---1')
        
        else:
            _status = 'missing'
            # print('missing')
            
        return _status
    

    def save_track(self, _insect_num):
        _insect_track = self.insect_tracks.loc[self.insect_tracks['insect_num'] == _insect_num].reset_index()
        #_insect_track = _insect_track[:_insect_track['x0'].last_valid_index()+1]
        _species_num = self.insect_tracks['species'][self.insect_tracks.loc[self.insect_tracks['insect_num'] == _insect_num, 'confidence'].idxmax()]
        _insect_track.insert(6,'y_adj', (self.height-_insect_track['y0']) if pt_cfg.POLYTRACK.FACING_NORTH else _insect_track['y0'])
        _species = pt_cfg.POLYTRACK.TRACKING_INSECTS[int(_species_num)]
        _insectname = str(_species)+'_'+str(_insect_num)

        if pt_cfg.POLYTRACK.FILTER_TRACKS:
            last_detections = self.get_last_detections(_insect_num, pt_cfg.POLYTRACK.FILTER_TRACKS_VERIFY_FRAMES)

            if self.cal_cum_distance(last_detections) >= pt_cfg.POLYTRACK.FILTER_TRACKS_DIST_THRESHOLD:
                # _insectname = str(_species)+'_'+str(_insect_num)
                _insect_track.to_csv(str(pt_cfg.POLYTRACK.OUTPUT)+str(_insectname)+'.csv', sep=',') #Save the csv file with insect track

            else:
                pass
        
        else:
            # _insectname = str(_species)+'_'+str(_insect_num)
            _insect_track.to_csv(str(pt_cfg.POLYTRACK.OUTPUT)+str(_insectname)+'.csv', sep=',') #Save the csv file with insect track

        
        
        if pt_cfg.POLYTRACK.SIGHTING_TIMES:
            exit_time = TrackUtilities.cal_abs_time(_insect_track['nframe'][_insect_track.loc[_insect_track['insect_num'] == _insect_num, 'x0'].last_valid_index()], pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS)
            self.insects_sightings.loc[self.insects_sightings['insect_num'] == _insect_num, 'end_time'] = exit_time

    
        if pt_cfg.POLYTRACK.UPDATE_FLOWER_ANALYSIS: self.update_flower_analysis(_insect_track, _insectname)
        
        del _insect_track

        return None

    


    def last_det_check(self, _insect_num):
    
        in_edge = False
        
        last_x = self.insect_tracks['x0'][self.insect_tracks.loc[self.insect_tracks['insect_num'] == _insect_num, 'x0'].last_valid_index()]
        last_y = self.insect_tracks['y0'][self.insect_tracks.loc[self.insect_tracks['insect_num'] == _insect_num, 'y0'].last_valid_index()]
        
        if ((last_x < self.edge_pixels) or (last_x > (self.width-self.edge_pixels))):
            in_edge = True
        elif ((last_y < self.edge_pixels) or (last_y > (self.height-self.edge_pixels))):
            in_edge = True
        else:
            in_edge = False
            
        return in_edge
    
    def complete_tracking(self, _predicted_position):
        
        print('======== Tracking Completed ======== ')
        _tracking_insects = [int(i[0]) for i in _predicted_position]

        pt_cfg.POLYTRACK.DARK_SPOTS = []
        
        
        for _insect in _tracking_insects:
            self.save_track(_insect)

        # save_flowers()
        
        if pt_cfg.POLYTRACK.RECORD_ENTRY_EXIT_FLOWER: self.save_flower_entry_exit()

        if pt_cfg.POLYTRACK.SIGHTING_TIMES: self.insects_sightings.to_csv(str(pt_cfg.POLYTRACK.OUTPUT)+'sight_times.csv', sep=',')

        self.insect_tracks.to_csv(str(pt_cfg.POLYTRACK.OUTPUT)+'tracks.csv', sep=',')


    def get_data_predictions(self, _nframe):
    
        _for_predictions = np.empty([0,5])
        _active_insects = list(set(self.insect_tracks.loc[(self.insect_tracks['nframe'] == _nframe)&(self.insect_tracks['status'] == 'In')]['insect_num'].values.tolist()))
        _missing_insects = list(set(self.insect_tracks.loc[(self.insect_tracks['nframe'] == _nframe)&(self.insect_tracks['status'] == 'missing')]['insect_num'].values.tolist()))
        #print(_nframe, _active_insects, _missing_insects)
        
        for insect in _active_insects:
            _x0 = self.insect_tracks.loc[self.insect_tracks['insect_num'] == insect].iloc[-1]['x0']
            _y0 = self.insect_tracks.loc[self.insect_tracks['insect_num'] == insect].iloc[-1]['y0']

            if not np.isnan(_x0):

                _past_detections = len(self.insect_tracks.loc[self.insect_tracks['insect_num'] == insect])
                
                if(_past_detections>=2):
                    _x1 = float(self.insect_tracks.loc[self.insect_tracks['insect_num'] == insect].iloc[-2]['x0'])
                    _y1 = float(self.insect_tracks.loc[self.insect_tracks['insect_num'] == insect].iloc[-2]['y0'])

                    
                    if np.isnan(_x1):
                        _x1  =_x0 
                        _y1 =_y0
                        
                    else:
                        _x1 = int(_x1)
                        _y1 = int(_y1)
                        

                else:
                    _x1=_x0 
                    _y1=_y0

                    
                _for_predictions = np.vstack([_for_predictions,(insect,_x0,_y0,_x1,_y1)])
                if self.output_video: cv2.line(self.track_frame,(int(_x1),int(_y1)),(int(_x0),int(_y0)),self.track_colour(insect),2)
            
            else:
                pass



        
        if (_missing_insects):
            for insect in _missing_insects:
                _x0m = _x1m = self.insect_tracks['x0'][self.insect_tracks.loc[self.insect_tracks['insect_num'] == insect, 'x0'].last_valid_index()]
                _y0m = _y1m = self.insect_tracks['y0'][self.insect_tracks.loc[self.insect_tracks['insect_num'] == insect, 'y0'].last_valid_index()]

                _for_predictions = np.vstack([_for_predictions,(insect,_x0m,_y0m,_x1m,_y1m)])
        
        return _for_predictions
    

    def update_visit_num(self, _flower_current, _insect_num,_insect_tracks):
        if np.isnan(_flower_current):
            _visit_number = np.nan
        else:
            previously_visited = _insect_tracks[_insect_tracks['insect_num'] == _insect_num].flower.dropna()
            if bool(len(previously_visited.values)):
                _last_visited_flower = previously_visited.iloc[-1]
                if (_last_visited_flower != _flower_current):
                    previously_visited_unique = previously_visited.unique()
                    if (_flower_current in previously_visited_unique):
                        _visit_number = _insect_tracks['visit_num'][_insect_tracks.loc[(_insect_tracks['insect_num'] == _insect_num) & (_insect_tracks['flower'] == _flower_current), 'flower'].last_valid_index()]+1
                    else:
                        _visit_number = 1
                else:
                    _visit_number = _insect_tracks['visit_num'][_insect_tracks.loc[(_insect_tracks['insect_num'] == _insect_num) & (_insect_tracks['flower'] == _flower_current), 'flower'].last_valid_index()]
            else:
                _visit_number = 1

        return _visit_number


    def record_entry_exit(self,_nframe, _current_flower,_insect_tracks, _insect_num, new_insect=False):
        current_position = _current_flower
        last_frame_position = _insect_tracks[_insect_tracks['insect_num'] == _insect_num].flower.values[-1]

        if not np.isnan(current_position) and np.isnan(last_frame_position):
            print(_nframe, "Entered the flower", _insect_num, last_frame_position, current_position, TrackUtilities.cal_abs_time(_nframe, pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS))
            entry_record = [_nframe, current_position, _insect_num,  TrackUtilities.cal_abs_time(_nframe, pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS), np.nan]
            self.flower_entry_exit.loc[len(self.flower_entry_exit)] = entry_record

        elif not np.isnan(current_position) and new_insect:
            print(_nframe, "Entered the flower throgh new insect", _insect_num, last_frame_position, current_position, TrackUtilities.cal_abs_time(_nframe, pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS))
            entry_record = [_nframe, current_position, _insect_num,  TrackUtilities.cal_abs_time(_nframe, pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS), np.nan]
            self.flower_entry_exit.loc[len(self.flower_entry_exit)] = entry_record

        elif np.isnan(current_position) and not np.isnan(last_frame_position):
            flower_entry_record = self.flower_entry_exit.loc[(self.flower_entry_exit['flower'] == int(last_frame_position)) & (self.flower_entry_exit['insect_num'] == _insect_num)].last_valid_index()
            print(last_frame_position, _insect_num, flower_entry_record)
            self.flower_entry_exit.loc[flower_entry_record,'exit_time'] = TrackUtilities.cal_abs_time(_nframe, pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS)
            # print(_nframe, "exited the flower", _insect_num, last_frame_position, current_position, cal_abs_time(_nframe, pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS))
        else:
            pass



    def save_flower_entry_exit(self):

        self.flower_entry_exit.to_csv(str(pt_cfg.POLYTRACK.OUTPUT)+'flower_entry_exit.csv', sep=',')

        return None
    
    
    





