import cv2
import numpy as np
import os
import logging
import matplotlib.pyplot as plt


LOGGER = logging.getLogger()

class VideoWriter:

    def __init__(self,
                 input_video_dimensions: list[int],
                 output_video_dimensions: list[int],
                 framerate: int,
                 video_source: str,
                 output_directory: str,
                 show_video_output: bool,
                 save_video_output: bool,
                 tracking_insects: list,
                 edge_pixels: int,
                 video_codec: str) -> None:
        
        self.width, self.height, self.fps = input_video_dimensions[0], input_video_dimensions[1], framerate
        self.show_video_output = show_video_output 
        self.save_video_output = save_video_output
        self.video_codec = video_codec
        self.output_directory = output_directory
        self.video_source = video_source
        self.output_video_dimensions = output_video_dimensions
        self.tracking_insects = tracking_insects
        self.edge_pixels = edge_pixels
        self.latest_flower_positions = []

        if self.save_video_output or self.show_video_output:
            self.trajectory_frame = np.zeros((input_video_dimensions[1],input_video_dimensions[0],3), np.uint8)
            self.trajectory_frame = self.mark_boundary_edges(self.trajectory_frame, self.edge_pixels)
            self.output_video = self.setup_video_recording(output_directory, video_codec)
        
        return None
    
    def mark_boundary_edges(self,
                            frame: np.ndarray,
                            edge_pixels: int) -> None:

        cv2.rectangle(frame, (edge_pixels, edge_pixels), (self.width-edge_pixels, self.height-edge_pixels), (0, 0, 255), 2)
        
        return frame
    
    def setup_video_recording(self, 
                              output_directory: str,
                              video_codec: str):
        
        codec = cv2.VideoWriter_fourcc(*video_codec)
        output_filename = os.path.join(output_directory, os.path.basename(self.output_directory))+'_out.mp4'
        output_video = cv2.VideoWriter(output_filename, codec, self.fps, (self.width, self.height))
        LOGGER.info(f"Video output saved to: {output_filename}")

        return output_video
    
    def update_flower_positions(self,
                                flower_positions: np.ndarray,
                                flower_border: float) -> None:
        
        self.latest_flower_positions = flower_positions
        self.flower_border = flower_border
        self.updated_flower_positions_recorded = False

        return None

    
    def process_video_output(self, 
                            frame: np.ndarray,
                            nframe: int, 
                            mapped_frame_num: int,
                            new_insect_detections: np.ndarray, 
                            detections_for_predictions: np.ndarray):
        
        try:
            if len(self.latest_flower_positions) > 0 and self.updated_flower_positions_recorded is False:
                for flower in self.latest_flower_positions:
                    _flower_num, _center_x, _center_y, _radius = int(flower[0]), int(flower[1]), int(flower[2]), int(flower[3]*self.flower_border)
                    cv2.circle(self.trajectory_frame, (_center_x, _center_y), _radius, (0,0,255), 4)
                    cv2.putText(self.trajectory_frame, 'F' +str(_flower_num), (_center_x+_radius, _center_y), cv2.FONT_HERSHEY_DUPLEX , 0.7, (0,255,255), 1, cv2.LINE_AA)
                self.updated_position_recorded = True
        except Exception as e:
            LOGGER.error(f'Error while updating flower positions: {e}')

        for detection in detections_for_predictions:
            _insect_num, _x0, _y0, _x1, _y1 = detection

            if _x0 is not None:
                cv2.circle(self.trajectory_frame, (int(_x0), int(_y0)), 3, self.track_colour(_insect_num), 4)

            if _x0 is not None and _x1 is not None:
                cv2.line(self.trajectory_frame, (int(_x1),int(_y1)),(int(_x0),int(_y0)),self.track_colour(_insect_num),2)


        for record in new_insect_detections:
            _insect_num,_species, _x, _y = record
            cv2.circle(self.trajectory_frame, (_x, _y), 3, self.track_colour(_insect_num), 4)
            cv2.putText(self.trajectory_frame, str(self.tracking_insects[int(_species)])+' ' + str(_insect_num), (_x+20, _y+20), cv2.FONT_HERSHEY_DUPLEX , 0.7, self.track_colour(_insect_num), 1, cv2.LINE_AA) 


        cv2.putText(frame, f"Compressed Frame: {str(nframe)} | Uncompressed Frame: {str(mapped_frame_num)}", (20, 20), cv2.FONT_HERSHEY_DUPLEX , 0.8, (255,255,255), 1, cv2.LINE_AA)
   
        output_frame = cv2.add(frame, self.trajectory_frame)



        if self.show_video_output:
            cv2.imshow("PolyTrack - Insect Tracks only", cv2.resize(output_frame, (self.output_video_dimensions[0], self.output_video_dimensions[1])))

        if self.save_video_output:
            self.output_video.write(output_frame)

        return None

    
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
    

   

    
class Recorder(VideoWriter):

    def __init__(self,
                 output_config: dict,
                 insect_config: dict,
                 source_config: dict,
                 directory_config: dict) -> None:

        
        VideoWriter.__init__(self,
                            input_video_dimensions = source_config.resolution,
                            output_video_dimensions = output_config.resolution,
                            video_source = directory_config.source,
                            framerate = source_config.framerate,
                            tracking_insects= insect_config.labels,
                            output_directory = directory_config.output,
                            show_video_output = output_config.show,
                            save_video_output = output_config.save,
                            edge_pixels = insect_config.edge_analysis.edge_pixels,
                            video_codec = output_config.codec) 
        
        self.insect_tracks = []
        self.edge_pixels = insect_config.edge_analysis.edge_pixels
        self.width, self.height, self.fps = source_config.resolution[0], source_config.resolution[1], source_config.framerate
        self.max_occlusions = insect_config.max_occlusions
        self.max_occlusions_edge = insect_config.edge_analysis.max_edge_occlusions
        self.max_occlusions_on_flower = insect_config.max_occlusions_on_flower
        self.tracking_insects = insect_config.labels
        self.min_track_length = insect_config.min_track_length
        self.compressed_time_jump = insect_config.edge_analysis.compressed_video_time_jump
        self.insect_count = 0
        self.last_recorded_frame_number = 0
        self.compressed_video = source_config.compressed_video
        self.continious_edge_analysis = insect_config.edge_analysis.continious_analysis
        self.video_frame_width, self.video_frame_height = output_config.resolution[0], output_config.resolution[1]



        return None
    
    def record_track(self,
                     frame: np.ndarray, 
                     nframe: int,
                     mapped_frame_num: int, 
                     fgbg_associated_detections: np.ndarray, 
                     dl_associated_detections: np.ndarray, 
                     missing_insects:np.ndarray, 
                     new_insects: np.ndarray) -> np.ndarray:
        
        self.active_tracks = []
        self.missing_tracks = []
        
        self.record_FGBG_detections(mapped_frame_num, fgbg_associated_detections)
        self.record_DL_detections(mapped_frame_num, dl_associated_detections)
        self.record_missing(mapped_frame_num, missing_insects)
        new_insect_detections = self.record_new_insect(frame, nframe ,mapped_frame_num,  new_insects)
       
        detections_for_predictions = self.get_insect_positions_for_predictions(mapped_frame_num)

        if self.show_video_output or self.save_video_output:
            self.process_video_output(frame, nframe, mapped_frame_num, new_insect_detections, detections_for_predictions)

        self.last_recorded_frame_number = mapped_frame_num

        return detections_for_predictions
    

    def check_time_jump(self,
                        mapped_frame_num: int) -> bool:
        
        if (mapped_frame_num - self.last_recorded_frame_number) > self.compressed_time_jump:
            return True
        else:
            return False
        


    def record_FGBG_detections(self, 
                             mapped_frame_num: int, 
                             fgbg_associated_detections: np.ndarray) -> np.ndarray:

        recorded_info = []
        
        for detection in fgbg_associated_detections:
            _insect_num = int((detection[0]))
            _x = int((detection[1]))
            _y = int((detection[2]))
            _flower = None

            if self.compressed_video is True and self.continious_edge_analysis is True:
                if self.check_compressed_video_jump(mapped_frame_num, _insect_num) is True:
                    break

            recorded_info.append([_insect_num, _x, _y,])
            insect_position = int(next((index for index, record in enumerate(self.insect_tracks) if record[0] == _insect_num), None))
            insect_record = [mapped_frame_num, _x, _y, _flower]
            self.insect_tracks[insect_position][3].append(insect_record)
            self.active_tracks.append(_insect_num)

        return None
    

    
    def check_compressed_video_jump(self,
                                     mapped_frame_num: int,
                                     insect_num: int) -> bool:
        
        if self.check_time_jump(mapped_frame_num):
            insect_position = int(next((index for index, record in enumerate(self.insect_tracks) if record[0] == insect_num), None))
            insect_detections = self.insect_tracks[insect_position][3]
            last_detected_frame_position = self.find_last_detected_frame(insect_detections)
            _, last_x, last_y, _ = insect_detections[last_detected_frame_position]

            if self.detected_on_edge(last_x, last_y) is True:
                self.save_track(insect_position)
                LOGGER.info(f'Insect {insect_num} was detected on the edge. Insect track saved')
                return True
            else:
                return False
        else:
            return False
     

    def record_DL_detections(self, 
                             mapped_frame_num: int, 
                             dl_associated_detections: np.ndarray) -> np.ndarray:

        recorded_info = []

        for detection in dl_associated_detections:
            _insect_num = int((detection[0]))
            _x = int((detection[1]))
            _y = int((detection[2]))
            _flower = None

            if self.compressed_video is True and self.continious_edge_analysis is True:
                if self.check_compressed_video_jump(mapped_frame_num, _insect_num) is True:
                    break

            recorded_info.append([_insect_num, _x, _y,])
            insect_position = int(next((index for index, record in enumerate(self.insect_tracks) if record[0] == _insect_num), None))
            insect_record = [mapped_frame_num, _x, _y, _flower]
            self.insect_tracks[insect_position][3].append(insect_record)
            self.active_tracks.append(_insect_num)

        return None
    

    def record_missing(self,
                       mapped_frame_num: int, 
                       missing_insects:np.ndarray) -> None:
        
        for insect in missing_insects:
            _insect_num = int(insect)
            insect_position = int(next((index for index, record in enumerate(self.insect_tracks) if record[0] == _insect_num), None))
            _x = None
            _y = None           
            _flower = None
            # _model = np.nan
            self.evaluate_missing_insect(_insect_num, mapped_frame_num, insect_position)
            
            insect_record = [mapped_frame_num, _x, _y, _flower]
            self.insect_tracks[insect_position][3].append(insect_record)

        return None
    
    def evaluate_missing_insect(self, 
                                insect_num: int,
                                mapped_frame_num: int, 
                                insect_position: int):
        
        insect_detections = self.insect_tracks[insect_position][3]
        last_detected_frame_position = self.find_last_detected_frame(insect_detections)
        last_detected_frame, last_x, last_y, last_flower = insect_detections[last_detected_frame_position]

        last_detected_along_edge = self.detected_on_edge(last_x, last_y)
        no_of_missing_frames = mapped_frame_num - last_detected_frame

        
        if ((last_detected_along_edge is True) and (no_of_missing_frames > self.max_occlusions_edge)):
            self.save_track(insect_position)

        elif (no_of_missing_frames > self.max_occlusions) and (last_flower is None):
            self.save_track(insect_position)

        elif (no_of_missing_frames > self.max_occlusions_on_flower) and (last_flower is not None):
            self.save_track(insect_position)
        else:
            self.missing_tracks.append(insect_num)
            
        return None
    
    def find_last_detected_frame(self,
                                 insect_detections: list) -> int:
        
        for i in range(len(insect_detections) - 1, -1, -1):  # Iterate backwards over the list
            nested_list = insect_detections[i]
            if len(nested_list) >= 3 and nested_list[1] is not None and nested_list[2] is not None:
                return i  # Return the index of the last valid nested list
            
    
    def record_new_insect(self, 
                          frame: np.ndarray, 
                          nframe: int,
                          mapped_frame_num: int, 
                          new_insect_detections: np.ndarray) -> np.ndarray:

        recorded_info = []
    
        for detection in new_insect_detections:

            self.insect_count += 1
                
            _x = int(float(detection[0]))
            _y = int(float(detection[1]))
            # _area = int(float(detection[2]))
            if len(detection) == 3:
                _species = 0
            else:
                _species = int(detection[3])
            _species_name = self.tracking_insects[_species]
            # _confidence = float(detection[4])
            # _status = 'In'
            # _model = 'DL'
            _insect_num = self.generate_insect_num(nframe, _species)
            # _flower = np.nan
            recorded_info.append([_insect_num, _species ,_x, _y,])

            self.manual_verification(frame,_insect_num, [_x, _y], self.tracking_insects[int(_species)])
        
            insect_record_new = [_insect_num, mapped_frame_num ,_species_name, [[mapped_frame_num, _x, _y, None]]]
            self.insect_tracks.append(insect_record_new)
            self.active_tracks.append(_insect_num)


        return recorded_info
    

    def generate_insect_num(self,
                            nframe: int,
                            species: int) -> int:
        
        current_time = int(nframe / self.fps)
        hours = current_time // 3600
        minutes = (current_time % 3600) // 60
        seconds = current_time % 60
        insect_num = int(int(f"{hours:02d}{minutes:02d}{seconds:02d}") + (species+1)*1000000000 + self.insect_count*1000000)

        return insect_num
  

    def manual_verification(self, 
                            frame: np.ndarray, 
                            insect_num: int, 
                            coords: list,
                            species: str)-> None:
        
        _insect_image = frame[max(coords[1]-50,1):min(coords[1]+50,1079), max(coords[0]-50,1):min(coords[0]+50,1919)]
        _filename= str(species)+'_'+str(insect_num)+'_img.png'
        cv2.imwrite(os.path.join(self.output_directory, os.path.basename(self.output_directory))+'_'+str(_filename), _insect_image)
        LOGGER.info(f'New {species} was detected. Insect image saved: {_filename}')

        return None
   
    

    def save_track(self, 
                   insect_position: int) -> None:
         
        insect_record = self.insect_tracks[insect_position]
        insect_num = insect_record[0]
        insect_species = insect_record[2]
        insect_track = insect_record[3]

        detected_positions = len([record[1] for record in insect_track if record[1] is not None])
        if detected_positions >= self.min_track_length:
            filename = str(insect_species)+'_'+str(insect_num)
            output_filepath = os.path.join(self.output_directory, os.path.basename(self.output_directory))+'_'+str(filename)+'.csv'

            with open(output_filepath, 'w') as f:
                f.write('nframe, x, y, flower\n')
                for record in insect_track:
                    f.write(f"{record[0]},{record[1]},{record[2]},{record[3]}\n")

            # self.plot_insect_track(insect_track, insect_num, insect_species)

            LOGGER.info(f'Completed tracking {insect_species}_{insect_num}. Insect track saved: {filename}')

        else:
            LOGGER.info(f'Insect {insect_species}_{insect_num} was not tracked long enough. Insect track not saved')
            image_filepath = os.path.join(self.output_directory, os.path.basename(self.output_directory))+'_'+(str(insect_species)+'_'+str(insect_num)+'_img.png')
            os.remove(image_filepath)

        

        return None
    
    def plot_insect_track(self,
                            insect_track: list,
                            insect_num: int,
                            insect_species:str) -> None:
        
        x = [record[1] for record in insect_track if record[1] is not None]
        y = [self.video_frame_height - record[2] for record in insect_track if record[2] is not None]

        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(0, self.video_frame_width)
        plt.ylim(0, self.video_frame_height)
        plt.grid()
        plt.title(str(insect_species)+'_'+str(insect_num))
        filename = str(insect_species)+'_'+str(insect_num)
        output_filepath = os.path.join(self.output_directory, os.path.basename(self.output_directory))+'_'+str(filename)+'.png'
        plt.savefig(output_filepath)



    def detected_on_edge(self, 
                                     last_x: int,
                                     last_y: int) -> np.ndarray:

        if ((last_x < self.edge_pixels) or (last_x > (self.width-self.edge_pixels))):
            return True
        
        elif ((last_y < self.edge_pixels) or (last_y > (self.height-self.edge_pixels))):
            return True
        
        else:
            return False

    

    def get_insect_positions_for_predictions(self, 
                                             mapped_frame_num: int) -> np.ndarray:
    
        current_insect_positions = np.empty([0,5])

        for insect in self.active_tracks:
            # insect_position = next((i for i, insect in enumerate(self.insect_tracks) if insect[0] == insect), None)
            insect_position = int(next((index for index, record in enumerate(self.insect_tracks) if record[0] == insect), None))

            if len(self.insect_tracks[insect_position][3]) >= 2:
                _x0 = self.insect_tracks[insect_position][3][-1][1]
                _y0 = self.insect_tracks[insect_position][3][-1][2]
                _x1 = self.insect_tracks[insect_position][3][-2][1]
                _y1 = self.insect_tracks[insect_position][3][-2][2]

                if _x1 is None or _y1 is None:
                    _x1 = _x0 
                    _y1 = _y0 

            else:
                _x0 = _x1 = self.insect_tracks[insect_position][3][-1][1]
                _y0 = _y1 = self.insect_tracks[insect_position][3][-1][2]

            current_insect_positions = np.vstack([current_insect_positions,(insect,_x0,_y0,_x1,_y1)])

        for insect in self.missing_tracks:
            insect_position = int(next((index for index, record in enumerate(self.insect_tracks) if record[0] == insect), None))
            last_detected_frame_position = self.find_last_detected_frame(self.insect_tracks[insect_position][3])
            _, last_x, last_y, _ = self.insect_tracks[insect_position][3][last_detected_frame_position]

            # if last_detected_frame is not None:
            _x0 = _x1 = last_x
            _y0 = _y1 = last_y

            current_insect_positions = np.vstack([current_insect_positions,(insect,_x0,_y0,_x1,_y1)])

        
        return current_insect_positions
    

    def save_inprogress_tracks(self, 
                               predicted_position: np.ndarray):
        
        _tracking_insects = [int(i[0]) for i in predicted_position]

        for _insect in _tracking_insects:
            insect_position = int(next((index for index, record in enumerate(self.insect_tracks) if record[0] == _insect), None))
            self.save_track(insect_position)
            LOGGER.info(f'Insect track saved: {_insect}')


    



    
    
    

# Divide to n number of pices based on missing n frames



