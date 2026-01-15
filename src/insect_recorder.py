import cv2
import numpy as np
import os
import logging
import matplotlib.pyplot as plt


LOGGER = logging.getLogger()

# EXCLUDE_BOX = [1950, 750, 2400, 1150]  # Exclude box for insect detection
# INCLUDE_BOX = [500, 500, 2500, 2500]  # Include box for insect detection

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
                 video_codec: str,
                 spatial_filtering) -> None:
        
        self.width, self.height, self.fps = input_video_dimensions[0], input_video_dimensions[1], framerate
        self.show_video_output = show_video_output 
        self.save_video_output = save_video_output
        self.video_codec = video_codec
        self.output_directory = output_directory
        self.video_source = video_source
        self.output_video_dimensions = output_video_dimensions
        self.tracking_insects = tracking_insects
        self.edge_pixels = edge_pixels
        self.spatial_filtering = spatial_filtering
        self.latest_flower_positions = []
        self.updated_flower_positions_recorded = True

        if self.save_video_output or self.show_video_output:
            self.trajectory_frame = np.zeros((input_video_dimensions[1],input_video_dimensions[0],3), np.uint8)
            self.trajectory_frame = self.mark_boundary_edges(self.trajectory_frame, self.edge_pixels)
            if self.spatial_filtering.use_exclude_zone:
                self.trajectory_frame = self.mark_excluded_zone(self.trajectory_frame, self.spatial_filtering.exclude_zone_coord)
            if self.spatial_filtering.use_include_zone:
                self.trajectory_frame = self.mark_included_zone(self.trajectory_frame, self.spatial_filtering.include_zone_coord)
        
            self.output_video = self.setup_video_recording(output_directory, video_codec)
        
        return None
    
    def mark_boundary_edges(self,
                            frame: np.ndarray,
                            edge_pixels: int) -> None:

        cv2.rectangle(frame, (edge_pixels, edge_pixels), (self.width-edge_pixels, self.height-edge_pixels), (0, 0, 255), 2)
        
        return frame
    
    def mark_excluded_zone(self,
                        frame: np.ndarray,
                        exclude_box: list) -> np.ndarray:
        """
        Draws a red rectangle on the frame for the exclude zone,
        clipping to the frame boundaries.

        Args:
            frame (np.ndarray): Image frame.
            exclude_box (list): [x_min, y_min, x_max, y_max] defining the exclude zone.

        Returns:
            np.ndarray: Modified frame with the rectangle drawn.
        """
        x_min, y_min, x_max, y_max = exclude_box

        # Clip the coordinates to frame dimensions
        x_min = max(0, min(x_min, self.width - 1))
        y_min = max(0, min(y_min, self.height - 1))
        x_max = max(0, min(x_max, self.width - 1))
        y_max = max(0, min(y_max, self.height - 1))

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 6)
        return frame
    
    def mark_included_zone(self,
                        frame: np.ndarray,
                        include_box: list) -> np.ndarray:
        """
        Draws a green rectangle on the frame for the include zone,
        clipping to the frame boundaries.

        Args:
            frame (np.ndarray): Image frame.
            include_box (list): [x_min, y_min, x_max, y_max] defining the include zone.

        Returns:
            np.ndarray: Modified frame with the rectangle drawn.
        """
        x_min, y_min, x_max, y_max = include_box

        # Clip the coordinates to frame dimensions
        x_min = max(0, min(x_min, self.width - 1))
        y_min = max(0, min(y_min, self.height - 1))
        x_max = max(0, min(x_max, self.width - 1))
        y_max = max(0, min(y_max, self.height - 1))

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 6)
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
                            detections_for_predictions: np.ndarray,
                            dl_associated_detections: np.ndarray):
        
        try:
            if len(self.latest_flower_positions) > 0 and self.updated_flower_positions_recorded is False:
                for flower in self.latest_flower_positions:
                    _flower_num = int(flower[0])
                    _center_x = int(flower[1])
                    _center_y = int(flower[2])
                    _radius = int(flower[3])
                    _expanded_radius = int(round(_radius * self.flower_border))
                    cv2.circle(self.trajectory_frame, (_center_x, _center_y), _expanded_radius, (0, 0, 255), 4)
                    cv2.circle(self.trajectory_frame, (_center_x, _center_y), _radius, (0, 255, 255), 2)
                    cv2.putText(self.trajectory_frame, 'F' + str(_flower_num), (_center_x + _expanded_radius, _center_y), cv2.FONT_HERSHEY_DUPLEX , 0.7, (0,255,255), 1, cv2.LINE_AA)
                self.updated_flower_positions_recorded = True
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

        if len(dl_associated_detections) > 0:
            for det in dl_associated_detections:
                _x = int(det[1])
                _y = int(det[2])
                size = 6
                cv2.line(self.trajectory_frame, (_x - size, _y - size), (_x + size, _y + size), (0, 0, 255), 2)
                cv2.line(self.trajectory_frame, (_x - size, _y + size), (_x + size, _y - size), (0, 0, 255), 2)


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
                 video_resolution: list[int],
                 framerate: int,
                 directory_config: dict) -> None:
        
        VideoWriter.__init__(self,
                            input_video_dimensions = video_resolution,
                            output_video_dimensions = output_config.resolution,
                            video_source = directory_config.source,
                            framerate = framerate,
                            tracking_insects= insect_config.labels,
                            output_directory = directory_config.output,
                            show_video_output = output_config.show,
                            save_video_output = output_config.save,
                            edge_pixels = insect_config.edge_analysis.edge_pixels,
                            video_codec = output_config.codec,
                            spatial_filtering = insect_config.spatial_filtering) 
        
        self.insect_tracks = []
        self.edge_pixels = insect_config.edge_analysis.edge_pixels
        self.width, self.height = video_resolution 
        self.fps = framerate
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
        self.compressed_time_as_filename = output_config.compressed_time_as_filename
        self.spatial_filtering = insect_config.spatial_filtering
        self.track_sources = {}
        self.dl_confirmed_tracks = set()
        self.dl_confirmed_saved = set()
        self.save_only_dl_verified = getattr(insect_config, "save_only_dl_verified", False)
   

        return None

    
    def record_track(self,
                     frame: np.ndarray, 
                     nframe: int,
                     mapped_frame_num: int, 
                     fgbg_associated_detections: np.ndarray, 
                     dl_associated_detections: np.ndarray, 
                     missing_insects:np.ndarray, 
                     new_insects: np.ndarray,
                     new_insects_fgbg: np.ndarray) -> np.ndarray:
        
        self.active_tracks = []
        self.missing_tracks = []

        if len(dl_associated_detections) > 0 and len(fgbg_associated_detections) > 0:
            dl_ids = set(int(det[0]) for det in dl_associated_detections)
            fgbg_associated_detections = np.array(
                [det for det in fgbg_associated_detections if int(det[0]) not in dl_ids]
            )
        
        self.record_FGBG_detections(mapped_frame_num, fgbg_associated_detections)
        self.record_DL_detections(mapped_frame_num, dl_associated_detections)
        self.record_missing(mapped_frame_num, missing_insects)
        new_insect_detections = self.record_new_insect(frame, nframe, mapped_frame_num, new_insects, source="dl")
        new_insect_detections_fgbg = self.record_new_insect(frame, nframe, mapped_frame_num, new_insects_fgbg, source="fgbg")
        if len(new_insect_detections) == 0:
            new_insect_detections = new_insect_detections_fgbg
        elif len(new_insect_detections_fgbg) > 0:
            new_insect_detections = np.vstack([new_insect_detections, new_insect_detections_fgbg])

       
        detections_for_predictions, current_insect_positions = self.get_insect_positions_for_predictions(mapped_frame_num)

        if self.show_video_output or self.save_video_output:
            self.process_video_output(frame, nframe, mapped_frame_num, new_insect_detections, detections_for_predictions, dl_associated_detections)

        self.last_recorded_frame_number = mapped_frame_num

        return detections_for_predictions, current_insect_positions
    

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
            insect_position = int(next((index for index, record in enumerate(self.insect_tracks) if record[0] == _insect_num), None))


            if self.compressed_video:
                if self.continious_edge_analysis and self.check_compressed_video_jump(mapped_frame_num, _insect_num):
                    continue

                if self.check_time_jump(mapped_frame_num):
                    if not self.evaluate_missing_insect(_insect_num, mapped_frame_num, insect_position):
                        continue

            recorded_info.append([_insect_num, _x, _y,])
            insect_record = [mapped_frame_num, _x, _y, _flower, "fgbg", None]
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
            _species = int(detection[4])
            _flower = None
            insect_position = int(next((index for index, record in enumerate(self.insect_tracks) if record[0] == _insect_num), None))
            if self.track_sources.get(_insect_num) == "fgbg" and _insect_num not in self.dl_confirmed_tracks:
                self.insect_tracks[insect_position][2] = self.tracking_insects[_species]
                self.dl_confirmed_tracks.add(_insect_num)
                if _insect_num not in self.dl_confirmed_saved:
                    self.save_track(insect_position)
                    self.dl_confirmed_saved.add(_insect_num)

            if self.compressed_video:
                if self.continious_edge_analysis and self.check_compressed_video_jump(mapped_frame_num, _insect_num):
                    continue

                if self.check_time_jump(mapped_frame_num):
                    if not self.evaluate_missing_insect(_insect_num, mapped_frame_num, insect_position):
                        continue



            recorded_info.append([_insect_num, _x, _y,])
            insect_position = int(next((index for index, record in enumerate(self.insect_tracks) if record[0] == _insect_num), None))
            insect_record = [mapped_frame_num, _x, _y, _flower, "dl", float(detection[5])]
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
            acive_but_missing = self.evaluate_missing_insect(_insect_num, mapped_frame_num, insect_position)
            if acive_but_missing is True:
                self.missing_tracks.append(_insect_num)
            
            insect_record = [mapped_frame_num, _x, _y, _flower, None, None]
            self.insect_tracks[insect_position][3].append(insect_record)

        return None
    
    def evaluate_missing_insect(self, 
                                insect_num: int,
                                mapped_frame_num: int, 
                                insect_position: int):

        active_but_missing = False

        
        insect_detections = self.insect_tracks[insect_position][3]
        last_detected_frame_position = self.find_last_detected_frame(insect_detections)
        last_detected_frame, last_x, last_y, last_flower = insect_detections[last_detected_frame_position][:4]

        last_detected_along_edge = self.detected_on_edge(last_x, last_y)
        no_of_missing_frames = mapped_frame_num - last_detected_frame

        
        if ((last_detected_along_edge is True) and (no_of_missing_frames > self.max_occlusions_edge)):
            self.save_track(insect_position)

        elif (no_of_missing_frames > self.max_occlusions):
            self.save_track(insect_position)
        else:
            active_but_missing = True
            # self.missing_tracks.append(insect_num)
            
        return active_but_missing
    
    def find_last_detected_frame(self,
                                 insect_detections: list) -> int:
        
        for i in range(len(insect_detections) - 1, -1, -1):  # Iterate backwards over the list
            nested_list = insect_detections[i]
            if len(nested_list) >= 3 and nested_list[1] is not None and nested_list[2] is not None:
                return i  # Return the index of the last valid nested list
            
    def is_valid_position(self, x, y):
        """
        Returns True if (x, y) passes the include/exclude box filters:

        Args:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.
            include_box (list): [x_min, y_min, x_max, y_max] for inclusion.
            exclude_box (list): [x_min, y_min, x_max, y_max] for exclusion.
            use_include_box (bool): Enable/disable include filtering.
            use_exclude_box (bool): Enable/disable exclude filtering.

        Returns:
            bool: True if point is valid based on filters, False otherwise.
        """
        # Check include box
        if self.spatial_filtering.use_include_zone:
            x_min, y_min, x_max, y_max = self.spatial_filtering.include_zone_coord
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                LOGGER.debug(f"Point ({x}, {y}) is outside the include box.")
                return False  # Outside include zone

        # Check exclude box
        if self.spatial_filtering.use_exclude_zone:
            x_min, y_min, x_max, y_max = self.spatial_filtering.exclude_zone_coord
            if x_min <= x <= x_max and y_min <= y <= y_max:
                LOGGER.debug(f"Point ({x}, {y}) is inside the exclude box.")
                return False  # Inside exclude zone

        return True
            
    
    def record_new_insect(self, 
                          frame: np.ndarray, 
                          nframe: int,
                          mapped_frame_num: int, 
                          new_insect_detections: np.ndarray,
                          source: str = "dl") -> np.ndarray:

        recorded_info = []
    
        for detection in new_insect_detections:

            _x = int(float(detection[0]))
            _y = int(float(detection[1]))

            if self.is_valid_position(_x, _y) is True:

                self.insect_count += 1
                # _area = int(float(detection[2]))
                if len(detection) == 3:
                    _species = 0
                else:
                    _species = int(detection[3])
                _species_name = self.tracking_insects[_species]
                # _confidence = float(detection[4])
                # _status = 'In'
                # _model = 'DL'
                if self.compressed_video is True and self.compressed_time_as_filename is False:
                    _insect_num = self.generate_insect_num(mapped_frame_num, _species)
                else:
                    _insect_num = self.generate_insect_num(nframe, _species)
                # _flower = np.nan
                recorded_info.append([_insect_num, _species ,_x, _y,])

                self.manual_verification(frame,_insect_num, [_x, _y], self.tracking_insects[int(_species)])
            
                conf = None
                if len(detection) > 4:
                    conf = float(detection[4])
                insect_record_new = [_insect_num, mapped_frame_num ,_species_name, [[mapped_frame_num, _x, _y, None, source, conf]]]
                self.insect_tracks.append(insect_record_new)
                self.active_tracks.append(_insect_num)
                self.track_sources[_insect_num] = source

        return recorded_info
    

    def generate_insect_num(self,
                            nframe: int,
                            species: int) -> int:
        
        """
        Encodes the insect number based on recording time, species, and insect count.
        The number is structured to allow sorting in ascending order based on recording time.

        Args:
            nframe (int): The frame number of the recording.
            species (int): The species identifier (0-indexed).

        Returns:
            int: The encoded insect number.
        """
        # Calculate time from frame number
        current_time = int(nframe / self.fps)  # Time in seconds
        hours = current_time // 3600
        minutes = (current_time % 3600) // 60
        seconds = current_time % 60

        # Format each component
        time_str = f"{hours:02d}{minutes:02d}{seconds:02d}"  # HHMMSS
        species_str = f"{species+1:01d}"  # SS (species)
        count_str = f"{self.insect_count:05d}"  # CCCC (insect count)

        # Concatenate to form the encoded insect number
        return int(f"{species_str}{count_str}{time_str}")
  

    def manual_verification(self, 
                            frame: np.ndarray, 
                            insect_num: int, 
                            coords: list,
                            species: str)-> None:
        
        _insect_image = frame[max(coords[1]-50,1):min(coords[1]+50,self.height-1), max(coords[0]-50,1):min(coords[0]+50,self.width-1)]
        _filename= str(species)+'_'+str(insect_num)+'_img.png'
        cv2.imwrite(os.path.join(self.output_directory, os.path.basename(self.output_directory))+'_'+str(_filename), _insect_image)
        LOGGER.info(f'New {species} was detected. Insect image saved: {_filename}')

        return None
   
    

    def save_track(self, 
                   insect_position: int) -> None:
         
        insect_record = self.insect_tracks[insect_position]
        insect_num = str(insect_record[0])
        insect_species = insect_record[2]
        insect_track = insect_record[3]
        track_id = int(insect_record[0])

        LOGGER.info(f'Saving insect track: {insect_species}_{insect_num}')

        if self.save_only_dl_verified and track_id not in self.dl_confirmed_tracks:
            LOGGER.info(f'Skipping save for unverified track: {insect_species}_{insect_num}')
            image_filepath = os.path.join(self.output_directory, os.path.basename(self.output_directory))+'_'+(str(insect_species)+'_'+str(insect_num)+'_img.png')
            if os.path.exists(image_filepath):
                os.remove(image_filepath)
            return None

        detected_positions = len([record[1] for record in insect_track if record[1] is not None])
        if detected_positions >= self.min_track_length:
            filename = str(insect_species)+'_'+str(insect_num)
            output_filepath = os.path.join(self.output_directory, os.path.basename(self.output_directory))+'_'+str(filename)+'.csv'

            with open(output_filepath, 'w') as f:
                f.write('nframe, x, y, flower, detection_method, confidence\n')
                for record in insect_track:
                    f.write(f"{record[0]},{record[1]},{record[2]},{record[3]},{record[4]},{record[5]}\n")

            if self.compressed_video is True:
                interpolated_output_filepath = os.path.join(self.output_directory, os.path.basename(self.output_directory))+'_'+str(filename)+'-post_processed.csv'

                with open(interpolated_output_filepath, 'w') as f:
                    insect_track = self.process_and_interpolate_track(insect_track)
                    f.write('nframe, x, y, flower, detection_method, confidence\n')
                    for record in insect_track:
                        f.write(f"{record[0]},{record[1]},{record[2]},{record[3]},{record[4]},{record[5]}\n")

            


            LOGGER.info(f'Completed tracking {insect_species}_{insect_num}. Insect track saved: {filename}')

        else:
            LOGGER.info(f'Insect {insect_species}_{insect_num} was not tracked long enough. Insect track not saved')
            image_filepath = os.path.join(self.output_directory, os.path.basename(self.output_directory))+'_'+(str(insect_species)+'_'+str(insect_num)+'_img.png')
            os.remove(image_filepath)

        

        return None
        
    

    def process_and_interpolate_track(self, data, max_none_gap=3):
        # Ensure data is sorted by frame
        data.sort(key=lambda row: row[0])

        # Find the first and last frame with non-None x, y values
        valid_frames = [row for row in data if row[1] is not None and row[2] is not None]
        if not valid_frames:
            return []  # Return empty if no valid frames exist
        
        start_frame, end_frame = valid_frames[0][0], valid_frames[-1][0]

        # Filter data to only include frames within the valid range
        data = [row for row in data if start_frame <= row[0] <= end_frame]

        # Create a dictionary for fast lookup and fill missing frames
        frame_dict = {row[0]: row[1:] for row in data}
        interpolated_data = []

        for frame in range(start_frame, end_frame + 1):
            if frame in frame_dict:
                interpolated_data.append([frame] + frame_dict[frame])
            else:
                interpolated_data.append([frame, None, None, None, None, None])

        # Interpolate x and y values only for missing frames (not explicitly None in original data)
        for i in range(len(interpolated_data)):
            if interpolated_data[i][0] not in frame_dict:  # Missing frame, interpolate x and y
                prev = next((interpolated_data[j] for j in range(i - 1, -1, -1) if interpolated_data[j][1] is not None and interpolated_data[j][0] in frame_dict), None)
                next_ = next((interpolated_data[j] for j in range(i + 1, len(interpolated_data)) if interpolated_data[j][1] is not None and interpolated_data[j][0] in frame_dict), None)

                if prev and next_:
                    interpolated_data[i][1] = prev[1] + (next_[1] - prev[1]) * (i - interpolated_data.index(prev)) / (interpolated_data.index(next_) - interpolated_data.index(prev))
                    interpolated_data[i][2] = prev[2] + (next_[2] - prev[2]) * (i - interpolated_data.index(prev)) / (interpolated_data.index(next_) - interpolated_data.index(prev))
                elif prev:
                    interpolated_data[i][1] = prev[1]
                    interpolated_data[i][2] = prev[2]
                elif next_:
                    interpolated_data[i][1] = next_[1]
                    interpolated_data[i][2] = next_[2]


        # Fill flower values
        for i in range(len(interpolated_data)):
            # If the flower value is explicitly None in the original data, keep it as None
            if interpolated_data[i][0] in frame_dict and frame_dict[interpolated_data[i][0]][2] is None:
                interpolated_data[i][3] = None
                continue

            if interpolated_data[i][3] is None:
                prev_flower_index = next((j for j in range(i - 1, -1, -1) if interpolated_data[j][3] is not None), None)
                next_flower_index = next((j for j in range(i + 1, len(interpolated_data)) if interpolated_data[j][3] is not None), None)

                if prev_flower_index is not None and next_flower_index is not None:
                    if interpolated_data[prev_flower_index][3] == interpolated_data[next_flower_index][3]:
                        interpolated_data[i][3] = interpolated_data[prev_flower_index][3]
                    else:
                        interpolated_data[i][3] = None
                elif prev_flower_index is not None:
                    interpolated_data[i][3] = interpolated_data[prev_flower_index][3]
                elif next_flower_index is not None:
                    interpolated_data[i][3] = None

        # Additional processing: Fill short gaps with preceding flower value
        i = 0
        while i < len(interpolated_data):
            if interpolated_data[i][3] is None:
                # Identify the start and end of the gap
                gap_start = i
                while i < len(interpolated_data) and interpolated_data[i][3] is None:
                    i += 1
                gap_end = i

                # Check if the gap length is within the threshold and the preceding/succeeding flower values match
                if gap_end - gap_start <= max_none_gap:
                    prev_flower_index = next((j for j in range(gap_start - 1, -1, -1) if interpolated_data[j][3] is not None), None)
                    next_flower_index = next((j for j in range(gap_end, len(interpolated_data)) if interpolated_data[j][3] is not None), None)

                    if prev_flower_index is not None and next_flower_index is not None:
                        if interpolated_data[prev_flower_index][3] == interpolated_data[next_flower_index][3]:
                            for k in range(gap_start, gap_end):
                                interpolated_data[k][3] = interpolated_data[prev_flower_index][3]
                    elif prev_flower_index is not None:
                        for k in range(gap_start, gap_end):
                            interpolated_data[k][3] = interpolated_data[prev_flower_index][3]

            i += 1

        # Convert x, y, and flower to integers where applicable
        for row in interpolated_data:
            if row[1] is not None:
                row[1] = int(round(row[1]))
            if row[2] is not None:
                row[2] = int(round(row[2]))

        return interpolated_data
    
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
        insect_positions_for_predictions = current_insect_positions

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
            insect_positions_for_predictions = current_insect_positions

        for insect in self.missing_tracks:
            insect_position = int(next((index for index, record in enumerate(self.insect_tracks) if record[0] == insect), None))
            last_detected_frame_position = self.find_last_detected_frame(self.insect_tracks[insect_position][3])
            _, last_x, last_y, _ = self.insect_tracks[insect_position][3][last_detected_frame_position][:4]

            # if last_detected_frame is not None:
            _x0 = _x1 = last_x
            _y0 = _y1 = last_y

            insect_positions_for_predictions = np.vstack([insect_positions_for_predictions,(insect,_x0,_y0,_x1,_y1)])

        
        return insect_positions_for_predictions, current_insect_positions
    

    def save_inprogress_tracks(self, 
                               predicted_position: np.ndarray):
        
        _tracking_insects = [int(i[0]) for i in predicted_position]

        for _insect in _tracking_insects:
            insect_position = int(next((index for index, record in enumerate(self.insect_tracks) if record[0] == _insect), None))
            self.save_track(insect_position)
            LOGGER.info(f'Insect track saved: {_insect}')

    def get_unverified_track_ids(self) -> list[int]:
        return [track_id for track_id, source in self.track_sources.items() if source == "fgbg" and track_id not in self.dl_confirmed_tracks]


    



    
    
    

# Divide to n number of pices based on missing n frames
