import os
import json
import logging
import time
import yaml
from datetime import datetime
from pathlib import Path
from itertools import product
import cv2
import numpy as np
from insect_tracker import InsectTracker
from insect_recorder import Recorder
from event_logger import EventLogger
from flower_tracker import FlowerTracker
from flower_recorder import FlowerRecorder

LOGGER = logging.getLogger()

# LOGGER = logging.getLogger(__name__)
# LOGGER.setLevel(logging.INFO)

class Config:
    """
    Class to hold system configuration parameters.
    """
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries into Config objects
                value = Config(value)
            setattr(self, key, value)

    def to_dict(self):
        """
        Convert the Config instance back to a dictionary.
        """
        return {
            key: value.to_dict() if isinstance(value, Config) else value
            for key, value in self.__dict__.items()
        }

    def save_to_yaml(self, file_path):
        """
        Save the updated configuration to a YAML file.

        Args:
            file_path (str): Path to the YAML file where the configuration will be saved.
        """
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f)

# Load system configuration from YAML file
with open('./config/config.yaml', 'r') as f:
    yaml_config = yaml.safe_load(f)

# Create Config instances
DIRECTORY_CONFIG = Config(yaml_config["directories"])
INSECT_CONFIG = Config(yaml_config["insects"])
FLOWER_CONFIG = Config(yaml_config["flowers"])
SOURCE_CONFIG = Config(yaml_config["source"])
OUTPUT_CONFIG = Config(yaml_config["output"])




class TracknRecord():

    def __init__(self,
                 video_source: str,
                 RecordTracks: Recorder,
                 TrackInsects: InsectTracker,
                 TrackFlowers: FlowerTracker,
                 RecordFlowers: FlowerRecorder,
                 compressed_video: bool,
                 info_filename: str,
                 flower_detection_interval:int) -> None:
        
        self.video_source = video_source
        self.RecordTracks = RecordTracks
        self.TrackInsects = TrackInsects
        self.TrackFlowers = TrackFlowers
        self.RecordFlowers = RecordFlowers
        self.compressed_video = compressed_video
        self.info_filename = info_filename
        self.flower_detection_interval = flower_detection_interval
        self.vid = cv2.VideoCapture(self.video_source)
        LOGGER.info(f"Processing video: {self.video_source}")
        

        if self.compressed_video:
            _, _, self.full_frame_numbers = self.TrackInsects.get_compression_details(self.video_source, self.info_filename)

    def run(self) -> None:
        nframe = 0
        predicted_position = []
        flower_predictions = []
        while True:
            _, frame = self.vid.read()

            if frame is not None:
                nframe += 1
                mapped_frame_num = self.TrackInsects.map_frame_number(nframe, self.compressed_video)
                fgbg_associated_detections, dl_associated_detections, missing_insects, new_insects = self.TrackInsects.run_tracker(frame, nframe, predicted_position)
                for_predictions = self.RecordTracks.record_track(frame, nframe, mapped_frame_num,fgbg_associated_detections, dl_associated_detections, missing_insects, new_insects)
                predicted_position = self.TrackInsects.predict_next(for_predictions)


                if self.TrackFlowers is not None and ((self.compressed_video and (nframe in self.full_frame_numbers)) or (not self.compressed_video and (nframe == 5 or nframe % self.flower_detection_interval == 0))):
                    associated_flower_detections, missing_flowers, new_flower_detections = self.TrackFlowers.run_flower_tracker(frame, flower_predictions)
                    flower_detections_for_predictions, latest_flower_positions = self.RecordFlowers.record_flowers(mapped_frame_num, associated_flower_detections, missing_flowers, new_flower_detections)
                    flower_predictions = self.TrackFlowers.predict_next(flower_detections_for_predictions)
                    self.RecordTracks.update_flower_positions(latest_flower_positions, self.RecordFlowers.flower_border)


                if (len(for_predictions) > 0) and self.RecordFlowers is not None:
                    insect_flower_visits = self.RecordFlowers.monitor_flower_visits(for_predictions)
                    self.RecordFlowers.record_flower_visitations(insect_flower_visits, mapped_frame_num, self.RecordTracks.insect_tracks)
                    

                if cv2.waitKey(1) & 0xFF == ord('q'): break

            else:
                LOGGER.info("Finished processing video. Exiting...")
                self.RecordTracks.save_inprogress_tracks(predicted_position)
                if self.RecordFlowers is not None: self.RecordFlowers.save_flower_tracks()
                
                break

        self.vid.release()
        cv2.destroyAllWindows()

        return nframe






def main(directory_config: Config):
    start = time.time()
    

    if type(directory_config.source) is str:
        output_filename = os.path.splitext(os.path.basename(directory_config.source))[0]
    else:
        output_filename = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine the output directory based on user input

    if os.path.isdir(directory_config.output):
        output_parent_directory = Path(directory_config.output, "Polytrack")
        log_message = f"Outputting to {output_parent_directory}"
    else:
        output_parent_directory = Path(directory_config.source, "Polytrack")
        log_message = f"Output directory not specified or unavailable. Outputting to video source directory  {output_parent_directory}"



    # output_parent_directory = Path(config.output_directory , "EcoMotionZip")
    if not output_parent_directory.exists():
        os.makedirs(output_parent_directory, exist_ok=True)


    output_directory = Path(f"{output_parent_directory}/{output_filename}")
    # output_directory = Path(f"out/{output_filename}")
    if not output_directory.exists():
        output_directory.mkdir()

    EventLogger(output_directory)
    LOGGER.info(f"Starting processing at :  {datetime.fromtimestamp(start)}")
    # LOGGER.info(f"Running main() with Config:  {config.__dict__}")
    LOGGER.info(f"Outputting to {output_filename}")
    
    #Create a copy of the config file in the output directory
        # Save the updated configurations back to the YAML file
    # config_yaml = {
    #     'directories': directory_config.to_dict(),
    #     'insects': INSECT_CONFIG.to_dict(),
    #     'flowers': FLOWER_CONFIG.to_dict(),
    #     'recording': RECORDING_CONFIG.to_dict()
    # }
 
    # # Save the updated YAML file to the Data directory in format YYYYMMDD_HHMMSS.yaml
    # with open(output_parent_directory / "config.yaml", 'w') as f:
    #     yaml.dump(config_yaml, f)
    

    # Create all of our threads
    track_insects = InsectTracker(
        config = INSECT_CONFIG,
        source_config=SOURCE_CONFIG,
        directory_config=directory_config)
    
    
    record_tracks = Recorder(
        output_config=OUTPUT_CONFIG,
        source_config=SOURCE_CONFIG,
        directory_config=directory_config)
    
    if FLOWER_CONFIG.track:
        track_flowers = FlowerTracker(
            config = FLOWER_CONFIG)
        
        record_flowers = FlowerRecorder(
            config = FLOWER_CONFIG,
            directory_config = directory_config)
    
    track_and_record = TracknRecord(
        video_source = directory_config.source,
        RecordTracks = record_tracks,
        TrackInsects = track_insects,
        TrackFlowers = track_flowers if FLOWER_CONFIG.track else None,
        RecordFlowers = record_flowers if FLOWER_CONFIG.track else None,
        flower_detection_interval = FLOWER_CONFIG.tracking.detection_interval if FLOWER_CONFIG.track else None,
        compressed_video = SOURCE_CONFIG.compressed_video,
        info_filename = SOURCE_CONFIG.compression_info)
    
    
    # Run the TracknRecord instance
    frames_processed = track_and_record.run()


    # Add any extra stats/metadata to output too
    end = time.time()
    duration_seconds = end - start
    # formatted_end_time = end.strftime("%Y-%m-%d %H:%M:%S")
    LOGGER.info(f"Finished processing at :  {datetime.fromtimestamp(end)}")
    LOGGER.info(f"Finished main() in {duration_seconds:.2f} seconds.")
    LOGGER.info(f"Processed {frames_processed} frames at {frames_processed / duration_seconds:.2f} FPS.")
    


if __name__ == "__main__":
    video_source = DIRECTORY_CONFIG.source
    
    video_source = Path(video_source)
    if video_source.is_dir():
        video_source = [str(v) for v in video_source.iterdir() if v.suffix in ['.avi', '.mp4', '.h264', '.MTS']]
    elif type(video_source) is not list:
        video_source = [str(video_source)]


    parameter_combos = product(
        video_source
    )
    parameter_keys = [
        "video_source"
    ]

    for combo in parameter_combos:
        # Create a copy of the original CONFIG
        this_config_dict = DIRECTORY_CONFIG.to_dict()
        
        # Update the video_source field in the copied configuration
        this_config_dict["source"] = combo[0]  # combo is a tuple
        
        # Create a new Config object with the updated configuration
        this_config = Config(this_config_dict)

        # Pass this_config to the main function
        main(this_config)