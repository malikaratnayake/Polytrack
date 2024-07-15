import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from itertools import product
import cv2
import numpy as np
# from ultralytics import YOLO
from polytrack.InsectTracker import InsectTracker
from polytrack.InsectRecorder import Recorder
from polytrack.EventLogger import EventLogger
from polytrack.FlowerTracker import FlowerTracker
from polytrack.FlowerRecorder import FlowerRecorder

LOGGER = logging.getLogger()

# LOGGER = logging.getLogger(__name__)
# LOGGER.setLevel(logging.INFO)

class Config:
    def __init__(
        self,
        video_source: str,
        output_directory: str,
        max_occlusions: int,
        max_occlusions_edge: int,
        max_occlusions_on_flower: int,
        tracking_insects: list,
        output_video_dimensions: int,
        input_video_dimensions: int,
        insect_detector: str,
        insect_iou_threshold: float,
        dl_detection_confidence: float,
        min_blob_area: int,
        max_blob_area: int,
        downscale_factor: int,
        dilate_kernel_size: int,
        movement_threshold: int,
        compressed_video: bool,
        max_interframe_travel: int,
        info_filename: str,
        iou_threshold: float,
        model_insects_large: str,
        edge_pixels: int,
        show_video_output: bool,
        save_video_output: bool,
        video_codec: str,
        framerate: int,
        prediction_method: str,
        flower_detector: str,
        flower_iou_threshold: float,
        flower_detection_confidence: float,
        flower_classes: np.ndarray,
        flower_border: int,
        tracking_insect_classes: np.ndarray,
        track_flowers: bool,
        additional_new_insect_verification: bool,
        additional_new_insect_verification_confidence: list,
        insect_boundary_extension: float,
        black_pixel_threshold: float,
        flower_detection_interval: int

    ) -> None:

        self.video_source = video_source
        self.output_directory = output_directory
        self.max_occlusions = max_occlusions
        self.max_occlusions_edge = max_occlusions_edge
        self.max_occlusions_on_flower = max_occlusions_on_flower
        self.tracking_insects = tracking_insects
        self.output_video_dimensions = output_video_dimensions
        self.input_video_dimensions = input_video_dimensions
        self.insect_detector = insect_detector
        self.insect_iou_threshold = insect_iou_threshold
        self.dl_detection_confidence = dl_detection_confidence
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.downscale_factor = downscale_factor
        self.dilate_kernel_size = dilate_kernel_size
        self.movement_threshold = movement_threshold
        self.compressed_video = compressed_video
        self.max_interframe_travel = max_interframe_travel
        self.info_filename = info_filename
        self.iou_threshold = iou_threshold
        self.model_insects_large = model_insects_large
        self.edge_pixels = edge_pixels
        self.show_video_output = show_video_output
        self.save_video_output = save_video_output
        self.video_codec = video_codec
        self.framerate = framerate
        self.prediction_method = prediction_method
        self.flower_detector = flower_detector
        self.flower_iou_threshold = flower_iou_threshold
        self.flower_detection_confidence = flower_detection_confidence
        self.flower_classes = flower_classes
        self.flower_border = flower_border
        self.tracking_insect_classes = tracking_insect_classes
        self.track_flowers = track_flowers
        self.additional_new_insect_verification = additional_new_insect_verification
        self.additional_new_insect_verification_confidence = additional_new_insect_verification_confidence
        self.insect_boundary_extension = insect_boundary_extension
        self.black_pixel_threshold = black_pixel_threshold
        self.flower_detection_interval = flower_detection_interval





# Create Config object from JSON file
with open("config.json", "r") as f:
    __config_dict = json.load(f)

CONFIG = Config(**__config_dict)




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
                mapped_frame_num = self.TrackInsects.map_frame_number(nframe, compressed_video)
                fgbg_associated_detections, dl_associated_detections, missing_insects, new_insects = self.TrackInsects.run_tracker(frame, nframe, predicted_position)
                for_predictions = self.RecordTracks.record_track(frame, nframe, mapped_frame_num,fgbg_associated_detections, dl_associated_detections, missing_insects, new_insects)
                predicted_position = self.TrackInsects.predict_next(for_predictions)

                if ((self.compressed_video and (nframe in self.full_frame_numbers)) or (not self.compressed_video and (nframe == 5 or nframe % self.flower_detection_interval == 0))) and self.TrackFlowers is not None:
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






def main(config: Config):
    start = time.time()
    

    # Make sure opencv doesn't use too many threads and hog CPUs
    # cv2.setNumThreads(config.num_opencv_threads)

    # Use the input filepath to figure out the output filename
    if type(config.video_source) is str:
        output_filename = os.path.splitext(os.path.basename(config.video_source))[0]
    else:
        output_filename = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine the output directory based on user input

    if os.path.isdir(config.output_directory):
        output_parent_directory = Path(config.output_directory, "Polytrack")
        log_message = f"Outputting to {output_parent_directory}"
    else:
        output_parent_directory = Path(config.video_source, "Polytrack")
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
    with open(output_parent_directory / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=4)


    # Create all of our threads
    track_insects = InsectTracker(
        insect_detector = config.insect_detector,
        insect_iou_threshold = config.insect_iou_threshold,
        dl_detection_confidence = config.dl_detection_confidence,
        min_blob_area = config.min_blob_area,
        max_blob_area = config.max_blob_area,
        downscale_factor = config.downscale_factor,
        dilate_kernel_size = config.dilate_kernel_size,
        movement_threshold = config.movement_threshold,
        compressed_video = config.compressed_video,
        max_interframe_travel = config.max_interframe_travel,
        video_filepath = config.video_source,
        info_filename = config.info_filename,
        iou_threshold = config.iou_threshold,
        model_insects_large = config.model_insects_large,
        prediction_method = config.prediction_method,
        tracking_insect_classes = config.tracking_insect_classes,
        additional_new_insect_verification= config.additional_new_insect_verification,
        additional_new_insect_verification_confidence= config.additional_new_insect_verification_confidence,
        insect_boundary_extension = config.insect_boundary_extension,
        black_pixel_threshold = config.black_pixel_threshold)
    
    record_tracks = Recorder(
        input_video_dimensions = config.input_video_dimensions,
        output_video_dimensions = config.output_video_dimensions,
        video_source = config.video_source,
        framerate = config.framerate,
        output_directory = output_directory,
        show_video_output = config.show_video_output,
        save_video_output = config.save_video_output,
        video_codec = config.video_codec,
        max_occlusions = config.max_occlusions,
        max_occlusions_edge = config.max_occlusions_edge,
        max_occlusions_on_flower = config.max_occlusions_on_flower,
        tracking_insects = config.tracking_insects,
        edge_pixels = config.edge_pixels)
    
    if config.track_flowers:
        track_flowers = FlowerTracker(
            flower_detector = config.flower_detector,
            flower_iou_threshold = config.flower_iou_threshold,
            flower_detection_confidence = config.flower_detection_confidence,
            flower_classes = config.flower_classes,
            prediction_method = config.prediction_method)
        
        record_flowers = FlowerRecorder(
            output_directory = output_directory,
            flower_border = config.flower_border)
    
    track_and_record = TracknRecord(
        video_source = config.video_source,
        RecordTracks = record_tracks,
        TrackInsects = track_insects,
        TrackFlowers = track_flowers if config.track_flowers else None,
        RecordFlowers = record_flowers if config.track_flowers else None,
        flower_detection_interval = config.flower_detection_interval if config.track_flowers else None,
        compressed_video = config.compressed_video,
        info_filename = config.info_filename)
    
    
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
    video_source = CONFIG.video_source
    output_directory = CONFIG.output_directory
    max_occlusions = CONFIG.max_occlusions
    max_occlusions_edge = CONFIG.max_occlusions_edge
    max_occlusions_on_flower = CONFIG.max_occlusions_on_flower
    tracking_insects = CONFIG.tracking_insects
    input_video_dimensions = CONFIG.input_video_dimensions
    output_video_dimensions = CONFIG.output_video_dimensions
    insect_detector = CONFIG.insect_detector
    insect_iou_threshold = CONFIG.insect_iou_threshold
    dl_detection_confidence = CONFIG.dl_detection_confidence
    min_blob_area = CONFIG.min_blob_area
    max_blob_area = CONFIG.max_blob_area
    downscale_factor = CONFIG.downscale_factor
    dilate_kernel_size = CONFIG.dilate_kernel_size
    movement_threshold = CONFIG.movement_threshold
    compressed_video = CONFIG.compressed_video
    max_interframe_travel = CONFIG.max_interframe_travel
    info_filename = CONFIG.info_filename
    iou_threshold = CONFIG.iou_threshold
    model_insects_large = CONFIG.model_insects_large
    edge_pixels = CONFIG.edge_pixels
    show_video_output = CONFIG.show_video_output
    save_video_output = CONFIG.save_video_output
    video_codec = CONFIG.video_codec
    prediction_method = CONFIG.prediction_method
    flower_detector = CONFIG.flower_detector
    flower_iou_threshold = CONFIG.flower_iou_threshold
    flower_detection_confidence = CONFIG.flower_detection_confidence
    flower_classes = CONFIG.flower_classes
    flower_border = CONFIG.flower_border
    tracking_insect_classes = CONFIG.tracking_insect_classes
    track_flowers = CONFIG.track_flowers
    additional_new_insect_verification = CONFIG.additional_new_insect_verification
    additional_new_insect_verification_confidence = CONFIG.additional_new_insect_verification_confidence
    insect_boundary_extension = CONFIG.insect_boundary_extension
    black_pixel_threshold = CONFIG.black_pixel_threshold
    flower_detection_interval = CONFIG.flower_detection_interval
    

    video_source = Path(video_source)
    if video_source.is_dir():
        video_source = [str(v) for v in video_source.iterdir() if v.suffix in ['.avi', '.mp4', '.h264', '.MTS']]
    elif type(video_source) is not list:
        video_source = [str(video_source)]

    


    # if type(downscale_factor) is not list:
    #     downscale_factor = [downscale_factor]
    # if type(dilate_kernal_size) is not list:
    #     dilate_kernel_size = [dilate_kernal_size]
    # if type(movement_threshold) is not list:
    #     movement_threshold = [movement_threshold]


    parameter_combos = product(
        video_source
    )
    parameter_keys = [
        "video_source"
    ]
    # print("Length of parameter_combos:", len(parameter_combos))

    for combo in parameter_combos:
        this_config_dict = dict(zip(parameter_keys, combo))
        this_config_dict.update(
            {
                "output_directory": CONFIG.output_directory,
                "max_occlusions": CONFIG.max_occlusions,
                "max_occlusions_edge": CONFIG.max_occlusions_edge,
                "max_occlusions_on_flower": CONFIG.max_occlusions_on_flower,
                "tracking_insects": CONFIG.tracking_insects,
                "input_video_dimensions": CONFIG.input_video_dimensions,
                "output_video_dimensions": CONFIG.output_video_dimensions,
                "insect_detector": CONFIG.insect_detector,
                "insect_iou_threshold": CONFIG.insect_iou_threshold,
                "dl_detection_confidence": CONFIG.dl_detection_confidence,
                "min_blob_area": CONFIG.min_blob_area,
                "max_blob_area": CONFIG.max_blob_area,
                "downscale_factor": CONFIG.downscale_factor,
                "dilate_kernel_size": CONFIG.dilate_kernel_size,
                "movement_threshold": CONFIG.movement_threshold,
                "compressed_video": CONFIG.compressed_video,
                "max_interframe_travel": CONFIG.max_interframe_travel,
                "info_filename": CONFIG.info_filename,
                "iou_threshold": CONFIG.iou_threshold,
                "model_insects_large": CONFIG.model_insects_large,
                "edge_pixels": CONFIG.edge_pixels,
                "show_video_output": CONFIG.show_video_output,
                "save_video_output": CONFIG.save_video_output,
                "video_codec": CONFIG.video_codec,
                "framerate": CONFIG.framerate,
                "prediction_method": CONFIG.prediction_method,
                "flower_detector" : CONFIG.flower_detector,
                "flower_iou_threshold" : CONFIG.flower_iou_threshold,
                "flower_detection_confidence": CONFIG.flower_detection_confidence,
                "flower_classes" : CONFIG.flower_classes,
                "flower_border" : CONFIG.flower_border,
                "tracking_insect_classes" : CONFIG.tracking_insect_classes,
                "track_flowers" : CONFIG.track_flowers,
                "additional_new_insect_verification": CONFIG.additional_new_insect_verification,
                "additional_new_insect_verification_confidence": CONFIG.additional_new_insect_verification_confidence,
                "insect_boundary_extension": CONFIG.insect_boundary_extension,
                "black_pixel_threshold": CONFIG.black_pixel_threshold,
                "flower_detection_interval": CONFIG.flower_detection_interval
            }
            
        )
        this_config = Config(**this_config_dict)
        main(this_config)