import os
import logging
import time
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from itertools import product
import importlib
import cv2
import numpy as np
from insect_tracker import InsectTracker
from insect_recorder import Recorder
from event_logger import EventLogger
from flower_tracker import FlowerTracker
from flower_recorder import FlowerRecorder
import argparse

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




class TracknRecord():

    def __init__(self,
                 video_source: str,
                 RecordTracks: Recorder,
                 TrackInsects: InsectTracker,
                 TrackFlowers: FlowerTracker,
                 RecordFlowers: FlowerRecorder,
                 compressed_video: bool,
                 info_filename: str,
                 skip_frames: bool,
                 flower_detection_interval:int,
                 video_index: int | None = None,
                 total_videos: int | None = None) -> None:
        
        self.video_source = video_source
        self.RecordTracks = RecordTracks
        self.TrackInsects = TrackInsects
        self.TrackFlowers = TrackFlowers
        self.RecordFlowers = RecordFlowers
        self.compressed_video = compressed_video
        self.info_filename = info_filename
        self.flower_detection_interval = flower_detection_interval
        self.skip_frames = skip_frames
        self.video_index = video_index
        self.total_videos = total_videos
        self.vid = cv2.VideoCapture(self.video_source)
        self.total_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)) if self.vid is not None else 0
        LOGGER.info(f"Processing video: {self.video_source}")        

        if self.compressed_video:
            _, _, self.full_frame_numbers = self.TrackInsects.get_compression_details(self.video_source, self.info_filename)

    def run(self) -> None:
        nframe = 0
        predicted_position = []
        flower_predictions = []
        start_time = time.time()
        processed_frames = 0
        bar_len = 30
        video_progress = ""
        while True:
            _, frame = self.vid.read()

            if frame is not None:
                nframe += 1
                if not self.skip_frames or (self.skip_frames and nframe % 2 != 0):
                    
                    mapped_frame_num = self.TrackInsects.map_frame_number(nframe, self.compressed_video)
                    processed_frames = mapped_frame_num
                    tracking_count, verified_count, saved_count, saved_verified = self.RecordTracks.get_tracking_stats()
                    flower_count = len(self.RecordTracks.latest_flower_positions) if hasattr(self.RecordTracks, "latest_flower_positions") else 0
                    if self.total_frames > 0:
                        progress_ratio = min(1.0, mapped_frame_num / self.total_frames)
                        filled = int(bar_len * progress_ratio)
                        bar = f"[{'#' * filled}{'-' * (bar_len - filled)}] {progress_ratio * 100:5.1f}%"
                        progress = f"{mapped_frame_num}/{self.total_frames}"
                    else:
                        bar = "[------------------------------]  ---.-%"
                        progress = f"{mapped_frame_num}"
                    print(
                        f"\r{Path(self.video_source).name} {bar} | {progress} frames processed | "
                        f"{tracking_count} active tracks ({verified_count} verified) | "
                        f"{saved_count} saved tracks ({saved_verified} verified) | "
                        f"{flower_count} flowers recorded",
                        end="",
                        flush=True,
                    )
                    unverified_track_ids = self.RecordTracks.get_unverified_track_ids()
                    fgbg_associated_detections, dl_associated_detections, missing_insects, new_insects, new_insects_fgbg, low_conf_associated_detections = self.TrackInsects.run_tracker(frame, nframe, predicted_position, unverified_track_ids)
                    for_predictions, current_insect_positions = self.RecordTracks.record_track(frame, nframe, mapped_frame_num, fgbg_associated_detections, dl_associated_detections, missing_insects, new_insects, new_insects_fgbg, low_conf_associated_detections)
                    predicted_position = self.TrackInsects.predict_next(for_predictions)


                    if self.TrackFlowers is not None and ((self.compressed_video and (nframe in self.full_frame_numbers)) or (not self.compressed_video and (nframe == 5 or nframe % self.flower_detection_interval == 0))):
                        associated_flower_detections, missing_flowers, new_flower_detections = self.TrackFlowers.run_flower_tracker(frame, flower_predictions)
                        flower_detections_for_predictions, latest_flower_positions = self.RecordFlowers.record_flowers(mapped_frame_num, associated_flower_detections, missing_flowers, new_flower_detections)
                        flower_predictions = self.TrackFlowers.predict_next(flower_detections_for_predictions)
                        self.RecordTracks.update_flower_positions(latest_flower_positions, self.RecordFlowers.flower_border)


                    if (len(for_predictions) > 0) and self.RecordFlowers is not None:
                        insect_flower_visits = self.RecordFlowers.monitor_flower_visits(current_insect_positions)
                        self.RecordFlowers.record_flower_visitations(insect_flower_visits, mapped_frame_num, self.RecordTracks.insect_tracks)
                        

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.RecordTracks.save_inprogress_tracks(predicted_position)
                    if self.RecordFlowers is not None:
                        self.RecordFlowers.save_flower_tracks()
                    self.RecordTracks.plot_track_summary()
                    break

            else:
                print()
                duration_seconds = max(0.0, time.time() - start_time)
                fps = (processed_frames / duration_seconds) if duration_seconds > 0 else 0.0
                tracking_count, verified_count, saved_count, saved_verified = self.RecordTracks.get_tracking_stats()
                flower_count = len(self.RecordTracks.latest_flower_positions) if hasattr(self.RecordTracks, "latest_flower_positions") else 0
                print(
                    f"Tracking finished for {Path(self.video_source).name} | "
                    f"{processed_frames} frames processed | "
                    f"{saved_count} tracks saved ({saved_verified} verified) | "
                    f"{flower_count} flowers recorded"
                )
                if self.video_index is not None and self.total_videos is not None:
                    print(f"Processing time: {duration_seconds:.2f}s | Processing FPS: {fps:.2f} | Completed {self.video_index}/{self.total_videos} videos.")
                else:
                    print(f"Processing time: {duration_seconds:.2f}s | Processing FPS: {fps:.2f}")
                LOGGER.info("Finished processing video. Exiting...")
                self.RecordTracks.save_inprogress_tracks(predicted_position)
                if self.RecordFlowers is not None: self.RecordFlowers.save_flower_tracks()
                self.RecordTracks.plot_track_summary()
                
                break

        self.vid.release()
        cv2.destroyAllWindows()

        return nframe

def get_video_properties(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        LOGGER.error(f"Error opening video source: {video_source}")
        return None, None

    ret, frame = cap.read()
    if not ret or frame is None:
        LOGGER.error(f"Couldn't read video stream from file \"{video_source}\"")
        cap.release()
        return None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if width <= 0 or height <= 0:
        height, width = frame.shape[:2]

    if width <= 0 or height <= 0 or framerate <= 0:
        LOGGER.error(f"Invalid video properties for {video_source}: {width}x{height} @ {framerate} FPS")
        return None, None

    LOGGER.info(f"Video resolution: {width}x{height}")
    LOGGER.info(f"Video framerate: {framerate}")

    return (width, height), framerate


def main(
    directory_config: Config,
    video_index: int | None = None,
    total_videos: int | None = None,
    override_output: bool = False,
    skip_existing: bool = False,
):

    start = time.time()

    if type(directory_config.source) is str:
        output_filename = os.path.splitext(os.path.basename(directory_config.source))[0]
    else:
        output_filename = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine the output directory based on user input

    if os.path.isdir(directory_config.output):
        output_parent_directory = Path(directory_config.output, "Polytrack")
        EventLogger.temp_log('info',f"Outputting to {output_parent_directory}")
    else:
        output_parent_directory = Path(os.path.dirname(directory_config.source), "Polytrack")
        EventLogger.temp_log('info',f"Output directory not specified or unavailable. Outputting to video source directory  {output_parent_directory}")


    # output_parent_directory = Path(config.output_directory , "EcoMotionZip")

    if not output_parent_directory.exists():
        os.makedirs(output_parent_directory, exist_ok=True)


    output_directory = Path(f"{output_parent_directory}/{output_filename}")
    # output_directory = Path(f"out/{output_filename}")
    if output_directory.exists():
        if override_output:
            shutil.rmtree(output_directory)
            output_directory.mkdir()
        elif skip_existing:
            LOGGER.info(f"Skipping {output_filename} (output directory exists).")
            return 0
        else:
            response = input(
                f"Output directory exists for {output_filename}. Override? [y/N]: "
            ).strip().lower()
            if response in ("y", "yes"):
                shutil.rmtree(output_directory)
                output_directory.mkdir()
            else:
                LOGGER.info(f"Skipping {output_filename} (output directory exists).")
                return 0
    else:
        output_directory.mkdir()

    EventLogger(output_directory, getattr(OUTPUT_CONFIG, "log_level", "INFO"))
    LOGGER.info(f"Outputting to {output_filename}")
    device = "cpu"
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            LOGGER.info(f"Available CUDA GPUs: {gpu_count} | {gpu_names}")
            device = "cuda:0"
            LOGGER.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            LOGGER.info("Available GPU: Apple MPS")
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        else:
            LOGGER.info("Using CPU (no GPU available)")
    else:
        LOGGER.info("Using CPU (torch not available)")

    device_override = getattr(SOURCE_CONFIG, "device", None)
    if device_override and str(device_override).lower() != "auto":
        device = str(device_override)
        LOGGER.info(f"Using device override: {device}")
    
    # Create a copy of the config file in the output directory
    #     Save the updated configurations back to the YAML file
    config_copy_yaml = {
        'directories': directory_config.to_dict(),
        'source': SOURCE_CONFIG.to_dict(),
        'output': OUTPUT_CONFIG.to_dict(),
        'insect_tracking': INSECT_CONFIG.to_dict(),
        'flower_tracking': FLOWER_CONFIG.to_dict()
    }
 
    # # Save the updated YAML file to the Data directory in format YYYYMMDD_HHMMSS.yaml
    with open(output_parent_directory / "config_copy.yaml", 'w') as f:
        yaml.dump(config_copy_yaml, f)

    # Update the output directory in the config file
    directory_config.output = output_directory

    video_resolution, framerate = get_video_properties(directory_config.source)
    if video_resolution is None or framerate is None:
        LOGGER.error(f"Skipping video due to unreadable/corrupt source: {directory_config.source}")
        return 0
    
    # Create all of our threads
    track_insects = InsectTracker(
        config = INSECT_CONFIG,
        source_config=SOURCE_CONFIG,
        directory_config=directory_config,
        device=device)
    
    
    record_tracks = Recorder(
        output_config=OUTPUT_CONFIG,
        insect_config=INSECT_CONFIG,
        source_config=SOURCE_CONFIG,
        flower_config=FLOWER_CONFIG,
        video_resolution = video_resolution,
        framerate = framerate,
        directory_config=directory_config)
    
    if FLOWER_CONFIG.track:
        track_flowers = FlowerTracker(
            config = FLOWER_CONFIG,
            device=device)
        
        record_flowers = FlowerRecorder(
            config = FLOWER_CONFIG,
            directory_config = directory_config,
            video_resolution = video_resolution)
    
    track_and_record = TracknRecord(
        video_source = directory_config.source,
        RecordTracks = record_tracks,
        TrackInsects = track_insects,
        TrackFlowers = track_flowers if FLOWER_CONFIG.track else None,
        RecordFlowers = record_flowers if FLOWER_CONFIG.track else None,
        flower_detection_interval = FLOWER_CONFIG.detection_interval if FLOWER_CONFIG.track else None,
        compressed_video = SOURCE_CONFIG.compressed_video,
        info_filename = SOURCE_CONFIG.compression_info,
        skip_frames = SOURCE_CONFIG.skip_frames,
        video_index = video_index,
        total_videos = total_videos)
    
    
    # Run the TracknRecord instance
    try:
        frames_processed = track_and_record.run()
    except KeyboardInterrupt:
        print()
        LOGGER.info("Keyboard interrupt received. Terminating process.")
        os._exit(1)


    # Add any extra stats/metadata to output too
    end = time.time()
    duration_seconds = end - start
    # formatted_end_time = end.strftime("%Y-%m-%d %H:%M:%S")
    LOGGER.info(f"Finished processing at :  {datetime.fromtimestamp(end)}")
    LOGGER.info(f"Finished main() in {duration_seconds:.2f} seconds.")
    LOGGER.info(f"Processed {frames_processed} frames at {frames_processed / duration_seconds:.2f} FPS.")
    


if __name__ == "__main__":

    EventLogger.temp_log('info',f"Starting processing at :  {datetime.fromtimestamp(time.time())}")

    default_config_directory = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

    # Parse command line arguments
    ap = argparse.ArgumentParser(prog='Polytrack',
                                 description='Polytrack is design to track unmarked freely foraging insects in outdoor environments and monitor their pollination behaviour.')
    ap.add_argument("--config", nargs='?', dest='custom_config', default=default_config_directory,
                help="Please Enter the directory of custom config.yaml file", type=str)
    ap.add_argument("--override-output", action="store_true", default=False,
                help="Override existing output directories without prompting.")
    ap.add_argument("--skip-existing", action="store_true", default=False,
                help="Skip videos with existing output directories without prompting.")
    
    args = ap.parse_args()
    config_directory = args.custom_config
    override_output = args.override_output
    skip_existing = args.skip_existing

    EventLogger.temp_log('info',f"Using config file: {config_directory}")

    # Load system configuration from YAML file
    with open(config_directory, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Create Config instances
    DIRECTORY_CONFIG = Config(yaml_config["directories"])
    INSECT_CONFIG = Config(yaml_config["insect_tracking"])
    FLOWER_CONFIG = Config(yaml_config["flower_tracking"])
    OUTPUT_CONFIG = Config(yaml_config["output"])
    SOURCE_CONFIG = Config(yaml_config["source"])
    
    
    video_source = DIRECTORY_CONFIG.source
    
    source_path = Path(video_source)
    if source_path.is_dir():
        video_source = [str(v) for v in source_path.iterdir() if v.suffix in ['.avi', '.mp4', '.h264', '.MTS']]
        if not video_source:
            video_source = [str(v) for v in source_path.rglob("*") if v.suffix in ['.avi', '.mp4', '.h264', '.MTS']]
        if not video_source:
            EventLogger.temp_log('info', f"No compatible videos found in {source_path} or its subdirectories.")
            raise SystemExit(0)
    elif type(video_source) is not list:
        video_source = [str(source_path)]


    parameter_combos = list(product(video_source))
    parameter_keys = [
        "video_source"
    ]

    total_videos = len(parameter_combos)
    for video_index, combo in enumerate(parameter_combos, start=1):
        # Create a copy of the original CONFIG
        this_config_dict = DIRECTORY_CONFIG.to_dict()
        
        # Update the video_source field in the copied configuration
        this_config_dict["source"] = combo[0]  # combo is a tuple
        
        # Create a new Config object with the updated configuration
        this_config = Config(this_config_dict)

        # Pass this_config to the main function
        main(
            this_config,
            video_index=video_index,
            total_videos=total_videos,
            override_output=override_output,
            skip_existing=skip_existing,
        )
