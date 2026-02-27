import cv2
import numpy as np
from ultralytics import YOLO
import logging
from tracking_methods import TrackingMethods
import math

LOGGER = logging.getLogger()


class DL_Detector():

    def __init__(self,
                insect_detector: str,
                model_insects_large: str | None,
                insect_iou_threshold: float,
                dl_detection_confidence: float,
                dl_detection_confidence_floor: float | None,
                use_fp16: bool,
                dl_image_size: list,
                tracking_insect_classes: list,
                black_pixel_threshold: float | None,
                device: str) -> None:
        
        self.device = device
        self.insect_detector = YOLO(insect_detector)
        self.insect_detector.to(self.device)
        self.tracking_insect_classes = tracking_insect_classes
        self.insect_iou_threshold = insect_iou_threshold
        self.dl_detection_confidence =dl_detection_confidence
        self.dl_detection_confidence_floor = self._normalize_conf_list(
            dl_detection_confidence_floor, len(self.dl_detection_confidence), self.dl_detection_confidence
        )
        self.min_detector_confidence = min(self.dl_detection_confidence)
        if any(
            floor < conf for floor, conf in zip(self.dl_detection_confidence_floor, self.dl_detection_confidence)
        ):
            self.min_detector_confidence = min(self.dl_detection_confidence_floor)
        self.use_fp16 = bool(use_fp16) and str(self.device).startswith("cuda")
        self.model_insects_large = None
        if model_insects_large:
            self.model_insects_large = YOLO(model_insects_large)
            self.model_insects_large.to(self.device)
        self.black_pixel_threshold = black_pixel_threshold
        self.dl_image_size = dl_image_size
        self.prev_small_dl_candidates = np.zeros((0, 5))
        self.small_box_area_thresh = 500.0
        self.small_box_match_distance = 15.0
        self.low_confidence_floor = 0.05
        self.fg_proximity_distance = 20.0

        return None

    def _normalize_conf_list(self, values, target_len: int, fallback: list) -> list[float]:
        if values is None:
            return list(map(float, fallback))
        if isinstance(values, (list, tuple, np.ndarray)):
            vals = [float(v) for v in values]
        else:
            vals = [float(values)]
        if len(vals) < target_len:
            vals.extend([vals[-1]] * (target_len - len(vals)))
        return vals[:target_len]
    
    def _decode_DL_results(self, 
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
    
    def __process_dl_detections(self, 
                        _results: np.ndarray,
                        fg_detections: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        
        _insect_detection = np.zeros(shape=(0,7))
        _low_conf_detection = np.zeros(shape=(0,7))
        small_candidates = []
        fg_detections = np.asarray(fg_detections) if fg_detections is not None else np.zeros((0, 3))

        for result in _results:
            insect_class = int(result[4])
            confidence = result[5]
            mid_x = int((result[0] + result[2])/2)
            mid_y = int((result[1] + result[3])/2)
            box_w = float(abs(result[2] - result[0]))
            box_h = float(abs(result[3] - result[1]))
            area = int(abs((result[0] - result[2])*(result[1] - result[3])))
            if confidence >= self.dl_detection_confidence[insect_class]:
                _insect_detection = np.vstack([_insect_detection,(mid_x, mid_y, area, insect_class, confidence, box_w, box_h)])
            elif confidence >= self.dl_detection_confidence_floor[insect_class]:
                _low_conf_detection = np.vstack([_low_conf_detection,(mid_x, mid_y, area, insect_class, confidence, box_w, box_h)])

            if area <= self.small_box_area_thresh:
                small_candidates.append([mid_x, mid_y, area, insect_class, confidence])

        self.prev_small_dl_candidates = np.array(small_candidates) if small_candidates else np.zeros((0, 5))
        return _insect_detection, _low_conf_detection
        

    def run_dl_detector(self, 
                        frame: np.ndarray,
                        fg_detections: np.ndarray | None = None) -> np.ndarray:

        
        results = self.insect_detector.predict(source=frame, 
                                                conf=self.min_detector_confidence, 
                                                show=False, 
                                                verbose = False, 
                                                save = False,
                                                half=self.use_fp16,
                                                imgsz = (self.dl_image_size[1], self.dl_image_size[0]),
                                                iou = self.insect_iou_threshold, 
                                                classes = self.tracking_insect_classes,
                                                device=self.device)
        
        detections = self._decode_DL_results(results)
        processed_detections, low_conf_detections = self.__process_dl_detections(detections, fg_detections)
        self.last_low_confidence_detections = low_conf_detections

        return processed_detections
    
    def DL_verify_new_insects(self,
                            frame: np.ndarray,
                            potential_new_insects: np.ndarray,
                            secondary_verification_confidence: list,
                            image_size: list) -> list:
        
        low_confidence = []

        for dl_detection in np.arange(len(potential_new_insects)):

                
            mid_x = int(potential_new_insects[dl_detection][0])
            mid_y = int(potential_new_insects[dl_detection][1])
            insect_type = int(potential_new_insects[dl_detection][3])

            # Get frame width and height
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            insect_image = frame[max(mid_y-50,1):min(mid_y+50,frame_height-1), max(mid_x-50,1):min(mid_x+50,frame_width-1)]

            # Calculate black pixels in the image
            black_pixels = np.sum(insect_image == 0)

            # Calculate the percentage of black pixels in the image
            black_pixel_percentage = black_pixels / (insect_image.shape[0]*insect_image.shape[1])

            if black_pixel_percentage > self.black_pixel_threshold:
                low_confidence.append(dl_detection)

            else:

                x0 = max(0, int(mid_x - 160))
                y0 = max(0, int(mid_y - 160))
                x1 = min(int(mid_x + 160), frame_width)
                y1 = min(int(mid_y + 160), frame_height)

                croped_frame = frame[y0:y1, x0:x1] 

                # black_frame = np.zeros((480,864,3), np.uint8)
                black_frame = np.zeros((640,640,3), np.uint8)
                black_frame[100:100+croped_frame.shape[0], 100:100+croped_frame.shape[1]] = croped_frame

                crop = cv2.flip(black_frame, -1)

                # cv2.imshow("Crop", crop)

                confidence = secondary_verification_confidence[insect_type]


                new_insect_results = self.model_insects_large.predict(source=crop, 
                                                                      conf=confidence, 
                                                                      show=False, 
                                                                      verbose = False, 
                                                                      classes = [insect_type], 
                                                                      augment =True, 
                                                                      half=self.use_fp16,
                                                                      imgsz = (image_size[1],image_size[0]),
                                                                      device=self.device)

                new_insect_detections = self._decode_DL_results(new_insect_results)

                if len(new_insect_detections) == 0:
                    low_confidence.append(dl_detection)
                else:
                    LOGGER.debug(f"New insect verification: {new_insect_detections}")
    
        new_insects = np.delete(potential_new_insects, low_confidence, axis=0)

        return new_insects
    



class FGBG_Detector(TrackingMethods):

    
    last_full_frame = None

    def __init__(self,
                 model: list,
                 min_blob_area: int,
                 max_blob_area: int,
                 downscale_factor: int,
                 dilate_kernel_size: int,
                 movement_threshold: int,
                 warmup_frames: int,
                 compressed_video: bool,
                 video_filepath: str,
                 info_filename:str,
                 prediction_method:str,
                 show_fgbg_frame: bool,
                 max_fgbg_candidates: int | None = None) -> None:
        
        super().__init__(prediction_method = prediction_method)
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.max_fgbg_candidates = max_fgbg_candidates
        self.downscale_factor = downscale_factor
        self.dilate_kernel_size = dilate_kernel_size
        self.movement_threshold = movement_threshold
        self.warmup_frames = warmup_frames
        self.warmup_count = 0
        self.compressed_video = compressed_video
        self.video_filepath = video_filepath
        self.info_filename = info_filename
        self.show_fgbg_frame = show_fgbg_frame
        self.fgbf_model = model[0]
        if self.fgbf_model  == "MOG2":
            self.mog2_model = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=100, detectShadows=True)
        else:
            self.prev_frame = None

        if self.compressed_video:
            self.video_frame_num, self.actual_frame_num, self.full_frame_num = self.get_compression_details(self.video_filepath, self.info_filename)

        downscaled_kernel_size = int(dilate_kernel_size / downscale_factor)
        self.dilation_kernel = np.ones((downscaled_kernel_size, downscaled_kernel_size))
        self.fx = self.fy = 1 / self.downscale_factor

        return None
    
    def preprocess_frame(self, 
                         frame: np.ndarray,
                         nframe: int) -> np.ndarray:

        if self.downscale_factor == 1:
            downscaled_frame = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
        else:
            downscaled_frame = cv2.resize(cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY), dsize=None, fx=self.fx, fy=self.fy)
            
        if self.compressed_video:
            if nframe in self.full_frame_num:
                self.last_full_frame = downscaled_frame
                preprocessed_frame = downscaled_frame
            else:
                diff_frame = cv2.absdiff(downscaled_frame, self.last_full_frame)
                preprocessed_frame = cv2.add(downscaled_frame, diff_frame)
        else:
            preprocessed_frame = downscaled_frame

        return preprocessed_frame
    
    def detect_foreground_blobs(self,
                                frame: np.ndarray) -> np.ndarray:
        
        if self.fgbf_model != "MOG2" and self.prev_frame is None:
            self.prev_frame = frame
        
        # Compute pixel difference between consecutive frames 
        if self.fgbf_model == "MOG2":
            diff_frame = self.mog2_model.apply(frame)
            self.warmup_count += 1
            if self.warmup_count <= self.warmup_frames:
                return np.zeros_like(frame)
        else:
            diff_frame = cv2.absdiff(frame, self.prev_frame)

        # Convert to grayscale
        dilated_frame = cv2.dilate(diff_frame, kernel=self.dilation_kernel)

        # Cut off pixels that did not have "enough" movement. This is now a 2D array of just 1s and 0s
        _, threshed_diff = cv2.threshold(src=dilated_frame, thresh=self.movement_threshold, maxval=255, type=cv2.THRESH_BINARY)

        frame_mask = cv2.medianBlur(cv2.dilate(threshed_diff, kernel=self.dilation_kernel), 21) #31

        if self.show_fgbg_frame:
            cv2.imshow("Foreground Mask", frame_mask)

        if self.fgbf_model != "MOG2":
            self.prev_frame = frame

        return frame_mask
    

    def process_foreground_blobs(self, 
                                 contours: np.ndarray)-> np.ndarray:
        
        _possible_insects = np.zeros(shape=(0,7))

        for c in contours:
            cog_coord, _box_dims, _ = cv2.minAreaRect(c)
            cog_coord = [int(num * self.downscale_factor) for num in cog_coord]
            _box_dims = [int(num * self.downscale_factor) for num in _box_dims]
        
            _area = _box_dims[0] * _box_dims[1]

            if (_area > self.min_blob_area) and (_area<self.max_blob_area):
                _possible_insects = np.vstack([
                    _possible_insects,
                    (int(cog_coord[0]), int(cog_coord[1]), int(_area), 0, 0.0, float(_box_dims[0]), float(_box_dims[1])),
                ])
            else:
                pass
        
        return _possible_insects
    

    def verify_detections(self,
                            detections: np.ndarray,
                            frame: np.ndarray) -> np.ndarray:
        
        for detection in detections:
            x, y, area = detection[:3]
            x, y = int(x), int(y)  # Ensure x and y are integers
            
            # Extract insect image with bounds checking
            insect_image = frame[
                max(y - 1, 0):min(y + 1, frame.shape[0]),
                max(x - 1, 0):min(x + 1, frame.shape[1])
            ]

            # Check if all pixels in the cropped image are black
            if np.all(insect_image == 0):
                detections = np.delete(detections, np.where((detections == detection).all(axis=1)), axis=0)

        return detections
    

    def run_fgbg_detector(self,
                        frame: np.ndarray, 
                        nframe) -> np.ndarray:
        
        # if self.compressed_video:
        preprocessed_frame = self.preprocess_frame(frame, nframe)

        foreground_blobs = self.detect_foreground_blobs(preprocessed_frame)

        contours, _ = cv2.findContours(foreground_blobs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detections = self.process_foreground_blobs(contours)

        if self.max_fgbg_candidates is not None and self.max_fgbg_candidates > 0:
            if len(detections) > self.max_fgbg_candidates:
                LOGGER.debug(
                    f"Skipping FGBG detections (candidate count {len(detections)} > {self.max_fgbg_candidates})"
                )
                return np.zeros((0, 7))

        if self.compressed_video:
            detections = self.verify_detections(detections, frame)

        # print(detections)

        return detections

class InsectTracker(DL_Detector, FGBG_Detector):

    def __init__(self,
                 config: dict,
                 source_config: dict,
                 directory_config: dict,
                 device: str) -> None:

        
        TrackingMethods.__init__(self,
                                 prediction_method=config.prediction_method)
        self.dl_detector, self.secondary_verification, self.fgbg_detector = self.detectors_in_use(config.detectors)
        
        if self.dl_detector is True or self.secondary_verification is True:
            secondary_props = config.detector_properties.secondary_verification if self.secondary_verification else None
            DL_Detector.__init__(self,
                                insect_detector = config.detector_properties.dl_detection.model,
                                model_insects_large = secondary_props.model if secondary_props is not None else None,
                                insect_iou_threshold = config.detector_properties.dl_detection.iou_threshold,
                                dl_detection_confidence = config.detector_properties.dl_detection.detection_confidence,
                                dl_detection_confidence_floor = getattr(config.detector_properties.dl_detection, "detection_confidence_floor", None),
                                use_fp16 = getattr(config.detector_properties.dl_detection, "use_fp16", False),
                                dl_image_size = config.detector_properties.dl_detection.image_size,
                                tracking_insect_classes = config.classes,
                                black_pixel_threshold = secondary_props.black_pixel_threshold if secondary_props is not None else None,
                                device=device)
            dl_props = config.detector_properties.dl_detection
            self.small_box_area_thresh = getattr(dl_props, "small_box_area_thresh", self.small_box_area_thresh)
            self.small_box_match_distance = getattr(dl_props, "small_box_match_distance", self.small_box_match_distance)
            self.low_confidence_floor = getattr(dl_props, "low_confidence_floor", self.low_confidence_floor)
            self.fg_proximity_distance = getattr(dl_props, "fg_proximity_distance", self.fg_proximity_distance)
        
        if self.fgbg_detector is True:
            FGBG_Detector.__init__(self,
                                    model = config.detector_properties.fgbg_detection.model,        
                                    min_blob_area = config.min_blob_area,
                                    max_blob_area = config.max_blob_area,
                                    downscale_factor = config.detector_properties.fgbg_detection.downscale_factor,
                                    dilate_kernel_size = config.detector_properties.fgbg_detection.dilate_kernel_size,
                                    movement_threshold = config.detector_properties.fgbg_detection.movement_threshold,
                                    warmup_frames = config.detector_properties.fgbg_detection.warmup_frames,
                                    compressed_video = source_config.compressed_video,
                                    video_filepath = directory_config.source,
                                    info_filename = source_config.compression_info,
                                    prediction_method = config.prediction_method,
                                    show_fgbg_frame = config.detector_properties.fgbg_detection.show,
                                    max_fgbg_candidates = getattr(config.detector_properties.fgbg_detection, "max_fgbg_candidates", None))

        self.predictions = []
        self.max_interframe_travel = config.jump_distance
        self.compressed_video = source_config.compressed_video
        self.iou_threshold = config.iou_threshold
        self.insect_boundary_extension = config.insect_boundary_extension
       
        if self.secondary_verification:
            self.secondary_verification_confidence = config.detector_properties.secondary_verification.detection_confidence
            self.secondary_verification_imgsz = config.detector_properties.secondary_verification.image_size
        else:
            self.secondary_verification_confidence = []
            self.secondary_verification_imgsz = []
        self.clean_fgbg_detections = config.detector_properties.fgbg_detection.clean_detections
        if self.clean_fgbg_detections:
            self.prev_fgbg_detection = None
        fgbg_props = config.detector_properties.fgbg_detection
        self.clean_distance_thresh = getattr(fgbg_props, "clean_distance_thresh", 3.0)
        self.clean_area_ratio_thresh = getattr(fgbg_props, "clean_area_ratio_thresh", 0.25)
        self.missing_jump_scale = getattr(config, "missing_jump_scale", 1.5)
        self.associated_columns = 8

        try:
            self.assignment_method = config.assignment_method[0]
        except:
            self.assignment_method = "HungarianMethod"
        LOGGER.info(f"Assignment method: {self.assignment_method}")

        return None
    

    def detectors_in_use(self, 
                          detectors: list) -> tuple:
        
        dl_detection = "dl_detection" in detectors
        secondary_verification = "secondary_verification" in detectors
        fgbg_detection = "fgbg_detection" in detectors

        LOGGER.info(f"Detectors in use: Deep Learning Detector: {dl_detection}, "
                    f"Secondary Deep Learning Detector: {secondary_verification}, "
                    f"FGBG Detector: {fgbg_detection}")

        return dl_detection, secondary_verification, fgbg_detection


    def run_tracker(self, 
                    frame: np.ndarray,
                    nframe: int,
                    predictions: np.ndarray,
                    unverified_track_ids: list[int] | None = None) -> np.ndarray:
                
        
        self.predictions = predictions

        LOGGER.debug(f"Frame Number: {nframe}------------------------------")

        if unverified_track_ids is None:
            unverified_track_ids = []

        if self.fgbg_detector is True:
        
            fg_detections = self.run_fgbg_detector(frame, nframe)

            if self.clean_fgbg_detections:

                if self.prev_fgbg_detection is None:
                    self.prev_fgbg_detection = fg_detections

                fg_detections = self.clean_detections(
                    self.prev_fgbg_detection,
                    fg_detections,
                    distance_thresh=self.clean_distance_thresh,
                    area_ratio_thresh=self.clean_area_ratio_thresh,
                )
                
            dl_detections = None

            if len(fg_detections) > 0:
                fgbg_associated_detections, fgbg_missing_insects, fgbg_unassociated_detections = self.process_detections(fg_detections,
                                                                                                                        dl_detections, 
                                                                                                                        self.predictions)
            else:
                fgbg_associated_detections, fgbg_unassociated_detections = [], []
                fgbg_missing_insects = [i[0] for i in self.predictions]

            LOGGER.debug(
                f"FG detections: {fg_detections}, "
                f"FG associated detections: {fgbg_associated_detections}, "
                f"FG missing insects: {fgbg_missing_insects}, "
                f"FG unassociated detections: {fgbg_unassociated_detections}")

            if self.clean_fgbg_detections and len(fg_detections)>0:
                self.prev_fgbg_detection = fg_detections

        else:
            fgbg_associated_detections, fgbg_unassociated_detections, fg_detections = [], [], []
            fgbg_missing_insects = [i[0] for i in self.predictions]

        new_insects_fgbg = fgbg_unassociated_detections if self.fgbg_detector else []

        if self.dl_detector is True:

            # if ((len(fgbg_missing_insects) >0  or len(fgbg_unassociated_detections)>0 or (len(fg_detections) > len(self.predictions)) or (len(fg_detections) == 0 and self.compressed_video)) and ((self.compressed_video and (nframe not in self.full_frame_num)) or (not self.compressed_video))) or self.fgbg_detector is False:
                # dl_predictions = np.zeros(shape=(0,3))
                # for pred in np.arange(len(fgbg_missing_insects)):
                #     dl_predictions = np.vstack([dl_predictions,([row for row in self.predictions if fgbg_missing_insects[pred] == row[0]])])
            if ((len(fgbg_missing_insects) > 0 or len(fgbg_unassociated_detections) > 0)
                or len(unverified_track_ids) > 0
                or self.fgbg_detector is False):

                
                dl_detections = self.run_dl_detector(frame, fg_detections)
                low_conf_detections = getattr(self, "last_low_confidence_detections", np.zeros((0, 7)))

                # if len(fgbg_associated_detections) > 0: 
                #     fgbg_associated_positions = fgbg_associated_detections[:, 1:4]
                #     combined_fgbg_detections = [fgbg_associated_positions, fgbg_unassociated_detections]
                # else:
                #     combined_fgbg_detections = [fgbg_associated_detections, fgbg_unassociated_detections]

                dl_associated_detections, dl_missing_insects, potential_new_insects = self.process_detections(fg_detections, dl_detections, self.predictions)
                low_conf_associated_detections = np.zeros((0, self.associated_columns))
                if len(low_conf_detections) > 0 and len(unverified_track_ids) > 0 and len(self.predictions) > 0:
                    unverified_predictions = np.array(
                        [row for row in self.predictions if int(row[0]) in set(unverified_track_ids)]
                    )
                    if unverified_predictions.size > 0:
                        low_conf_associated_detections, _, _ = self.process_detections(
                            None, low_conf_detections, unverified_predictions
                        )

                if len(potential_new_insects) > 0 and len(dl_missing_insects) > 0:
                    relinked, potential_new_insects, dl_missing_insects = self.relink_missing_tracks(
                        potential_new_insects,
                        dl_missing_insects,
                        self.predictions,
                    )
                    if len(relinked) > 0:
                        dl_associated_detections = np.vstack([dl_associated_detections, relinked])

                if potential_new_insects.any() and (not self.compressed_video or (self.compressed_video and (nframe not in self.full_frame_num))):
                    new_insects = self.verify_new_insects(frame, potential_new_insects, fgbg_associated_detections)
                else:
                    new_insects = []

                if len(dl_missing_insects) > 0 and len(fgbg_associated_detections) > 0:
                    fg_predictions = np.zeros(shape=(0,3))

                    for pred in np.arange(len(dl_missing_insects)):
                        fg_predictions = np.vstack([fg_predictions,([row for row in self.predictions if dl_missing_insects[pred] == row[0]])])

                    fg_detections = self.remove_associated_detections(fg_detections, dl_associated_detections)

                    fgbg_associated_detections, fgbg_missing_insects, _ = self.process_detections(fg_detections, None, fg_predictions)

                    dl_missing_insects = fgbg_missing_insects

                else:
                    fgbg_associated_detections = []

                

                LOGGER.debug(f"DL Detection: {dl_detections},"
                             f"DL Associated Detections: {dl_associated_detections},"
                             f"DL Missing Insects: {dl_missing_insects},"
                             f"Possible New Insects: {potential_new_insects},"
                             f"New Insects: {new_insects}")
                

            else:
                dl_associated_detections, new_insects = [], []
                low_conf_associated_detections = np.zeros((0, self.associated_columns))
                if len(fgbg_missing_insects)>0:
                    dl_missing_insects = fgbg_missing_insects
                else:
                    dl_missing_insects = []

        else:
            dl_associated_detections = []
            dl_missing_insects = fgbg_missing_insects
            new_insects = fgbg_unassociated_detections
            low_conf_associated_detections = np.zeros((0, self.associated_columns))

        if self.fgbg_detector:
            base_missing = fgbg_missing_insects
        else:
            base_missing = dl_missing_insects
        dl_ids = set(int(det[0]) for det in dl_associated_detections) if len(dl_associated_detections) > 0 else set()
        missing_insects_out = [mid for mid in base_missing if mid not in dl_ids]

        return (
            fgbg_associated_detections,
            dl_associated_detections,
            missing_insects_out,
            new_insects,
            new_insects_fgbg,
            low_conf_associated_detections,
        )
    

    def clean_detections(self, prev_detections, current_detections, distance_thresh=3.0, area_ratio_thresh=0.25):
        """
        Removes near-duplicate detections using distance and area similarity.
        """
        prev_detections = np.asarray(prev_detections)
        current_detections = np.asarray(current_detections)

        if prev_detections.size == 0 or current_detections.size == 0:
            return current_detections

        cleaned = []
        for det in current_detections:
            x, y, area = det[:3]
            is_duplicate = False
            for px, py, parea in prev_detections[:, :3]:
                if np.hypot(x - px, y - py) <= distance_thresh:
                    if parea <= 0 or abs(area - parea) / parea <= area_ratio_thresh:
                        is_duplicate = True
                        break
            if not is_duplicate:
                cleaned.append(det)

        if not cleaned:
            return np.zeros((0, current_detections.shape[1]))

        return np.array(cleaned, dtype=float)

    def relink_missing_tracks(self, unassociated_detections, missing_ids, predictions, relink_scale=1.5):
        unassociated_detections = np.asarray(unassociated_detections)
        predictions = np.asarray(predictions) if isinstance(predictions, list) else predictions

        if unassociated_detections.size == 0 or predictions.size == 0 or len(missing_ids) == 0:
            return np.zeros((0, self.associated_columns)), unassociated_detections, missing_ids

        missing_mask = np.isin(predictions[:, 0], missing_ids)
        missing_predictions = predictions[missing_mask]

        if missing_predictions.size == 0:
            return np.zeros((0, self.associated_columns)), unassociated_detections, missing_ids

        max_dist = self.max_interframe_travel[0] * relink_scale
        assignments = self.assign_by_proximity(
            detections=unassociated_detections,
            predictions=missing_predictions,
            cost_threshold=max_dist,
        )

        if not assignments:
            return np.zeros((0, self.associated_columns)), unassociated_detections, missing_ids

        relinked = np.zeros((len(assignments), self.associated_columns))
        assigned_det_indices = []
        relinked_ids = []
        for i, (det_idx, pred_idx) in enumerate(assignments):
            x, y, area, species, confidence = self.decode_detections(unassociated_detections, det_idx)
            insect_id = int(missing_predictions[pred_idx, 0])
            relinked[i, :6] = [insect_id, x, y, area, species, confidence]
            box_w, box_h = self.decode_bbox_dims(unassociated_detections, det_idx)
            if box_w is not None and box_h is not None:
                relinked[i, 6:8] = [box_w, box_h]
            assigned_det_indices.append(det_idx)
            relinked_ids.append(insect_id)

        remaining_unassociated = np.delete(unassociated_detections, assigned_det_indices, axis=0)
        remaining_missing = [mid for mid in missing_ids if mid not in relinked_ids]

        return relinked, remaining_unassociated, remaining_missing

    def rescue_missing_assignments(self,
                                   detections: np.ndarray,
                                   predictions: np.ndarray,
                                   assigned_det_indices: np.ndarray,
                                   assigned_pred_indices: np.ndarray,
                                   max_interframe_travel_distance: float,
                                   rescue_scale: float):
        detections = np.asarray(detections)
        predictions = np.asarray(predictions)

        if detections.size == 0 or predictions.size == 0:
            return []

        unassigned_det_mask = np.ones(len(detections), dtype=bool)
        if len(assigned_det_indices) > 0:
            unassigned_det_mask[assigned_det_indices] = False
        unassigned_det_indices = np.where(unassigned_det_mask)[0]

        missing_pred_indices = np.setdiff1d(np.arange(len(predictions)), assigned_pred_indices)
        if len(unassigned_det_indices) == 0 or len(missing_pred_indices) == 0:
            return []

        rescue_threshold = max_interframe_travel_distance * rescue_scale
        candidates = []
        for pred_idx in missing_pred_indices:
            px, py = predictions[pred_idx, 1], predictions[pred_idx, 2]
            for det_idx in unassigned_det_indices:
                dx = detections[det_idx][0] - px
                dy = detections[det_idx][1] - py
                dist = float(np.hypot(dx, dy))
                if dist <= rescue_threshold:
                    candidates.append((dist, det_idx, pred_idx))

        candidates.sort(key=lambda x: x[0])
        used_det = set()
        used_pred = set()
        rescues = []
        for dist, det_idx, pred_idx in candidates:
            if det_idx in used_det or pred_idx in used_pred:
                continue
            rescues.append((det_idx, pred_idx, dist))
            used_det.add(det_idx)
            used_pred.add(pred_idx)

        return rescues



        

    def verify_new_insects(self,
                            frame: np.ndarray,
                            potential_new_insects: np.ndarray,
                            fgbg_associated_detections: np.ndarray) -> list:
          
        if fgbg_associated_detections is None:
            fgbg_associated_detections = []


        potential_new_insects = self.remove_associated_detections(potential_new_insects, fgbg_associated_detections)

        if self.secondary_verification:

            new_insects = self.DL_verify_new_insects(frame, potential_new_insects, self.secondary_verification_confidence, self.secondary_verification_imgsz)
        else:
            new_insects = potential_new_insects
    
        return new_insects
    
    
    

    def process_detections(self, fg_detections: np.ndarray, dl_detections: np.ndarray, predictions) -> tuple:
        """
        Associates detections (FG-BG or DL) with existing predictions and identifies unassociated detections.

        Args:
            fg_detections (np.ndarray): Foreground-background detections (shape: [N, 3+])
            dl_detections (np.ndarray): Deep learning detections (shape: [M, 3+])
            predictions (np.ndarray or list): Existing tracked insect predictions (shape: [P, 3+])

        Returns:
            tuple: (associated_detections, missing_detections, unassociated_detections)
        """

        # Convert predictions to NumPy array if it's a list
        predictions = np.array(predictions) if isinstance(predictions, list) else predictions

        # Determine which detection source to use
        use_dl = dl_detections is not None
        detections = dl_detections if use_dl else fg_detections
        max_interframe_travel_distance = self.max_interframe_travel[0] if use_dl else self.max_interframe_travel[1]

        if len(detections) == 0 or len(predictions) == 0:
            return (
                np.zeros((0, self.associated_columns)),
                predictions[:, 0].tolist() if predictions.size > 0 else [],
                np.array(detections),
            )  # Ensure NumPy array

        # Assign detections to predictions using specified assignment method
        assign_func = self.assign_by_proximity if self.assignment_method == "ABP" else self.hungarian_assignment
        assignments = assign_func(detections=detections, predictions=predictions, cost_threshold=max_interframe_travel_distance)

        # Extract assigned indices
        assigned_det_indices, assigned_pred_indices = zip(*assignments) if assignments else ([], [])

        # Convert to NumPy arrays
        assigned_det_indices = np.array(assigned_det_indices, dtype=int)
        assigned_pred_indices = np.array(assigned_pred_indices, dtype=int)

        rescues = self.rescue_missing_assignments(
            detections,
            predictions,
            assigned_det_indices,
            assigned_pred_indices,
            max_interframe_travel_distance,
            self.missing_jump_scale,
        )
        if rescues:
            for det_idx, pred_idx, _dist in rescues:
                assignments.append((det_idx, pred_idx))
            assigned_det_indices = np.array([a[0] for a in assignments], dtype=int)
            assigned_pred_indices = np.array([a[1] for a in assignments], dtype=int)

        # Extract associated detections
        associated_detections = np.zeros((len(assignments), self.associated_columns))
        for i, (det_idx, pred_idx) in enumerate(assignments):
            x, y, area, species, confidence = self.decode_detections(detections, det_idx)
            associated_detections[i, :6] = [int(predictions[pred_idx, 0]), x, y, area, species, confidence]
            box_w, box_h = self.decode_bbox_dims(detections, det_idx)
            if box_w is not None and box_h is not None:
                associated_detections[i, 6:8] = [box_w, box_h]

        # Extract unassociated detections (ensure it's a NumPy array)
        unassociated_detections = np.array(np.delete(detections, assigned_det_indices, axis=0)) if len(assigned_det_indices) > 0 else np.array(detections)

        # Extract missing detections
        missing_detections = predictions[:, 0][np.setdiff1d(np.arange(len(predictions)), assigned_pred_indices)].tolist() if predictions.size > 0 else []

        if LOGGER.isEnabledFor(logging.DEBUG):
            min_dists = []
            if len(detections) > 0:
                for i in range(len(predictions)):
                    px, py = predictions[i, 1], predictions[i, 2]
                    dists = np.hypot(detections[:, 0] - px, detections[:, 1] - py)
                    min_dists.append(float(np.min(dists)))
            else:
                min_dists = [None for _ in range(len(predictions))]
            missing_debug = []
            for idx in np.setdiff1d(np.arange(len(predictions)), assigned_pred_indices):
                missing_debug.append({
                    "id": int(predictions[idx, 0]),
                    "min_dist": min_dists[idx],
                })
            LOGGER.debug({
                "min_pred_to_det_dist": min_dists,
                "cost_threshold": max_interframe_travel_distance,
                "rescue_threshold": max_interframe_travel_distance * self.missing_jump_scale,
                "rescued_pairs": [(int(predictions[p[1], 0]), float(p[2])) for p in rescues] if rescues else [],
                "missing_after_rescue": missing_debug,
            })

        # Debug Logging (Ensure all logged data is NumPy arrays)
        LOGGER.debug({
            "Associated Detections": associated_detections.tolist(),
            "Missing Detections": missing_detections,
            "Unassociated Detections": unassociated_detections.tolist() if isinstance(unassociated_detections, np.ndarray) else unassociated_detections
        })

        # Apply filtering for DL-based detections if enabled
        if self.dl_detector and use_dl:
            unassociated_detections = self.remove_associated_detections(unassociated_detections, associated_detections)

            # Filter out detections with zero confidence
            unassociated_detections = unassociated_detections[unassociated_detections[:, 4] > 0]

        return associated_detections, missing_detections, unassociated_detections



    
    
    def decode_detections(self, 
                          detections: np.ndarray, 
                          insect_num: int):
        
        _center_x = int(detections[insect_num][0])
        _center_y = int(detections[insect_num][1])
        _area = int(detections[insect_num][2])

        if len(detections[insect_num]) > 3:
            _species = detections[insect_num][3]
            _confidence = detections[insect_num][4]
        else:
            _species = 0
            _confidence = 0

        return _center_x, _center_y, _area, _species, _confidence

    def decode_bbox_dims(self, detections: np.ndarray, insect_num: int):
        if detections is None or len(detections) == 0:
            return None, None
        if len(detections[insect_num]) >= 7:
            return float(detections[insect_num][5]), float(detections[insect_num][6])
        return None, None
    

    def remove_associated_detections(self, dl_detections: np.ndarray, bs_associated_detections: np.ndarray) -> np.ndarray:
        """
        Removes overlapping DL detections that have high IoU with baseline associated detections.

        Args:
            dl_detections (np.ndarray): DL detections (shape: [N, >=3], containing x, y, area, ...).
            bs_associated_detections (np.ndarray): BS-associated detections (shape: [M, >=4], containing ID, x, y, area, ...).

        Returns:
            np.ndarray: Filtered DL detections.
        """
        if len(dl_detections) == 0 or len(bs_associated_detections) == 0:
            return dl_detections  # No processing needed if either is empty

        # Extract only (x, y, area) from BS-associated detections
        bs_boxes = bs_associated_detections[:, 1:4]  

        # Compute IoU for all DL detections against BS-associated detections using broadcasting
        iou_matrix = np.array([
            [self.calculate_iou(bs_box[:3], dl_box[:3], self.insect_boundary_extension) for dl_box in dl_detections]
            for bs_box in bs_boxes
        ])

        # Find indices of overlapping detections
        duplicate_indices = np.unique(np.where(iou_matrix > self.iou_threshold)[1])

        # Remove duplicate detections
        filtered_detections = np.delete(dl_detections, duplicate_indices, axis=0)

        # Debugging logs
        LOGGER.debug({
            "Total DL Detections": len(dl_detections),
            "Total BS Associated Detections": len(bs_associated_detections),
            "Duplicate Indices": duplicate_indices.tolist(),
            "Remaining Detections": len(filtered_detections)
        })

        return filtered_detections
    
    def remove_duplicate_detections(self, 
                                    dl_detections: np.ndarray, 
                                    bs_detections: np.ndarray) -> np.ndarray:
        """
        Removes duplicate detections from the deep learning (DL) detections by comparing them 
        to the baseline (BS) detections using Intersection over Union (IoU).

        Args:
            dl_detections (np.ndarray): Array of detections from deep learning model (shape: [N, 3])
            bs_detections (np.ndarray): Array of baseline detections (shape: [M, 3])

        Returns:
            np.ndarray: Filtered DL detections with duplicates removed.
        """
        if len(dl_detections) == 0 or len(bs_detections) == 0:
            return dl_detections  # No duplicates to remove if either list is empty

        # Compute IoU for all DL detections against BS detections using broadcasting
        iou_matrix = np.array([
            [self.calculate_iou(bs_box, dl_box, self.insect_boundary_extension) for dl_box in dl_detections]
            for bs_box in bs_detections
        ])

        # Get indices of duplicate detections where IoU exceeds the threshold
        duplicate_indices = np.unique(np.where(iou_matrix > self.iou_threshold)[1])

        # Remove duplicates
        filtered_detections = np.delete(dl_detections, duplicate_indices, axis=0)

        # Logging for debugging
        LOGGER.debug({
            "Total DL Detections": len(dl_detections),
            "Total BS Detections": len(bs_detections),
            "Duplicate Indices": duplicate_indices.tolist(),
            "Remaining Detections": len(filtered_detections)
        })

        return filtered_detections
    

    

    def calculate_iou(self, bs_bounding_box: np.ndarray, dl_bounding_box: np.ndarray, insect_boundary_extension: float) -> float:
        """
        Computes the Intersection over Union (IoU) between two bounding boxes.

        Args:
            bs_bounding_box (np.ndarray): [x, y, area] of the baseline detection.
            dl_bounding_box (np.ndarray): [x, y, area] of the DL detection.
            insect_boundary_extension (float): Expansion factor for boundary calculations.

        Returns:
            float: IoU score between the two bounding boxes.
        """
        x1, y1, area1 = bs_bounding_box
        x2, y2, area2, *_ = dl_bounding_box  # Ensure only the first three values are used

        # Compute bounding box radius
        radius1 = np.sqrt(area1 / np.pi) * insect_boundary_extension
        radius2 = np.sqrt(area2 / np.pi) * insect_boundary_extension

        # Compute bounding box coordinates
        x1_min, y1_min, x1_max, y1_max = x1 - radius1, y1 - radius1, x1 + radius1, y1 + radius1
        x2_min, y2_min, x2_max, y2_max = x2 - radius2, y2 - radius2, x2 + radius2, y2 + radius2

        # Compute intersection
        inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        inter_area = inter_w * inter_h

        # Compute union
        union_area = (2 * radius1) ** 2 + (2 * radius2) ** 2 - inter_area

        # Compute IoU
        iou = inter_area / union_area if union_area > 0 else 0

        # Debugging logs
        LOGGER.debug({
            "BS Box": bs_bounding_box.tolist(),
            "DL Box": dl_bounding_box.tolist(),
            "Intersection Area": inter_area,
            "Union Area": union_area,
            "IoU": iou
        })

        return iou
