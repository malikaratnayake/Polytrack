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
                model_insects_large: str,
                insect_iou_threshold: float,
                dl_detection_confidence: float,
                dl_image_size: list,
                tracking_insect_classes: list,
                black_pixel_threshold: float) -> None:
        
        self.insect_detector = YOLO(insect_detector)
        self.tracking_insect_classes = tracking_insect_classes
        self.insect_iou_threshold = insect_iou_threshold
        self.dl_detection_confidence =dl_detection_confidence
        self.min_detector_confidence = min(dl_detection_confidence)
        self.model_insects_large = YOLO(model_insects_large)
        self.black_pixel_threshold = black_pixel_threshold
        self.dl_image_size = dl_image_size

        return None
    
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
                        _results: np.ndarray) -> np.ndarray:
        
        _insect_detection = np.zeros(shape=(0,5))

        for result in _results:
            insect_class = int(result[4])
            confidence = result[5]

            if confidence >= self.dl_detection_confidence[insect_class]:
                mid_x = int((result[0] + result[2])/2)
                mid_y = int((result[1] + result[3])/2)
                area = int(abs((result[0] - result[2])*(result[1] - result[3])))
                
                _insect_detection = np.vstack([_insect_detection,(mid_x, mid_y, area, insect_class, confidence)])

            else:
                pass

        return _insect_detection
        

    def run_dl_detector(self, 
                        frame: np.ndarray) -> np.ndarray:

        
        results = self.insect_detector.predict(source=frame, 
                                                conf=self.min_detector_confidence, 
                                                show=False, 
                                                verbose = False, 
                                                save = False,
                                                imgsz = (self.dl_image_size[1], self.dl_image_size[0]),
                                                iou = self.insect_iou_threshold, 
                                                classes = self.tracking_insect_classes)
        
        detections = self._decode_DL_results(results)
        processed_detections = self.__process_dl_detections(detections)

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
                                                                      imgsz = (image_size[1],image_size[0]))

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
                 compressed_video: bool,
                 video_filepath: str,
                 info_filename:str,
                 prediction_method:str,
                 show_fgbg_frame: bool) -> None:
        
        super().__init__(prediction_method = prediction_method)
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.downscale_factor = downscale_factor
        self.dilate_kernel_size = dilate_kernel_size
        self.movement_threshold = movement_threshold
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
        
        _possible_insects = np.zeros(shape=(0,3))

        for c in contours:
            cog_coord, _box_dims, _ = cv2.minAreaRect(c)
            cog_coord = [int(num * self.downscale_factor) for num in cog_coord]
            _box_dims = [int(num * self.downscale_factor) for num in _box_dims]
        
            _area = _box_dims[0] * _box_dims[1]

            if (_area > self.min_blob_area) and (_area<self.max_blob_area):
                _possible_insects = np.vstack([_possible_insects,(int(cog_coord[0]),int(cog_coord[1]),int(_area))])
            else:
                pass
        
        return _possible_insects
    

    def verify_detections(self,
                            detections: np.ndarray,
                            frame: np.ndarray) -> np.ndarray:
        
        for detection in detections:
            x, y, area = detection
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

        if self.compressed_video:
            detections = self.verify_detections(detections, frame)

        return detections

class InsectTracker(DL_Detector, FGBG_Detector):

    def __init__(self,
                 config: dict,
                 source_config: dict,
                 directory_config: dict) -> None:

        
        self.dl_detector, self.secondary_verification, self.fgbg_detector = self.detectors_in_use(config.detectors)
        
        if self.dl_detector is True or self.secondary_verification is True:
            DL_Detector.__init__(self,
                                insect_detector = config.detector_properties.dl_detection.model,
                                model_insects_large = config.detector_properties.secondary_verification.model,
                                insect_iou_threshold = config.detector_properties.dl_detection.iou_threshold,
                                dl_detection_confidence = config.detector_properties.dl_detection.detection_confidence,
                                dl_image_size = config.detector_properties.dl_detection.image_size,
                                tracking_insect_classes = config.classes,
                                black_pixel_threshold = config.detector_properties.secondary_verification.black_pixel_threshold)
        
        if self.fgbg_detector is True:
            FGBG_Detector.__init__(self,
                                    model = config.detector_properties.fgbg_detection.model,        
                                    min_blob_area = config.min_blob_area,
                                    max_blob_area = config.max_blob_area,
                                    downscale_factor = config.detector_properties.fgbg_detection.downscale_factor,
                                    dilate_kernel_size = config.detector_properties.fgbg_detection.dilate_kernel_size,
                                    movement_threshold = config.detector_properties.fgbg_detection.movement_threshold,
                                    compressed_video = source_config.compressed_video,
                                    video_filepath = directory_config.source,
                                    info_filename = source_config.compression_info,
                                    prediction_method = config.prediction_method,
                                    show_fgbg_frame = config.detector_properties.fgbg_detection.show)

        self.predictions = []
        self.max_interframe_travel = config.jump_distance
        self.compressed_video = source_config.compressed_video
        self.iou_threshold = config.iou_threshold
        self.insect_boundary_extension = config.insect_boundary_extension
       
        self.secondary_verification_confidence = config.detector_properties.secondary_verification.detection_confidence
        self.secondary_verification_imgsz = config.detector_properties.secondary_verification.image_size
        self.clean_fgbg_detections = config.detector_properties.fgbg_detection.clean_detections
        if self.clean_fgbg_detections:
            self.prev_fgbg_detection = None

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
                    predictions: np.ndarray) -> np.ndarray:
                
        
        self.predictions = predictions

        LOGGER.debug(f"Frame Number: {nframe}------------------------------")

        if self.fgbg_detector is True:
        
            fg_detections = self.run_fgbg_detector(frame, nframe)

            if self.clean_fgbg_detections:

                if self.prev_fgbg_detection is None:
                    self.prev_fgbg_detection = fg_detections

                fg_detections = self.clean_detections(self.prev_fgbg_detection, fg_detections)
                
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

        if self.dl_detector is True:

            # if ((len(fgbg_missing_insects) >0  or len(fgbg_unassociated_detections)>0 or (len(fg_detections) > len(self.predictions)) or (len(fg_detections) == 0 and self.compressed_video)) and ((self.compressed_video and (nframe not in self.full_frame_num)) or (not self.compressed_video))) or self.fgbg_detector is False:
                # dl_predictions = np.zeros(shape=(0,3))
                # for pred in np.arange(len(fgbg_missing_insects)):
                #     dl_predictions = np.vstack([dl_predictions,([row for row in self.predictions if fgbg_missing_insects[pred] == row[0]])])
            if (len(fgbg_missing_insects) > 0  
                or len(fgbg_unassociated_detections) > 0  
                or len(fg_detections) > len(self.predictions)  
                or (len(fg_detections) == 0 and self.compressed_video)  
                or self.fgbg_detector is False):    

                
                dl_detections = self.run_dl_detector(frame)

                # if len(fgbg_associated_detections) > 0: 
                #     fgbg_associated_positions = fgbg_associated_detections[:, 1:4]
                #     combined_fgbg_detections = [fgbg_associated_positions, fgbg_unassociated_detections]
                # else:
                #     combined_fgbg_detections = [fgbg_associated_detections, fgbg_unassociated_detections]

                dl_associated_detections, dl_missing_insects, potential_new_insects = self.process_detections(fg_detections, dl_detections, self.predictions)

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

                    fgbg_associated_detections, fgbg_missing_insects = [], []

                

                LOGGER.debug(f"DL Detection: {dl_detections},"
                             f"DL Associated Detections: {dl_associated_detections},"
                             f"DL Missing Insects: {dl_missing_insects},"
                             f"Possible New Insects: {potential_new_insects},"
                             f"New Insects: {new_insects}")
                

            else:
                dl_associated_detections, new_insects = [], []
                if len(fgbg_missing_insects)>0:
                    dl_missing_insects = fgbg_missing_insects
                else:
                    dl_missing_insects = []

        else:
            dl_associated_detections = []
            dl_missing_insects = fgbg_missing_insects
            new_insects = fgbg_unassociated_detections
        

        return (fgbg_associated_detections, dl_associated_detections, dl_missing_insects, new_insects)
    

    def clean_detections(self, prev_detections, current_detections):
        """
        Cleans `current_detections` by removing any detections close to `prev_detections`
        based on their x and y positions.

        Parameters:
            prev_detections (list of lists): List of previous detections [[x, y, a], ...].
            current_detections (list of lists): List of current detections [[x, y, a], ...].

        Returns:
            list of lists: Cleaned `current_detections`.
        """
        # Use a set for fast lookup of (x, y) positions
        prev_positions = {(x, y) for x, y, _ in prev_detections}

        # Filter current detections
        cleaned_detections = [det for det in current_detections if (det[0], det[1]) not in prev_positions]

        return cleaned_detections



        

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
            return np.zeros((0, 6)), predictions[:, 0].tolist() if predictions.size > 0 else [], np.array(detections)  # Ensure NumPy array

        # Assign detections to predictions using specified assignment method
        assign_func = self.assign_by_proximity if self.assignment_method == "ABP" else self.hungarian_assignment
        assignments = assign_func(detections=detections, predictions=predictions, cost_threshold=max_interframe_travel_distance)

        # Extract assigned indices
        assigned_det_indices, assigned_pred_indices = zip(*assignments) if assignments else ([], [])

        # Convert to NumPy arrays
        assigned_det_indices = np.array(assigned_det_indices, dtype=int)
        assigned_pred_indices = np.array(assigned_pred_indices, dtype=int)

        # Extract associated detections
        associated_detections = np.zeros((len(assignments), 6))
        for i, (det_idx, pred_idx) in enumerate(assignments):
            x, y, area, species, confidence = self.decode_detections(detections, det_idx)
            associated_detections[i] = [int(predictions[pred_idx, 0]), x, y, area, species, confidence]

        # Extract unassociated detections (ensure it's a NumPy array)
        unassociated_detections = np.array(np.delete(detections, assigned_det_indices, axis=0)) if len(assigned_det_indices) > 0 else np.array(detections)

        # Extract missing detections
        missing_detections = predictions[:, 0][np.setdiff1d(np.arange(len(predictions)), assigned_pred_indices)].tolist() if predictions.size > 0 else []

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



