from datetime import datetime
import glob
import math
import threading
from time import time
import numpy as np
from ultralytics import YOLO
import os
import shutil
import sys
import cv2
from PIL import Image

# set the path to the YOLOv5 model
yolo_path = "./data/yolov8_models/yolo-stingless-best.pt"

# global constants
ROI_HEIGHT = 500
ROI_WIDTH = 500
Resize_height = 256
Resize_width = 256
VideoExtension = ".mp4"
ImageExtension = ".png"

def process_yolo_results(result, class_names, conf_thresh=0.5):
    """
    processes the yolo results and returns the frame with the bounding boxes.
    results: list of results from yolo inference
    frame: np.array
    class_names: dict
    draw_bounding_box: bool
    return: frame, np.array
    """
    box_list = {}
    for i in range(len(class_names)):
        box_list[class_names[i]] = []
    if result is not None and len(result.boxes.cpu().numpy().data)>0:
        
        # class_names = {0: 'fly', 1: 'tag'}
        data = result.boxes.cpu().numpy().data
        boxes = data[:, 0:4].astype(int)
        confs = data[:, 4]
        class_ids = data[:, 5].astype(int)
        # for each box, draw the bounding box and label
        for i, box in enumerate(boxes):
            if confs[i] > conf_thresh and class_ids[i] in class_names:
                box_list[class_names[class_ids[i]]].append(box)

    return box_list

def draw_bounding_box(frame, box, class_name, color=(0, 105, 255)): # color for ocean blue (0, 105, 255)
    """
    Draw the bounding box on the frame
    frame: np.array
    box: tuple (x1, y1, x2, y2)
    class_id: int
    class_names: dict
    return: np.array
    """
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
    # cv2.putText(frame, class_name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame

def enque_deque_frame(frame, type='raw', enque=True):
    """
    enque or deque the frame from the queue
    frame: np.array
    type: str
    enque: bool
    return: None
    """
    global raw_frame_queue, processed_frame_queue
    if type == 'raw':
        if enque:
            raw_frame_queue.append(frame)
        else:
            frame = raw_frame_queue.popleft()
    elif type == 'processed':
        if enque:
            processed_frame_queue.append(frame)
        else:
            frame = processed_frame_queue.popleft()
    return frame

def get_batch_frames(batch_size=16):
    """
    get a batch of frames from the queue
    batch_size: int
    return: list of np.array
    """
    global raw_frame_queue
    batch_frames = []
    while len(batch_frames) < batch_size:
        if len(raw_frame_queue) > 0:
            frame = enque_deque_frame(None, type='raw', enque=False)
            batch_frames.append(frame)
        else:
            break
    return batch_frames

def process_video_v2(video_path, yolo, output_video_path):
    """
    Process the dronefly video
    video_path: str
    yolo: YOLO
    class_names: dict
    output_path: str
    output_video_path: str
    output_video_fps: int
    """
    class_names = {0: 'bee'}
    # load the video
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get the video's frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    # get the video's frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    t1 = time()
    frame_batch = []
    batch_size = 32
    csv_string = ""
    frame_ids = []

    # process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if len(frame_batch) < batch_size:
            frame_batch.append(frame)
            frame_ids.append(frame_no)
            continue
        # process the frame
        results = yolo(frame_batch, verbose=False)
        frame_counter = 0
        for result in results:
            frame = frame_batch[frame_counter]
            frame_counter += 1
            box_list = process_yolo_results(result, class_names, conf_thresh=0.4)
            bee_count = len(box_list['bee'])
            
            for class_name, boxes in box_list.items():
                for box in boxes:
                    frame = draw_bounding_box(frame, box, class_name)
                    center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                    csv_string += f"{frame_ids[frame_counter-1]},{center[0]},{center[1]}\n"
            # write the frame to the output video
            t = time() - t1
            fps = frame_no / t
            cv2.putText(frame, f"FPS: {fps:0.2f}, Bee Count: {bee_count:2d}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            out.write(frame)
            print(f"Processed frame: {frame_no:5d}/{n_frames:5d} ({frame_no/n_frames:.2%}), fps={fps:0.2f}", end='\r')
        frame_batch = []
        frame_ids = []
    # release the video capture and video writer objects
    cap.release()
    # out.release()
    csv_file = output_video_path.replace(VideoExtension, "_detections.csv")
    with open(csv_file, 'w') as f:
        f.write(csv_string)

    print(f"\nProcessed video: {video_path} -> {output_video_path} and {csv_file}")



def process_video(video_path, yolo, output_video_path):
    """
    Process the dronefly video
    video_path: str
    yolo: YOLO
    class_names: dict
    output_path: str
    output_video_path: str
    output_video_fps: int
    """
    class_names = {0: 'bee'}
    # load the video
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get the video's frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    # get the video's frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    t1 = time()
    # process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # process the frame
        results = yolo(frame, verbose=False)
        box_list = process_yolo_results(results, class_names, conf_thresh=0.4)
        bee_count = len(box_list['bee'])
        for class_name, boxes in box_list.items():
            for box in boxes:
                frame = draw_bounding_box(frame, box, class_name)
        # write the frame to the output video
        t = time() - t1
        fps = frame_no / t
        cv2.putText(frame, f"FPS: {fps:0.2f}, Bee Count: {bee_count:2d}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        out.write(frame)
        print(f"Processed frame: {frame_no:5d}/{n_frames:5d} ({frame_no/n_frames:.2%}), fps={fps:0.2f}", end='\r')
    # release the video capture and video writer objects
    cap.release()
    out.release()
    print(f"\nProcessed video: {video_path} -> {output_video_path}")

if __name__ == '__main__':
    # load the YOLO model
    yolo = YOLO(yolo_path)
    yolo.to('cuda:0')
    print("Loaded YOLO model successfully to GPU")
    # process the dronefly video
    path= "/Users/mrat0010/Documents/GitHub/Polytrack_WIP/data/video/Camera_43_20240422_113145/Camera_43_20240422_130601.avi"

    video_list = glob.glob(path)
    print(f"Found {len(video_list)} videos to process")
    os.makedirs('../../DATASETS/BeeHiveMonitoring', exist_ok=True)
    for video_path in video_list:
        print(f"Processing video: {video_path}")
        file_base_name = os.path.basename(video_path)
        output_video_path = os.path.join('../../DATASETS/BeeHiveMonitoring', file_base_name)
        print(f"Output video path: {output_video_path}")
        # # output_video_path = video_path.replace('./Videos/', './Output/')
        # if os.path.exists(output_video_path):
        #     print(f"Output video already exists: {output_video_path}")
        #     # continue
        process_video_v2(video_path, yolo, output_video_path)
    