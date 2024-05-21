"""
This script trains a YOLOv8 model using Ultralytics and Roboflow.
"""

import os
import ultralytics
from ultralytics import YOLO
from roboflow import Roboflow

# Set the HOME directory path
HOME = "/home/mrat0006/bm75_scratch/mrat0006/yolov8/polytrack_lite"

# Run ultralytics checks
print("Ultralytics chscks: ", ultralytics.checks())

# Set the Roboflow dataset
# rf = Roboflow(api_key="DGcoUXklJTwwbZSIEgJb")
# project = rf.workspace("precpollination").project("polytrack-sunnyr")
# dataset = project.version(3).download("yolov8")

# Load the pretrained model
model = YOLO('yolov8m.pt')

_dataset = '/home/mrat0006/bm75_scratch/mrat0006/yolov8/polytrack_lite/datasets/polytrack_lite-flowers-2/data.yaml'
_epoches = 200
_image_size = 640
_devices = [0,1] # use 'mps' for mac
_save = True
_check_points = 25



# Train the model
results = model.train(data=_dataset, epochs=_epoches, imgsz=_image_size, device=_devices, save=_save,save_period=_check_points, plots = True)
