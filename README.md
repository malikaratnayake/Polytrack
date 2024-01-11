# Polytrack

## Introduction

Polytrack is designed to track multiple species of insect pollinators in complex dynamic environments and to monitor their pollination behaviour. It uses a combination of foreground-background segmentation and deep learning-based object detection for tracking. Polytrack is capable of indentifying and recording insect-flower interactions for pollination monitoring.  

> ### New in this version:
> - Ultralytics YOLOv8 based object detection model for improved tracking and easier implementation.
> - Seperate YOLOv8 models for insect tracking and flower tracking.
> - Simplified workflow.
> - Ability to reconstruct tracks from motion compression based videos [Beta!].

## Installation and Dependencies

Polytrack uses OpenCV for image processing and Ultralytics YOLOv8 for deep learning-based object detection. Dependencies related to this code is provided in `requirements.txt` and `environment_polytrack.yml` files.

### Training YOLOv8 Object detection model 

Polytrack uses a YOLOv8 object detection model to accuratly detect insects and flowers in the video. Polytrack comes with an option to use seperate YOLOv8 models for insect and flower detection. This enables use of existing annoted datasets with Polytrack. For more information on how to train YOLv8 model, please refer to the YOLOv8 tutorials below. 

- [How to Train YOLOv8 Object Detection on a Custom Dataset](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/)
- [Model Training with Ultralytics YOLO](https://docs.ultralytics.com/modes/train/)

Alternatively, you can downloaded pretrained YOLOv8 model for detecting four insect types (Honeybee, Syrphids, Lepidoptera and Vespids) and strawberry flowers [here](https://drive.google.com/drive/folders/1HR-dEtR69Rl_2Su5Dk06OGpXLXWYFL2v?usp=sharing). This dataset is associated with the articles published in [International Journal of Computer Vision](https://link.springer.com/article/10.1007/s11263-022-01715-4).



## Usage

Code related to the core functionality of the Polytrack algorithm is in the folder "polytrack" of this repository.

Tracking parameters and working derectories of the code can be specified in the file "./polytrack/config.py". The user has the option of specifying a single input video or collection of videos. Descriptions related to the tracking parameters are defined alongside the parameter value.

After declaring relevant parameters, navigate to the root folder of the repository and run use the following command to run Polytrack.

```
python PolyTrack.py 
```

## Output

Polytrack will output following files related to tracking. The output directory can be in the config file.

* Insect movement tracks with flower visit information (One track per each detected insect).
* Snapshot of detected insects (For species verfication, if required).
* Flower tracks.
* Final position of flowers (For visualisations).

In addition to the above metioned files, user can select the option to output the tracking video in the config file. This will output a video that contains only the instances where an insect being tracked. 


## Contact

If there are any inquiries, please don't hesitate to contact me at Malika DOT Ratnayake AT monash DOT edu.
 
## References
 
The YOLOv8 component of this repository was adopted from [Model Training with Ultralytics YOLO](https://docs.ultralytics.com/modes/train/).
