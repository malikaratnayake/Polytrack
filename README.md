# Polytrack

## Introduction

Polytrack is designed to track multiple species of insect pollinators in diverse natural habitats or outdoor agricultural settings, monitoring their pollination behaviour in real-time. To achieve this, Polytrack uses a combination of foreground-background segmentation and deep learning-based object detection for accurate tracking.

### About this version
 This version of Polytrack (v3) will be continuously monitored and updated to incorporate new features and implement bug fixes. For previous versions of this software associated with publications, please refer to the repositories [Polytrack2.0](https://github.com/malikaratnayake/Polytrack2.0) and [Polytrack v1](https://github.com/malikaratnayake/Polytrack_v1).

### What's New

- **Enhanced object detection and streamlined integration:** Now powered by Ultralytics YOLOv8 for improved tracking and easier implementation.
- **Specialized models for targeted tracking:** Offers separate YOLOv8 models for insect tracking and flower tracking, tailored to specific use cases.
- **Simplified workflow:** A more intuitive and efficient user experience.
- **Additional insect verification:** A larger YOLOv8 model with higher recall and preceision can be used to verify new insect detections, minimising false positive track generation.
- **Motion compression video track reconstruction:** Explore the ability to reconstruct tracks from motion compression-based videos generated using [EcoMotionZip](https://github.com/malikaratnayake/EcoMotionZip).

## Installation and Dependencies

Polytrack utilizes OpenCV for image processing and Ultralytics YOLOv8 for deep learning-based object detection. Dependencies related to this code are provided in the requirements.txt and environment_polytrack.yml files.

### Training YOLOv8 Object detection model 

Polytrack uses a YOLOv8 object detection model to accurately detect insects and flowers in videos. It offers the option to use separate YOLOv8 models for insect and flower detection, enabling the use of existing annotated datasets. For more information on training YOLOv8 models, please refer to the YOLOv8 tutorials below.

- [How to Train YOLOv8 Object Detection on a Custom Dataset](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/)
- [Model Training with Ultralytics YOLO](https://docs.ultralytics.com/modes/train/)

Alternatively, you can download a pre-trained YOLOv8 model for detecting four insect types (Honeybee, Syrphids, Lepidoptera, and Vespids) and strawberry flowers [here.](https://drive.google.com/drive/folders/1yklR4kOtzVzgwknWcgWC7yKiby1JMjBt?usp=sharing). This model is associated with the research article [Spatial monitoring and insect behavioural analysis using computer vision for precision pollination](https://link.springer.com/article/10.1007/s11263-022-01715-4) and the [Spatial monitoring and insect behavioural analysis dataset](https://doi.org/10.26180/21533760).

## Usage

Code related to the core functionality of the Polytrack algorithm is located in the folder `polytrack` within this repository.

 Tracking parameters and working directories for the code can be specified in the file `./config.json`. Users have the option to specify either a single input video or a collection of videos. Descriptions of each tracking parameter are provided alongside its corresponding value.

 After declaring the relevant parameters, navigate to the root folder of the repository and use the following command to execute Polytrack:

```
python PolyTrack.py 
```

## Output

Polytrack will output the following tracking-related files. The output directory can be specified in the config file.

* Insect movement tracks with flower visit information (one track per detected insect)
* Snapshots of detected insects (for species verification, if required)
* Flower tracks
* Final positions of flowers (for visualisations)
  
In addition to the files mentioned above, users can choose to output a tracking video by selecting the appropriate option in the `config.json` file. This video will include only the instances where an insect is being tracked.


## Contact

If there are any inquiries, please don't hesitate to contact me at Malika DOT Ratnayake AT monash DOT edu.
 
## References
 
[1] The YOLOv8 component of this repository was adopted from [Model Training with Ultralytics YOLO](https://docs.ultralytics.com/modes/train/).

[2] Ratnayake, M. N., Amarathunga, D. C., Zaman, A., Dyer, A. G., & Dorin, A. (2023). [Spatial monitoring and insect behavioural analysis using computer vision for precision pollination.](https://rdcu.be/c0BsR) International Journal of Computer Vision, 131(3), 591-606.

[3] Ratnayake, M. N., Dyer, A. G., & Dorin, A. (2021). [Towards computer vision and deep learning facilitated pollination monitoring for agriculture.](https://openaccess.thecvf.com/content/CVPR2021W/AgriVision/html/Ratnayake_Towards_Computer_Vision_and_Deep_Learning_Facilitated_Pollination_Monitoring_for_CVPRW_2021_paper.html) In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 2921-2930).

[4] Ratnayake, M. N., Dyer, A. G., & Dorin, A. (2021). [Tracking individual honeybees among wildflower clusters with computer vision-facilitated pollinator monitoring.](https://doi.org/10.1371/journal.pone.0239504) Plos one, 16(2), e0239504.

