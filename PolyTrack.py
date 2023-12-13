#from __future__ import print_function
import os
import time
import sys
from polytrack.record import record_track, complete_tracking, setup_video_save
from polytrack.config import pt_cfg
from polytrack.flowers import track_flowers
from polytrack.general import *
from polytrack.tracker import  InsectTracker, LowResMode
import cv2
import numpy as np
from datetime import datetime
from absl import app
from absl import app, flags, logging
from absl.flags import FLAGS

# flags.DEFINE_string('input', pt_cfg.POLYTRACK.INPUT_DIR, 'path to input video directory')
# flags.DEFINE_string('extension', pt_cfg.POLYTRACK.VIDEO_EXT, 'Video extension of the input video')
# flags.DEFINE_string('output', pt_cfg.POLYTRACK.OUTPUT, 'path to output folder')
# flags.DEFINE_boolean('show_video', pt_cfg.POLYTRACK.SHOW_VIDEO_OUTPUT, 'Show video output')
#flags.DEFINE_boolean('save_video', pt_cfg.POLYTRACK.SAVE_VIDEO_OUTPUT, 'Save video output')

TrackInsect = InsectTracker(pt_cfg.POLYTRACK.INPUT_DIR)
LowResProcessor = LowResMode()


compressed_video =pt_cfg.POLYTRACK.COMPRESSED_VIDEO


def main(_argv):
    start_time = datetime.now()
    start_time_py = time.time()
    print("Start:  " + str(start_time))
    nframe = 0
    total_frames = 0
    predicted_position =[]
    idle = False
    processing_text= open(str(pt_cfg.POLYTRACK.OUTPUT)+"video_details.txt","w+") #Print processing time to a file

    video_list = get_video_list(pt_cfg.POLYTRACK.INPUT_DIR, pt_cfg.POLYTRACK.VIDEO_EXT)

    if compressed_video: _, actual_frame_num, frame_in_video = TrackInsect.get_video_info(pt_cfg.POLYTRACK.INPUT_DIR)


    for video_name in video_list:
        
        print('===================' + str(video_name) + '===================')

        video = str(pt_cfg.POLYTRACK.INPUT_DIR) + str(video_name)

        try:
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        
        width, height, video_frames = get_video_details(vid)
        pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS = get_video_start_time(video_name, nframe)

        if not pt_cfg.POLYTRACK.VIDEO_OUTPUT_SETUP:setup_video_save(pt_cfg.POLYTRACK.OUTPUT)
        
        if pt_cfg.POLYTRACK.SIGHTING_TIMES: 
            try:
                pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS = get_video_start_time(video_name, nframe)
            except:
                print('Invalied filename format. Try renaming the file or setting the value of SIGHTING_TIMES to False in Configuration file')
                pt_cfg.POLYTRACK.SIGHTING_TIMES = False

        total_frames += video_frames
        if compressed_video:
            act_nframe = actual_frame_num[0]
        else:
            act_nframe = nframe

        video_start_frame = act_nframe

        #nframe = 14400
        #vid.set(1, nframe)

        while True:
            return_value, frame = vid.read()
            if frame is not None:
                nframe += 1
                act_nframe = TrackInsect.map_frame_number(nframe, compressed_video)

                if (compressed_video and nframe in frame_in_video) or (not compressed_video and nframe % pt_cfg.POLYTRACK.FLOWER_UPDATE_FREQUENCY == 0):
                    # print('Updating flower positions', nframe)
                    track_flowers(act_nframe, frame)
                else:
                    pass

                # idle = False # Force the idle mode off
                idle = LowResProcessor.check_idle(nframe, predicted_position, compressed_video)
                # insectsBS =  BS_Detector.get_bs_detection(bg_sub_frame)
                # insectsBS =  foreground_changes(bg_sub_frame ,width, height, act_nframe, idle)

                possible_insects = LowResProcessor.process_frame(frame, compressed_video, idle)



                # associated_det_BS, associated_det_DL, missing,new_insect = track(frame, predicted_position, insectsBS)
                if possible_insects:
                    associated_det_BS, associated_det_DL, missing, new_insect = TrackInsect.track(compressed_video, frame, nframe,  predicted_position)
                # print(nframe, act_nframe,associated_det_BS, associated_det_DL, missing,new_insect)
                
                act_nframe, idle, new_insect = LowResProcessor.prepare_to_track(act_nframe, vid, idle, new_insect, video_start_frame)
                for_predictions = record_track(frame, act_nframe,associated_det_BS, associated_det_DL, missing, new_insect, idle)
                predicted_position = predict_next(for_predictions)

                #print(nframe, len(insectsBS), new_insect, for_predictions)
                

                fps = round(nframe/ (time.time() - start_time_py),2)
                #print(str(nframe) + ' out of ' + str(total_frames) + ' frames processed | ' + str(fps) + ' FPS | Tracking Mode:  '+ str(get_tracking_mode(idle)) , end = "\r")
                print(str(nframe) + ' out of ' + str(total_frames) + ' frames processed | ' + str(fps) +' FPS | Tracking Mode:  '+ str(get_tracking_mode(idle)) +'        ' , end = '\r') 
 
     
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
            else:
                print('Video has ended')
                break

        if not pt_cfg.POLYTRACK.CONTINUOUS_VIDEO:
            complete_tracking(predicted_position)
            predicted_position =[]
            pt_cfg.POLYTRACK.RECORDED_DARK_SPOTS = []
            flowers_recorded = False

    cv2.destroyAllWindows()
    complete_tracking(predicted_position)
    end_time = datetime.now()
    print()
    print("End:  " + str(end_time))
    print("Processing Time:  " + str(end_time-start_time))
    processing_text.write("Start:  " + str(start_time) + "\n End time: " +  str(end_time)+ "\n Processing time: " + str(end_time-start_time)+ "\n Frames: " + str(video_frames))
    processing_text.close() 





if __name__ == '__main__':
    # try:
    app.run(main)

    # except:
    #     complete_tracking([])
    #     # save_flowers()
    #     sys.exit(0)

