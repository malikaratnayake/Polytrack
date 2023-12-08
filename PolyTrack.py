#from __future__ import print_function
import os
import time
import sys
# from polytrack.record import record_track, complete_tracking, setup_video_save
from polytrack.recorder import Recorder
from polytrack.config import pt_cfg
# from polytrack.flowers import track_flowers, save_flowers
# from polytrack.general import *
from polytrack.utilities import Utilities


import cv2
# import numpy as np
from datetime import datetime
from absl import app
# from absl import app, flags, logging
from absl.flags import FLAGS
from polytrack.tracker import  InsectTracker, LowResMode, FlowerTracker

# flags.DEFINE_string('input', pt_cfg.POLYTRACK.INPUT_DIR, 'path to input video directory')
# flags.DEFINE_string('extension', pt_cfg.POLYTRACK.VIDEO_EXT, 'Video extension of the input video')
# flags.DEFINE_string('output', pt_cfg.POLYTRACK.OUTPUT, 'path to output folder')
# flags.DEFINE_boolean('show_video', pt_cfg.POLYTRACK.SHOW_VIDEO_OUTPUT, 'Show video output')
# flags.DEFINE_boolean('save_video', pt_cfg.POLYTRACK.SAVE_VIDEO_OUTPUT, 'Save video output')
TrackUtilities = Utilities()
TrackInsect = InsectTracker()
TrackFlowers = FlowerTracker()
LowResProcessor = LowResMode()
compressed_video = pt_cfg.POLYTRACK.COMPRESSED_VIDEO


def main(_argv):
    
    nframe = 0
    total_frames = 0
    predicted_position =[]
    
    # output_directory = str(pt_cfg.POLYTRACK.OUTPUT)
    

    video_list = TrackUtilities.get_video_list(pt_cfg.POLYTRACK.INPUT_DIR, pt_cfg.POLYTRACK.VIDEO_EXT)# 

    for video_name in video_list:
        print('===================' + str(video_name) + '===================')

        start_time = datetime.now()
        start_time_py = time.time()
        print("Start:  " + str(start_time))
        idle = False

        if compressed_video: compressed_frame_num, actual_frame_num, frame_in_video = TrackInsect.get_video_info(pt_cfg.POLYTRACK.INPUT_DIR,video_name)
        
        

        video = str(pt_cfg.POLYTRACK.INPUT_DIR) + str(video_name)

        try:
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        if pt_cfg.POLYTRACK.CONTINUOUS_VIDEO: 
            pt_cfg.POLYTRACK.OUTPUT = pt_cfg.POLYTRACK.OUTPUT_DIR
        else:
            pt_cfg.POLYTRACK.OUTPUT = TrackUtilities.create_output_directory(pt_cfg.POLYTRACK.OUTPUT_DIR , video_name)

        processing_text= open(str(pt_cfg.POLYTRACK.OUTPUT)+"_video_details.txt","w+") #Print processing time to a file

        width, height, video_frames = TrackUtilities.get_video_details(vid)

        pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS = TrackUtilities.get_video_start_time(video_name, nframe)

        DataRecorder = Recorder()
        # RecordFlowers = FlowerRecorder()

        if not pt_cfg.POLYTRACK.VIDEO_OUTPUT_SETUP:DataRecorder.setup_video_save(pt_cfg.POLYTRACK.OUTPUT)
        
        if pt_cfg.POLYTRACK.SIGHTING_TIMES: 
            try:
                pt_cfg.POLYTRACK.CURRENT_VIDEO_DETAILS = TrackUtilities.get_video_start_time(video_name, nframe)
            except:
                print('Invalied filename format. Try renaming the file or setting the value of SIGHTING_TIMES to False in Configuration file')
                pt_cfg.POLYTRACK.SIGHTING_TIMES = False

        total_frames += video_frames
        if compressed_video:
            act_nframe = actual_frame_num[0]
        else:
            act_nframe = nframe

        video_start_frame = act_nframe

        while True:
            return_value, frame = vid.read()
            if frame is not None:
                nframe += 1
                act_nframe = TrackInsect.map_frame_number(nframe, compressed_video)

                if len(predicted_position) == 0 and (compressed_video and nframe in compressed_frame_num):
                    TrackInsect.reset_bg_model()

                if (compressed_video and nframe in frame_in_video) or (not compressed_video and nframe % pt_cfg.POLYTRACK.FLOWER_UPDATE_FREQUENCY == 1):
                    current_flower_details = DataRecorder.get_flower_details()
                    updated_flower_positions = DataRecorder.record_flower_positions(TrackFlowers.track_flowers(act_nframe, frame, current_flower_details))

                else:
                    pass
                    
                if compressed_video: 
                    audit_frame = TrackUtilities.audit_frame(nframe, frame_in_video)
                else:
                    audit_frame = False

                idle = LowResProcessor.check_idle(nframe, predicted_position, compressed_video)
                possible_insects = LowResProcessor.process_frame(frame, compressed_video, idle)

                if possible_insects:
                    associated_det_BS, associated_det_DL, missing, new_insect = TrackInsect.track(compressed_video, frame,nframe, audit_frame , predicted_position)
                
                act_nframe, idle, new_insect = LowResProcessor.prepare_to_track(act_nframe, vid, idle, new_insect, video_start_frame)
                for_predictions = DataRecorder.record_track(frame, act_nframe,associated_det_BS, associated_det_DL, missing, new_insect, updated_flower_positions ,idle)
                predicted_position = TrackInsect.predict_next(for_predictions)
                

                fps = round(nframe/ (time.time() - start_time_py),2)
                
                print(str(nframe) + ' out of ' + str(total_frames) + ' frames processed | ' + str(fps) +' FPS | Tracking Mode:  '+ str(TrackUtilities.get_tracking_mode(idle)) +'        ' , end = '\r') 
 
     
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
            else:
                print('Video has ended')
                break

        if not pt_cfg.POLYTRACK.CONTINUOUS_VIDEO:
            DataRecorder.complete_tracking(predicted_position)
            DataRecorder.save_flowers()
            predicted_position =[]
            flowers_recorded = False
            nframe = 0
            total_frames = 0
            idle = False

            cv2.destroyAllWindows()
            end_time = datetime.now()
            print("End:  " + str(end_time))
            print("Processing Time:  " + str(end_time-start_time))
            processing_text.write("Start:  " + str(start_time) + "\n End time: " +  str(end_time)+ "\n Processing time: " + str(end_time-start_time))
            processing_text.close() 
    
    if pt_cfg.POLYTRACK.CONTINUOUS_VIDEO:
        cv2.destroyAllWindows()
        DataRecorder.complete_tracking(predicted_position)
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
                # save_flowers()
        # sys.exit(0)

