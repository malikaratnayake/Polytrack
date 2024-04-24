import cv2
import threading

def run():
    cap = cv2.VideoCapture('/Users/mrat0010/Documents/GitHub/Polytrack_WIP/data/video/Ratnayake2023_processed/cam_7_S_video_20210310_142405.h264.avi')
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print("Video is over")
            break
        cv2.imshow('preview', frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    print("Bye :)")