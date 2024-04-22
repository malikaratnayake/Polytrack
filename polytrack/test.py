import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def predict_position(pos, vel, dt):
    return pos + vel * dt

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def track_objects(video_path, d=50, dt=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video file")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    tracks = []
    track_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, 150, 255, cv2.THRESH_BINARY)
        dilation_kernel = np.ones((8, 8))
        mask = cv2.medianBlur(cv2.dilate(thresh, kernel=dilation_kernel), 7)
        cv2.imshow('Mask', mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            centroid = np.array([x + w / 2, y + h / 2])
            detections.append(centroid)

        if tracks:
            predictions = [predict_position(track[-1][0], track[-1][1], dt) for track in tracks]
            cost_matrix = np.zeros((len(tracks), len(detections)))
            for i, prediction in enumerate(predictions):
                for j, detection in enumerate(detections):
                    cost_matrix[i][j] = distance(prediction, detection)

            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
            for ti, di in zip(track_indices, detection_indices):
                if cost_matrix[ti, di] < d:
                    tracks[ti].append((detections[di], (detections[di] - predictions[ti]) / dt))
                else:
                    tracks.append([(detections[di], np.array([0, 0]))])
        else:
            tracks = [[(detection, np.array([0, 0]))] for detection in detections]

        # Visualize tracks on the frame
        for track in tracks:
            for i in range(1, len(track)):
                prev_pos = (int(track[i-1][0][0]), int(track[i-1][0][1]))
                curr_pos = (int(track[i][0][0]), int(track[i][0][1]))
                cv2.line(frame, prev_pos, curr_pos, (0, 255, 0), 2)
                cv2.circle(frame, curr_pos, 4, (0, 255, 0), -1)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()

    return tracks

if __name__ == "__main__":
    video_path = "/Users/mrat0010/Documents/GitHub/EcoMotionZip/testing/test_videos/Ratnayake2023/cam_1_N_video_20210315_132804.h264.avi"
    tracked_objects = track_objects(video_path)
    print(tracked_objects)
