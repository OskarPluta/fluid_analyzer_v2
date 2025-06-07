import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from utils import detect_movement
from alternative import Preprocess


class Measure:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.preprocess = Preprocess()
        self.signal = self.preprocess.preprocess_video(filepath)
        self.start, self.stop = self.get_movement()
        # self.start = 1453
        # self.stop = 1493
    def get_movement(self):
        start, stop = detect_movement(self.signal, n_bkps=2)
        return start, stop

    def start_video_capture(self):
        cap = cv.VideoCapture(self.filepath)
        cap.set(cv.CAP_PROP_POS_FRAMES, self.stop)
        while 69:
            ret, frame = cap.read()
            if not ret:
                print("End of video or bullshit some error occurred")
                break
            cv.imshow('Video', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

    # def vertical_limits(self, frame: np.ndarray):
    #     """
        
    #     """

    #     if len(self.prev_yts) >= 1:
    #         avg_yt = int(np.mean(self.prev_yts))
    #     else:
    #         avg_yt = None

    #     if len(self.prev_ybs) >= 1:
    #         avg_yb = int(np.mean(self.prev_ybs))
    #     else:
    #         avg_yb = None

    #     if frame.shape[2] == 3:
    #         gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #     else:
    #         gray_frame = np.copy(frame)

    #     blurred_frame = cv.GaussianBlur(gray_frame, (7, 7), 0)
    #     edges = cv.Canny(blurred_frame, 50, 100, apertureSize=3)
    #     lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
    #                             minLineLength=50, maxLineGap=10)

    #     horizontal_lines = []
        
    #     if len(lines) == 0:
    #         return avg_yt, avg_yb
        
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         if abs(y2 - y1) < 10:
    #             if x1 > x2:
    #                 x1, y1, x2, y2 = x2, y2, x1, y1 # y1 is always the leftmost point y-coordinate
    #             horizontal_lines.append((x1, y1, x2, y2))
        
    #     if not horizontal_lines:
    #         return avg_yt, avg_yb

    #     top_line = min(horizontal_lines, key=lambda x: x[1]+x[3])
    #     bottom_line = max(horizontal_lines, key=lambda x: x[1]+x[3])

    #     upper_lines = []
    #     lower_lines = []


    #     for line in horizontal_lines:
    #         if (line[1]+line[3])/2 > top_line[1]-10 and (line[1]+line[3])/2 < top_line[1]+10:
    #             upper_lines.append(line)
    #         if (line[1]+line[3])/2 > bottom_line[1]-10 and (line[1]+line[3])/2 < bottom_line[1]+10:
    #             lower_lines.append(line)

    #     # yt1 and yb1 are the leftmost points of the upper and lower lines, respectively
    #     # yt2 and yb2 are the rightmost points of the upper and lower lines, respectively
    #     yt1 = min(upper_lines, key=lambda x: x[0])[1]
    #     yt2 = max(upper_lines, key=lambda x: x[2])[3]

    #     yb1 = min(lower_lines, key=lambda x: x[0])[1]
    #     yb2 = max(lower_lines, key=lambda x: x[2])[3]

    #     yt = int((yt1 + yt2) / 2)
    #     yb = int((yb1 + yb2) / 2)

    #     if avg_yt and abs(yt - avg_yt) > 10:
    #         yt = avg_yt
    #     if avg_yb and abs(yb - avg_yb) > 10:
    #         yb = avg_yb

    #     self.prev_ybs.append(yb)
    #     self.prev_yts.append(yt)

    #     yt = int(np.mean(self.prev_yts))
    #     yb = int(np.mean(self.prev_ybs))

    #     self.upper_line = yt
    #     self.lower_line = yb



if __name__ == "__main__":
    filepath = 'videos/nowy2.MP4'  # Replace with your video file path
    measure = Measure(filepath).start_video_capture()
    print(f"Movement starts at index: {measure.start}, stops at index: {measure.stop}")