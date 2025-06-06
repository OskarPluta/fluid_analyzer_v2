import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from utils import detect_movement


class Preprocess:

    def __init__(self, size:tuple = (640, 480)):

        self.size = size # resolution to find the edges of the fluid

        self.prev_still_x = None
        self.prev_still_xs = deque(maxlen=5)
        self.prev_moving_x = None
        self.prev_yts = deque(maxlen=5) # deque storing y-coordinates of the previous frame's upper line of the rheometer
        # to average it
        self.prev_ybs = deque(maxlen=5) # deque storing y-coordinates of the previous frame's lower line of the rheometer
        # to average it

        self.distances = [] # list to store distances between edges of the fluid
        self.diameters = [] # list to store diameters of the middle of thefluid
        self.distances_deque = deque(maxlen=10) # deque to store distances between edges of the fluid, to determine whether
        # the machine is moving or not

        self.movement_start = False # boolean to determine whether the machine has started moving
        self.movement_end = False # boolean to determine whether the machine has stopped moving 

        self.frame_index = 0 # frame index of the video
        self.movement_start_index = None # frame index of the first frame when the machine starts moving
        self.movement_end_index = None # frame index of the first frame when the machine stops moving

        # after it stops moving

        """Bellow are some bullshit variables that will get deleted probably."""

        self.start_measurement = False # boolean to determine whether the machine is moving or not
        self.last_five_whites = deque(maxlen=5) 
        self.xd_counts = [] # do wyjebania
        
    
    def preprocess_video(self, filepath: str):
        white_pixels_list = []
        cap = cv.VideoCapture(filepath)

        while 69:
            ret, frame = cap.read()
            if not ret:
                print("End of video or some bullshit error")
                break
            frame = cv.resize(frame, self.size)
            self.vertical_limits(frame)
            white_pixels = self.thresh_roi(frame)
            white_pixels_list.append(white_pixels)
        
        cap.release()
        return np.array(white_pixels_list)
        

    def vertical_limits(self, frame: np.ndarray):
        """
        
        """

        if len(self.prev_yts) >= 1:
            avg_yt = int(np.mean(self.prev_yts))
        else:
            avg_yt = None

        if len(self.prev_ybs) >= 1:
            avg_yb = int(np.mean(self.prev_ybs))
        else:
            avg_yb = None

        if frame.shape[2] == 3:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray_frame = np.copy(frame)

        blurred_frame = cv.GaussianBlur(gray_frame, (7, 7), 0)
        edges = cv.Canny(blurred_frame, 50, 100, apertureSize=3)
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=10)

        horizontal_lines = []
        
        if len(lines) == 0:
            return avg_yt, avg_yb
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:
                if x1 > x2:
                    x1, y1, x2, y2 = x2, y2, x1, y1 # y1 is always the leftmost point y-coordinate
                horizontal_lines.append((x1, y1, x2, y2))
        
        if not horizontal_lines:
            return avg_yt, avg_yb

        top_line = min(horizontal_lines, key=lambda x: x[1]+x[3])
        bottom_line = max(horizontal_lines, key=lambda x: x[1]+x[3])

        upper_lines = []
        lower_lines = []


        for line in horizontal_lines:
            if (line[1]+line[3])/2 > top_line[1]-10 and (line[1]+line[3])/2 < top_line[1]+10:
                upper_lines.append(line)
            if (line[1]+line[3])/2 > bottom_line[1]-10 and (line[1]+line[3])/2 < bottom_line[1]+10:
                lower_lines.append(line)

        # yt1 and yb1 are the leftmost points of the upper and lower lines, respectively
        # yt2 and yb2 are the rightmost points of the upper and lower lines, respectively
        yt1 = min(upper_lines, key=lambda x: x[0])[1]
        yt2 = max(upper_lines, key=lambda x: x[2])[3]

        yb1 = min(lower_lines, key=lambda x: x[0])[1]
        yb2 = max(lower_lines, key=lambda x: x[2])[3]

        yt = int((yt1 + yt2) / 2)
        yb = int((yb1 + yb2) / 2)

        if avg_yt and abs(yt - avg_yt) > 10:
            yt = avg_yt
        if avg_yb and abs(yb - avg_yb) > 10:
            yb = avg_yb

        self.prev_ybs.append(yb)
        self.prev_yts.append(yt)

        yt = int(np.mean(self.prev_yts))
        yb = int(np.mean(self.prev_ybs))

        self.upper_line = yt
        self.lower_line = yb

    def thresh_roi(self, frame: np.ndarray):
        """
        Threshold the region of interest (ROI) in the frame to isolate the fluid
        track expansion.

        Returns the number of white pixels in the thresholded ROI.
        """
        if frame.shape[2] == 3:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray_frame = np.copy(frame)


        roi = gray_frame[self.upper_line:self.lower_line, :]

        blurred_roi = cv.GaussianBlur(roi, (7, 7), 0)
        _, threshed_roi = cv.threshold(blurred_roi, 120, 255, cv.THRESH_BINARY)
        
        white_pixels = np.sum(threshed_roi == 255)

        return white_pixels
    
filepath = "videos/nowy2.MP4"
signal = Preprocess().preprocess_video(filepath)
start, stop = detect_movement(signal, n_bkps=2)
print(f"Start: {start}, Stop: {stop}")