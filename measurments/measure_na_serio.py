import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from utils import detect_movement, vertical_limits
from alternative import Preprocess


class Measure:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.preprocess = Preprocess()
        self.signal = self.preprocess.preprocess_video(filepath)

        self.prev_bots = deque(maxlen=5)  # previous bottom line positions
        self.prev_tops = deque(maxlen=5)  # previous top line positions
        
        self.size = (640, 480)  
        # self.start = 1453
        # self.stop = 1492 # stop real 2683 for 3
        self.start, self.stop = self.get_movement()
        # self.start, self.stop = 2031, 2082

    def get_movement(self):
        start, stop = detect_movement(self.signal, n_bkps=2)
        print(start, stop)
        return start, stop

    def start_video_capture(self):
        cap = cv.VideoCapture(self.filepath)
        cap.set(cv.CAP_PROP_POS_FRAMES, self.start)
        while 69:
            ret, frame = cap.read()
            if not ret:
                print("End of video or bullshit some error occurred")
                break
            frame = cv.resize(frame, self.size)
            # limits = vertical_limits(frame, self.prev_tops, self.prev_bots)
            top_limit, bottom_limit = vertical_limits(frame, self.prev_tops, self.prev_bots, self.size)
            # frame = frame[top_limit:bottom_limit, :]  
            frame = frame[int(np.mean(top_limit)):int(np.mean(bottom_limit))]
            horizontal_frame = self.horizontal_limits(frame)

            cv.imshow('Video', frame)

            cv.imshow('Scharr', horizontal_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
    
    def get_histogram(self, frame):
        """
        This function will calculate the histogram of the frame.
        """
        hist = cv.calcHist([frame], [0], None, [256], [0, 256])
        return hist
    
    def horizontal_limits(self, frame):
        """
        This function will calculate the diameter of the fluid track
        using the top and bottom limits of the fluid track.
        """
        if frame.shape[2] == 3:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray_frame = np.copy(frame)
        gray_frame = cv.GaussianBlur(gray_frame, (7, 7), 0, frame)
        # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # frame = clahe.apply(frame)
        # Apply Scharr filter for edge detection
        scharr_x = cv.Scharr(gray_frame, cv.CV_64F, 1, 0)
        scharr_y = cv.Scharr(gray_frame, cv.CV_64F, 0, 1)
        scharr = np.sqrt(scharr_x**2 + scharr_y**2)
        scharr = np.uint8(scharr)
        scharr = cv.GaussianBlur(frame, (7, 7), 0, frame)
        # Threshold the image to get binary image
        # _, threshed = cv.threshold(scharr, 50, 255, cv.THRESH_BINARY)
        # scharr = cv.morphologyEx(scharr, cv.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=5)
        scharr = cv.morphologyEx(scharr, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=5)
        scharr = cv.cvtColor(scharr, cv.COLOR_BGR2GRAY)
        _, threshed = cv.threshold(scharr, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        contours = cv.findContours(threshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        # find the two largest contours 
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:2]
        if len(contours) >= 2:
            # Get bounding rectangles for the two largest contours
            left_rect = cv.boundingRect(contours[0])
            right_rect = cv.boundingRect(contours[1])
            
            # Determine which contour is on the left and which is on the right
            if left_rect[0] > right_rect[0]:
                left_rect, right_rect = right_rect, left_rect
            
            # Calculate crop boundaries
            left_end = left_rect[0] + left_rect[2]  # x + width of left contour
            right_start = right_rect[0]  # x of right contour
            
            # Crop the image between the contours
            if left_end < right_start:
                scharr = scharr[:, left_end:right_start]
        return scharr
        

if __name__ == "__main__":
    filepath = 'videos/nowy2.MP4'  # Replace with your video file path
    measure = Measure(filepath)
    measure.start_video_capture()
    print(f"Movement starts at index: {measure.start}, stops at index: {measure.stop}")