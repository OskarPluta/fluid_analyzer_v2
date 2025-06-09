import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from utils import detect_movement, vertical_limits
from alternative import Preprocess


class Measure:
    def __init__(self, filepath: str):
        self.filepath = filepath
        # self.preprocess = Preprocess()
        # self.signal = self.preprocess.preprocess_video(filepath)
        # print(f"Signal shape: {self.signal.shape}")
        # self.start, self.stop = self.get_movement()
        # self.start = 1453 
        # self.stop = 1492
        self.prev_bots = deque(maxlen=5)  # previous bottom line positions
        self.prev_tops = deque(maxlen=5)  # previous top line positions
        
        self.size = (640, 480)  
        self.start = 1453
        self.stop = 1493
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
            cv.imshow('Video', frame)
            cv.imshow('Diameter', frame2)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
    
    def diameter(self, frame):
        """
        This function will calculate the diameter of the fluid track
        using the top and bottom limits of the fluid track.
        """
        frame = np.copy(frame)
        cv.GaussianBlur(frame, (7, 7), 0, frame)
        # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # frame = clahe.apply(frame)
        # Apply Scharr filter for edge detection
        scharr_x = cv.Scharr(frame, cv.CV_64F, 1, 0)
        scharr_y = cv.Scharr(frame, cv.CV_64F, 0, 1)
        scharr = np.sqrt(scharr_x**2 + scharr_y**2)
        frame = np.uint8(scharr)
        cv.GaussianBlur(frame, (7, 7), 0, frame)

        # Threshold the image to get binary image
        _, frame = cv.threshold(frame, 50, 255, cv.THRESH_BINARY)
        closing = cv.morphologyEx(frame, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=5)
        cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Find contours and draw them
        contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                # Draw contours on the frame
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        cv.drawContours(frame, contours, -1, (0, 255, 0), 2)

        return frame
        




if __name__ == "__main__":
    filepath = 'videos/C0697.MP4'  # Replace with your video file path
    measure = Measure(filepath)
    measure.start_video_capture()
    print(f"Movement starts at index: {measure.start}, stops at index: {measure.stop}")