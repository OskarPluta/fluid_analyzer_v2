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
        self.stop = 2634 # stop real 2683 for 3
    def get_movement(self):
        start, stop = detect_movement(self.signal, n_bkps=2)
        print(start, stop)
        return start, stop

    def start_video_capture(self):
        cap = cv.VideoCapture(self.filepath)
        cap.set(cv.CAP_PROP_POS_FRAMES, self.stop)
        while 69:
            ret, frame = cap.read()
            frame = cv.resize(frame, self.size)

           
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            limits = vertical_limits(frame, self.prev_tops, self.prev_bots)
            top_limit, bottom_limit = vertical_limits(frame, self.prev_tops, self.prev_bots, self.size)
            # frame = frame[top_limit:bottom_limit, :]  
            frame = frame[int(np.mean(top_limit)):int(np.mean(bottom_limit))]
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame2 = self.diameter(frame)
            

            if not ret:
                print("End of video or bullshit some error occurred")
                break
            cv.imshow('Video', frame)
            cv.imshow('Diameter', frame2)
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
        scharr_x = cv.Scharr(frame, cv.CV_64F, 0, 1)
        scharr_y = cv.Scharr(frame, cv.CV_64F, 0, 1)
        scharr = np.sqrt(scharr_x**2 + scharr_y**2)
        frame = np.uint8(scharr)
        # cv.GaussianBlur(frame, (7, 7), 0, frame)
        # frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=10)
        frame = cv.morphologyEx(frame, cv.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        frame = cv.medianBlur(frame, 11)
        _, tresh = cv.threshold(frame, 5, 255, cv.THRESH_BINARY)
        frame = tresh

        # frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=10)
        return frame
        




if __name__ == "__main__":
    filepath = 'videos/nowy3.MP4'  # Replace with your video file path
    measure = Measure(filepath).start_video_capture()
    print(f"Movement starts at index: {measure.start}, stops at index: {measure.stop}")