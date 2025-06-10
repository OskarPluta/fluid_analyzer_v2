import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from utils import detect_movement, vertical_limits, horizontal_limits
from alternative import Preprocess


class Measure:
    def __init__(self, filepath: str):
        self.filepath = filepath
        # self.preprocess = Preprocess()
        # self.signal = self.preprocess.preprocess_video(filepath)

        self.prev_bots = deque(maxlen=5)  # previous bottom line positions
        self.prev_tops = deque(maxlen=5)  # previous top line positions
        
        self.size = (640, 480)  
        # self.start = 1453
        # self.stop = 1492 # stop real 2683 for 3
        # self.start, self.stop = self.get_movement()
        self.start, self.stop = 2634, 2683  
        self.left_limit, self.right_limit = self.get_horizontal_limits(self.stop)
        print(self.left_limit, self.right_limit)
    

    def get_horizontal_limits(self, frame_index: int, number_of_frames: int = 100):
        cap = cv.VideoCapture(self.filepath)
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
        
        horizontal_limits_list = []
        
        for i in range(number_of_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {i}")
                break
            frame = cv.resize(frame, self.size)
            top_limit, bottom_limit = vertical_limits(frame, self.prev_tops, self.prev_bots, self.size)
            horizontal_limits_ = horizontal_limits(frame[int(np.mean(top_limit)):int(np.mean(bottom_limit))])
            horizontal_limits_list.append(horizontal_limits_)
        
        cap.release()
        
        if horizontal_limits_list:
            return np.mean(horizontal_limits_list, axis=0).astype(int)
        else:
            return None

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
            horizontal_frame = frame[:, self.left_limit:self.right_limit]
            horizontal_frame = self.measure_diameter(horizontal_frame)
            cv.imshow('Video', frame)

            cv.imshow('Scharr', horizontal_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
        
    
    def measure_diameter(self, frame):
        """
        This function will measure the diameter of the fluid track in the frame..
        """
        
        # Apply Gaussian blur to reduce noise
        blurred = cv.GaussianBlur(frame, (5, 5), 0)
       
        # Return the edge-detected frame
        frame = edges
        
        return frame

if __name__ == "__main__":
    filepath = 'videos/C0209.MP4'  # Replace with your video file path
    measure = Measure(filepath)
    measure.start_video_capture()
    print(f"Movement starts at index: {measure.start}, stops at index: {measure.stop}")