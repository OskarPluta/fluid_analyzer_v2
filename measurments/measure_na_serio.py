import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from utils import detect_movement, vertical_limits, horizontal_limits
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
<<<<<<< HEAD
        # self.start, self.stop = 1312, 1368
=======
        # self.start, self.stop = 2031, 2082
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


>>>>>>> d2ea042bdb77730ec942ff7e936502f5649cfa38

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
<<<<<<< HEAD

            left_limit, right_limit = self.horizontal_limits(frame)

            # Draw the horizontal limits on the frame
            cv.line(frame, (left_limit, 0), (left_limit, frame.shape[0]), (255, 0, 0), 2)
            cv.line(frame, (right_limit, 0), (right_limit, frame.shape[0]), (255, 0, 0), 2)
            cv.imshow('Video', frame)


=======
            # limits = vertical_limits(frame, self.prev_tops, self.prev_bots)
            top_limit, bottom_limit = vertical_limits(frame, self.prev_tops, self.prev_bots, self.size)
            # frame = frame[top_limit:bottom_limit, :]  
            frame = frame[int(np.mean(top_limit)):int(np.mean(bottom_limit))]
            horizontal_frame = frame[:, self.left_limit:self.right_limit]

            cv.imshow('Video', frame)

            cv.imshow('Scharr', horizontal_frame)
>>>>>>> d2ea042bdb77730ec942ff7e936502f5649cfa38
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
    
<<<<<<< HEAD
    def horizontal_limits(self, frame):
=======
    def get_histogram(self, frame):
>>>>>>> d2ea042bdb77730ec942ff7e936502f5649cfa38
        """
        This function will calculate the histogram of the frame.
        """
<<<<<<< HEAD
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
        threshed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=5)
       
        contours, _ = cv.findContours(threshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:2]

        
        x_limits = []

        for contour in contours:
            x_values = contour[:, 0, 0]
            x_limits.append(np.min(x_values))
            x_limits.append(np.max(x_values))

        x_limits = sorted(x_limits)

        return x_limits[1], x_limits[2]
        

if __name__ == "__main__":
    filepath = 'videos/nowy2.MP4'  # Replace with your video file path
=======
        hist = cv.calcHist([frame], [0], None, [256], [0, 256])
        return hist
    

if __name__ == "__main__":
    filepath = 'videos/nowy3.MP4'  # Replace with your video file path
>>>>>>> d2ea042bdb77730ec942ff7e936502f5649cfa38
    measure = Measure(filepath)
    measure.start_video_capture()
    print(f"Movement starts at index: {measure.start}, stops at index: {measure.stop}")