import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from utils import vertical_limits

class Preprocess:

    def __init__(self, size:tuple = (640, 480)):

        self.size = size # resolution to find the edges of the fluid

        self.prev_bots = deque(maxlen=5)  # previous bottom line positions
        self.prev_tops = deque(maxlen=5)  # previous top line positions

        """Bellow are some bullshit variables that will get deleted probably."""

        
    
    def preprocess_video(self, filepath: str):
        white_pixels_list = []
        cap = cv.VideoCapture(filepath)

        while 69:
            ret, frame = cap.read()
            if not ret:
                print("End of video or some bullshit error")
                break
            frame = cv.resize(frame, self.size)
            top_limit, bottom_limit = vertical_limits(frame, self.prev_tops, self.prev_bots, self.size)
            # frame = frame[top_limit:bottom_limit, :]  
            white_pixels = self.thresh_roi(int(np.mean(top_limit)),
                                            int(np.mean(bottom_limit)), frame)
            white_pixels_list.append(white_pixels)
            # cv.imshow('Video', frame) 
            # cv.waitKey(1) & 0xFF
            # if cv.waitKey(1) & 0xFF == ord('q'):
                # break
        # cv.destroyAllWindows()       
        # cap.release()
        return np.array(white_pixels_list)
        

    def thresh_roi(self, upper_line, lower_line, frame: np.ndarray):
        """
        Threshold the region of interest (ROI) in the frame to isolate the fluid
        track expansion.

        Returns the number of white pixels in the thresholded ROI.
        """
        if frame.shape[2] == 3:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray_frame = np.copy(frame)


        roi = gray_frame[upper_line:lower_line, :]

        blurred_roi = cv.GaussianBlur(roi, (7, 7), 0)
        _, threshed_roi = cv.threshold(blurred_roi, 120, 255, cv.THRESH_BINARY)
        
        white_pixels = np.sum(threshed_roi == 255)

        return white_pixels