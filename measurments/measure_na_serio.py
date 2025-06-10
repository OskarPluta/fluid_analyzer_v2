import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour


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
        self.start, self.stop = self.get_movement()
        # self.start, self.stop = 1708, 1748
        self.left_limit, self.right_limit = self.get_horizontal_limits(self.stop)


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
            cv.imshow('Video', frame)

            cv.imshow('Scharr', horizontal_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

    def fluid_contours(self):
        cap = cv.VideoCapture(self.filepath)
        cap.set(cv.CAP_PROP_POS_FRAMES, self.stop-1)

        ret, frame = cap.read()

        x_ratio = frame.shape[1] / self.size[0]
        y_ratio = frame.shape[0] / self.size[1]

        left_limit = int(self.left_limit * x_ratio)
        right_limit = int(self.right_limit * x_ratio)

        middle_point = (left_limit + right_limit) // 2

        top_limit, bottom_limit = vertical_limits(frame, self.prev_tops, self.prev_bots, self.size)
        top_limit, bottom_limit = int(np.mean(top_limit)*y_ratio), int(np.mean(bottom_limit)*y_ratio)

        roi = frame[top_limit:bottom_limit, middle_point-20:right_limit+20]
        roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        scharr_x = cv.Scharr(roi, cv.CV_64F, 1, 0)
        scharr_y = cv.Scharr(roi, cv.CV_64F, 0, 1)
        scharr = np.sqrt(scharr_x**2 + scharr_y**2)
        scharr = cv.normalize(scharr, None, 0, 255, cv.NORM_MINMAX)
        scharr = np.uint8(scharr)
        # scharr = cv.GaussianBlur(scharr, (5, 5), 0)
        scharr = cv.medianBlur(scharr, 5, 0)

        _, threshed = cv.threshold(scharr, 15, 255, cv.THRESH_BINARY)
        threshed = cv.morphologyEx(threshed, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=3)
        middle_ys = threshed[:, 30]

        indices_255 = np.where(middle_ys == 255)[0]
        
        if len(indices_255) > 0:
            first_index = indices_255[0]
            last_index = indices_255[-1]
        else:
            first_index = None
            last_index = None
        
        cv.line(threshed, (0, first_index), (threshed.shape[1], first_index), 127, 5)
        cv.line(threshed, (0, last_index), (threshed.shape[1], last_index), 127, 5)
        cv.imshow('gowno', threshed)
        


        while 69:
            ret, frame = cap.read()
            if not ret:
                print("End of video or bullshit some error occurred")
                break
            
            if cv.waitKey(20) & 0xFF == ord('q'):
                break
            

            roi = frame[top_limit:bottom_limit, left_limit:right_limit]
            roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            scharr_x = cv.Scharr(roi, cv.CV_64F, 1, 0)
            scharr_y = cv.Scharr(roi, cv.CV_64F, 0, 1)
            scharr = np.sqrt(scharr_x**2 + scharr_y**2)
            scharr = cv.normalize(scharr, None, 0, 255, cv.NORM_MINMAX)
            scharr = np.uint8(scharr)
            # scharr = cv.GaussianBlur(scharr, (5, 5), 0)
            scharr = cv.medianBlur(scharr, 5, 0)

            _, threshed = cv.threshold(scharr, 15, 255, cv.THRESH_BINARY)
            threshed = cv.morphologyEx(threshed, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=3)
            cv.line(threshed, (0, first_index), (threshed.shape[1], first_index), 127, 5)
            cv.line(threshed, (0, last_index), (threshed.shape[1], last_index), 127, 5)
            cv.imshow('Fluid Contours', np.vstack([scharr, threshed]))

if __name__ == "__main__":
    filepath = 'videos/nowy3.MP4'  # Replace with your video file path
    measure = Measure(filepath)
    measure.fluid_contours()
    print(f"Movement starts at index: {measure.start}, stops at index: {measure.stop}")