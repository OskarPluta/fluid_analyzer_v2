import cv2 as cv
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class FluidDiameterMeasurement:
    def __init__(self, downsize:tuple = (640, 480)):

        self.downsize = downsize # resolution to find the edges of the fluid

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

        self.initial_diameter = None # initial diameter of the fluid in the rheometer
        # after it stops moving

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
    
    def still_edge(self, frame: np.ndarray, right: bool = True):
        """
        
        """

        if len(self.prev_still_xs) >= 1:
            avg_x = int(np.mean(self.prev_still_xs))
        else:
            avg_x = None

        if frame.shape[2] == 3:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray_frame = np.copy(frame)

        mask = np.zeros_like(gray_frame)
        mask[self.upper_line:self.lower_line, :] = 255
        
        blurred_frame = cv.GaussianBlur(gray_frame, (7, 7), 0)
        edges = cv.Canny(blurred_frame, 50, 100, apertureSize=3)
        
        masked_edges = cv.bitwise_and(edges, mask)
        
        lines = cv.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50,
                                minLineLength=20, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            self.still_x = avg_x
            self.prev_still_xs.append(self.still_x)
            return


        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 10:
                if y1 > y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1 # x1 is always the topmost point x-coordinate
                vertical_lines.append((x1, y1, x2, y2))

        if  len(vertical_lines) == 0:
            self.still_x = avg_x
            self.prev_still_xs.append(self.still_x)
            return

        if right:
            x1 = max(vertical_lines, key=lambda x: x[0])[0]
            x2 = max(vertical_lines, key=lambda x: x[2])[2]
        else:
            x1 = min(vertical_lines, key=lambda x: x[0])[0]
            x2 = min(vertical_lines, key=lambda x: x[2])[2]

        if avg_x and abs(x1 - avg_x) > 5 and abs(x2 - avg_x) > 5:
            self.still_x = avg_x
            self.prev_still_xs.append(self.still_x)
            return
        
        x = int((x1+x2)/2)
        self.prev_still_xs.append(x)
        x = int(np.mean(self.prev_still_xs))
        
        self.still_x = x

    def moving_edge(self, frame,  left: bool = True, distance_cap=100):
        """
      
        """
        
        if frame.shape[2] == 3:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray_frame = np.copy(frame)
        
        edges = cv.Canny(gray_frame, 50, 100, apertureSize=3)

        mask = np.zeros_like(gray_frame)
        if self.prev_moving_x:
            mask[self.upper_line:self.lower_line, self.prev_moving_x-10:self.prev_moving_x+10] = 255
        elif left:
            mask[self.upper_line:self.lower_line, self.still_x-distance_cap:] = 255

        masked_edges = cv.bitwise_and(edges, mask)

        lines = cv.HoughLinesP(masked_edges, 1, np.pi / 180, 1, minLineLength=20,
                                maxLineGap=30)
        
        if lines is None or len(lines) == 0:
            self.moving_x = self.prev_moving_x
            distance = abs(self.moving_x - self.still_x)
            self.distances_deque.append(distance)
            self.distances.append(distance)
            self.moving_x = self.prev_moving_x
            return

        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 10:
                if y1 > y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1 # x1 is always the topmost point x-coordinate
                vertical_lines.append((x1, y1, x2, y2))

        if  len(vertical_lines) == 0:
            self.moving_x = self.prev_moving_x
            distance = abs(self.moving_x - self.still_x)
            self.distances_deque.append(distance)
            self.distances.append(distance)
            self.moving_x = self.prev_moving_x
            return

        if left:
            x1 = min(vertical_lines, key=lambda x: x[0])[0]
            x2 = min(vertical_lines, key=lambda x: x[2])[2]
        else:
            x1 = max(vertical_lines, key=lambda x: x[0])[0]
            x2 = max(vertical_lines, key=lambda x: x[2])[2]

        if self.prev_moving_x and abs(x1 - self.prev_moving_x) > 5 and abs(x2 - self.prev_moving_x) > 5:
            self.moving_x = self.prev_moving_x
            return
        
        x = int((x1+x2)/2)

        distance = abs(x - self.still_x)

        self.distances_deque.append(distance)
        self.distances.append(distance)
        self.moving_x = x
        self.prev_moving_x = x

    def fluid_middle_diameter(self, frame):
        """
        Measures the diameter of the fluid in the rheometer.
        """
        
        if frame.shape[2] == 3:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray_frame = np.copy(frame)

        mask = np.zeros_like(gray_frame)

        x_ratio = gray_frame.shape[1] / self.downsize[0]
        y_ratio = gray_frame.shape[0] / self.downsize[1]

        mask_upper_line = int(self.upper_line * y_ratio)
        mask_lower_line = int(self.lower_line * y_ratio)

        mask_moving_x = int(self.moving_x * x_ratio)
        mask_still_x = int(self.still_x * x_ratio)

        mask[mask_upper_line:mask_lower_line, mask_moving_x:mask_still_x] = 255

        blurred_frame = cv.GaussianBlur(gray_frame, (3, 3), 0)

        edges = cv.Canny(blurred_frame, 20, 100, apertureSize=3)
        masked_edges = cv.bitwise_and(edges, mask)
        cv.imshow("Masked Edges", masked_edges)

        x_middle = (mask_moving_x + mask_still_x) // 2 # x-coordinate of the middle of the fluid
        mid_values = masked_edges[mask_upper_line:mask_lower_line, x_middle] # values of a slice of the middle of the fluid

        yd_bottom = None # y-coordinate of the bottom of the middle diameter
        yd_top = None # y-coordinate of the top of the middle diameter

        for i, val in enumerate(mid_values):
            if val == 255:
                yd_bottom = i + mask_upper_line
        for i, val in enumerate(mid_values[::-1]):
            if val == 255 :
                yd_top = mask_lower_line - i

        if yd_bottom and yd_top:
            cv.line(frame, (x_middle, yd_top), (x_middle, yd_bottom), (255, 0, 255), 2)
            diameter = yd_bottom - yd_top # diameter of the fluid in the rheometer
            self.diameters.append(diameter)
        else:
            self.diameters.append(-1) # if the diameter is not found, append -1

    def is_moving(self):
        """
        Returns True if the machine is moving, False otherwise.
        """
        if not self.movement_start:
            if self.distances_deque[-1] > 40:
                self.movement_start = True
                self.movement_start_index = self.frame_index
        elif self.movement_start and not self.movement_end:
            if  max(self.distances_deque) - min(self.distances_deque) < 10:
                self.movement_end = True
                self.movement_end_index = self.frame_index
                self.initial_diameter = self.diameters[-1]
                print('now')

    def measure(self, frame: np.ndarray):
        """
        """
        self.frame_index += 1
        frame_resized = cv.resize(frame, self.downsize)

        self.vertical_limits(frame_resized)
        self.still_edge(frame_resized, right=True)
        self.moving_edge(frame_resized, left=True)
        self.is_moving()



        self.fluid_middle_diameter(frame)
        
        
video_path = "videos/C0209.MP4"
cap = cv.VideoCapture(video_path)

diameter_measurement = FluidDiameterMeasurement()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    diameter_measurement.measure(frame)
    # cv.line(frame, (0, diameter_measurement.upper_line), (frame.shape[1], diameter_measurement.upper_line), (0, 255, 0), 2)
    # cv.line(frame, (0, diameter_measurement.lower_line), (frame.shape[1], diameter_measurement.lower_line), (0, 255, 0), 2)
    # cv.line(frame, (diameter_measurement.still_x, diameter_measurement.upper_line), (diameter_measurement.still_x, diameter_measurement.lower_line), (255, 0, 0), 2)
    # cv.line(frame, (diameter_measurement.moving_x, diameter_measurement.upper_line), (diameter_measurement.moving_x, diameter_measurement.lower_line), (0, 0, 255), 2)

    cv.imshow("Vertical Limits", frame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
fps = cap.get(cv.CAP_PROP_FPS)


plot_diameters = diameter_measurement.diameters[diameter_measurement.movement_end_index:]
plot_diameters = [d/diameter_measurement.initial_diameter for d in plot_diameters if d != -1] # remove -1 from the list
time = [1/fps * i for i in range(len(plot_diameters))]

plt.scatter(time, plot_diameters)
plt.show()