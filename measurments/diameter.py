import cv2 as cv
import numpy as np
from collections import deque
from general import vertical_limits, still_edge, moving_edge


def fluid_middle_diameter(frame, x_still, x_moving, upper_line, lower_line):
    """
    Inputs:
    frame is the current frame of the video, it can be bgr or grayscale.

    x_still and x_moving are the x-coordinates of the non-moving and moving
    edges of the rheometer, respectively.

    upper_line and lower_line are the y-coordinates of the upper and lower
    lines of the rheometer, respectively.

    Returns:
    The diameter of the fluid in the rheometer in the current frame.
    """
    
    if frame.shape[2] == 3:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        gray_frame = np.copy(frame)

    mask = np.zeros_like(gray_frame)
    mask[upper_line:lower_line, x_moving:x_still] = 255

    blurred_frame = cv.GaussianBlur(gray_frame, (3, 3), 0)

    edges = cv.Canny(blurred_frame, 20, 100, apertureSize=3)
    masked_edges = cv.bitwise_and(edges, mask)

    xd_middle = (x_moving + x_still) // 2 # x-coordinate of the middle of the fluid
    mid_values = masked_edges[upper_line:lower_line, xd_middle] # values of a slice of the middle of the fluid

    yd_bottom = None
    yd_top = None

    for i, val in enumerate(mid_values):
        if val == 255:
            yd_bottom = i + upper_line
    for i, val in enumerate(mid_values[::-1]):
        if val == 255 :
            yd_top = lower_line - i

    if yd_bottom and yd_top:
        cv.line(frame, (xd_middle, yd_top), (xd_middle, yd_bottom),
                (0, 255, 0), 2)
        

video_path = "videos/C0210.MP4"

prev_still_x = None
prev_still_xs = deque(maxlen=5)

prev_moving_x = None

oprev_yts = deque(maxlen=10)

oprev_ybs = deque(maxlen=10)

cap = cv.VideoCapture(video_path)

distances = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.resize(frame, (640, 480))

    top_y, bottom_y = vertical_limits(frame, oprev_yts, oprev_ybs)
    
    x_still = still_edge(frame, top_y, bottom_y, prev_still_xs, right=True)
    prev_x = x_still

    x_moving = moving_edge(frame, top_y, bottom_y, prev_moving_x, x_still,
                             left=True)
    prev_moving_x = x_moving

    fluid_middle_diameter(frame, x_still, x_moving, top_y, bottom_y)

    cv.line(frame, (0, top_y), (frame.shape[1], top_y), (0, 255, 0), 2)
    cv.line(frame, (0, bottom_y), (frame.shape[1], bottom_y), (0, 255, 0), 2)
    cv.line(frame, (x_still, top_y), (x_still, bottom_y), (255, 0, 0), 2)
    cv.line(frame, (x_moving, top_y), (x_moving, bottom_y), (255, 0, 0), 2)

    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()