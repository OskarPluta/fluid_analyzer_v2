"""
Works well for videos with white background and black rheometer.

TODO: Make it so that the functions take deque objects as inputs, so that they
can be used in different files, on different videos.

TODO: Finish the fluid_middle_diameter function, save the diameter of the middle,
based on the distance changes get the point in time when the machine stops moving,
and calculate the diamter ratio starting on from this point in time, save results
to a file and plot them.
"""
import cv2 as cv
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

def vertical_limits(frame: np.ndarray, prev_yts=None, prev_ybs=None):
    """
    Inputs:
    
    frame is the current frame of the video, it can be bgr or grayscale.
    
    prev_yts and prev_ybs are deque objects storing y-coordinates of the previous 
    frame's upper and lower lines of the rheometer, respectively.
    
    Returns:

    The y-coordinates of the upper and lower lines of the rheometer in the
    current frame.
    """

    if len(prev_yts) >= 1:
        avg_yt = int(np.mean(prev_yts))
    else:
        avg_yt = None

    if len(prev_ybs) >= 1:
        avg_yb = int(np.mean(prev_ybs))
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

    prev_ybs.append(yb)
    prev_yts.append(yt)

    yt = int(np.mean(prev_yts))
    yb = int(np.mean(prev_ybs))

    return yt, yb


def still_edge(frame: np.ndarray, upper_line, lower_line, prev_xs: deque,
                right: bool = True):
    """
    Inputs:
    frame is the current frame of the video, it can be bgr or grayscale.

    upper_line and lower_line are the y-coordinates of the upper and lower
    lines of the rheometer, respectively.

    prev_xs is a deque object that stores the x-coordinates of the non-moving
    edge of the rheometer in the previous frames.
    
    right is a boolean that indicates whether the non-moving edge is on the 
    right or the left side of the video. 
    
    Returns:
    The x-coordinate of the non-moving edge of the rheometer in the current frame.
    """

    if len(prev_xs) >= 1:
        avg_x = int(np.mean(prev_xs))
    else:
        avg_x = None

    if frame.shape[2] == 3:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        gray_frame = np.copy(frame)

    mask = np.zeros_like(gray_frame)
    mask[upper_line:lower_line, :] = 255
    
    blurred_frame = cv.GaussianBlur(gray_frame, (7, 7), 0)
    edges = cv.Canny(blurred_frame, 50, 100, apertureSize=3)
    
    masked_edges = cv.bitwise_and(edges, mask)
    
    lines = cv.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50,
                            minLineLength=20, maxLineGap=10)
    
    if lines is None or len(lines) == 0:
        return avg_x

    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 10:
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1 # x1 is always the topmost point x-coordinate
            vertical_lines.append((x1, y1, x2, y2))

    if  len(vertical_lines) == 0:
        return avg_x

    if right:
        x1 = max(vertical_lines, key=lambda x: x[0])[0]
        x2 = max(vertical_lines, key=lambda x: x[2])[2]
    else:
        x1 = min(vertical_lines, key=lambda x: x[0])[0]
        x2 = min(vertical_lines, key=lambda x: x[2])[2]

    if avg_x and abs(x1 - avg_x) > 5 and abs(x2 - avg_x) > 5:
        return avg_x
    
    x = int((x1+x2)/2)
    prev_xs.append(x)
    x = int(np.mean(prev_xs))
    
    return x


def moving_edge(frame, upper_line, lower_line, prev_x, still_edge_x,
                 left: bool = True, distance_cap=100):
    """
    Inputs:

    Returns:
    x is the x-coordinate of the moving edge of the rheometer in the current frame.

    distance is the distance between the moving edge and the non-moving edge.
    """
    
    if frame.shape[2] == 3:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        gray_frame = np.copy(frame)
    
    edges = cv.Canny(gray_frame, 50, 100, apertureSize=3)

    mask = np.zeros_like(gray_frame)
    if prev_x:
        mask[upper_line:lower_line, prev_x-10:prev_x+10] = 255
    elif left:
        mask[upper_line:lower_line, still_edge_x-distance_cap:] = 255

    masked_edges = cv.bitwise_and(edges, mask)

    lines = cv.HoughLinesP(masked_edges, 1, np.pi / 180, 1, minLineLength=20,
                            maxLineGap=30)
    if lines is None or len(lines) == 0:
        return prev_x

    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 10:
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1 # x1 is always the topmost point x-coordinate
            vertical_lines.append((x1, y1, x2, y2))

    if  len(vertical_lines) == 0:
        return prev_x

    if left:
        x1 = min(vertical_lines, key=lambda x: x[0])[0]
        x2 = min(vertical_lines, key=lambda x: x[2])[2]
    else:
        x1 = max(vertical_lines, key=lambda x: x[0])[0]
        x2 = max(vertical_lines, key=lambda x: x[2])[2]

    if prev_x and abs(x1 - prev_x) > 5 and abs(x2 - prev_x) > 5:
        return prev_x
    
    x = int((x1+x2)/2)

    distance = abs(x - still_edge_x)

    return x, distance
    


