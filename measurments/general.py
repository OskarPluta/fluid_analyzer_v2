"""
Works well for videos with white background and black rheometer.
"""
import cv2 as cv
import numpy as np
from collections import deque

def vertical_limits(frame: np.ndarray, prev_yt=None, prev_yb=None):
    """
    Returns the vertical limits of the rheometer in the frame.
    """
    if prev_yt:
        prev_yts.append(prev_yt)
    if len(prev_yts) >= 1:
        avg_yt = int(np.mean(prev_yts))

    if prev_yb:
        prev_ybs.append(prev_yb)
    if len(prev_ybs) >= 1:
        avg_yb = int(np.mean(prev_ybs))

    if frame.shape[2] == 3:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        gray_frame = np.copy(frame)

    blurred_frame = cv.GaussianBlur(gray_frame, (7, 7), 0)
    edges = cv.Canny(blurred_frame, 50, 100, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=50, maxLineGap=10)

    horizontal_lines = []
    
    if lines is None:
        return avg_yt, avg_yb
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 10:
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1 # y1 is always the leftmost point y-coordinate
            horizontal_lines.append(line[0])
    
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

    if prev_yt and abs(yt - avg_yt) > 5:
        yt = avg_yt
    if prev_yb and abs(yb - avg_yb) > 5:
        yb = avg_yb

    return yt, yb


def still_edge(frame: np.ndarray, upper_line, lower_line, prev_x=None,
                right: bool = True):
    """
    Returns the x-coordinates of the non-moving edge of the rheometer in the frame.
    """
    if prev_x:
        prev_xs.append(prev_x)
    if len(prev_xs) >= 1:
        prev_x = int(np.mean(prev_xs))

    if frame.shape[2] == 3:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        gray_frame = np.copy(frame)

    mask = np.zeros_like(gray_frame)
    mask[upper_line:lower_line, :] = 255
    
    blurred_frame = cv.GaussianBlur(gray_frame, (7, 7), 0)
    edges = cv.Canny(blurred_frame, 50, 100, apertureSize=3)
    
    masked_edges = cv.bitwise_and(edges, mask)
    cv.imshow("Masked Edges", masked_edges)
    
    lines = cv.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50,
                            minLineLength=20, maxLineGap=10)
    
    if lines is None or len(lines) == 0:
        return prev_x

    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 10:
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1 # x1 is always the topmost point x-coordinate
            vertical_lines.append(line[0])

    if  not len(vertical_lines):
        return prev_x

    if right:
        x1 = max(vertical_lines, key=lambda x: x[0])[0]
        x2 = max(vertical_lines, key=lambda x: x[2])[2]
    else:
        x1 = min(vertical_lines, key=lambda x: x[0])[0]
        x2 = min(vertical_lines, key=lambda x: x[2])[2]

    if prev_x and abs(x1 - prev_x) > 5 and abs(x2 - prev_x) > 5:
        return prev_x
    
    return int((x1+x2)/2)

def moving_edge(frame, top_limit, bottom_limit, x):
    if frame.shape[2] == 3:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        gray = np.copy(frame)

    
    lines = []
    lines = cv.HoughLinesP(canny, 1, np.pi / 180, 1, minLineLength=20,
                            maxLineGap=30)
    if lines is None:
        return x
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        tan = (x2 - x1) / (y2 - y1)
        if abs(np.degrees(np.arctan(tan))) < 5:
            vertical_lines.append(line)
    if len(vertical_lines) == 0:
        return x
    else:
        canny = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        return min(vertical_lines, key=lambda x: x[0][0])[0][0]

    
video_path = "videos/C0211.MP4"
prev_x = None
prev_xs = deque(maxlen=5)

prev_yt = None
prev_yts = deque(maxlen=10)

prev_yb = None
prev_ybs = deque(maxlen=10)

cap = cv.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (640, 480))
    if not ret:
        break

    top_y, bottom_y = vertical_limits(frame, prev_yt, prev_yb)
    prev_yt = top_y
    prev_yb = bottom_y
    
    x_still = still_edge(frame, top_y, bottom_y, prev_x, right=True)
    prev_x = x_still

    cv.line(frame, (0, top_y), (frame.shape[1], top_y), (0, 255, 0), 2)
    cv.line(frame, (0, bottom_y), (frame.shape[1], bottom_y), (0, 255, 0), 2)
    cv.line(frame, (x_still, top_y), (x_still, bottom_y), (255, 0, 0), 2)

    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break




