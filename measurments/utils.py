import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import ruptures as rpt


def detect_movement(signal: np.ndarray, n_bkps: int = 2) -> tuple:
    """
    Detect change points in the signal using the Kernel Change Point Detection algorithm.
    
    Parameters:
    - signal: np.ndarray - The input signal data.
    - n_bkps: int - The number of breakpoints to detect.
    
    Returns:
    - start: int - Start index of the first detected segment.
    - stop: int - End index of the last detected segment.
    """
    
    algo = rpt.KernelCPD(kernel="linear").fit(signal)
    result = algo.predict(n_bkps=n_bkps)
    
    start, stop, _ = result[0], result[1], result[2]
    
    return start, stop


def vertical_limits(frame: np.ndarray, prev_tops: deque,
                    prev_bots: deque, size: tuple = (640, 480)):

    """
    In order for this super function to work it needs two empty deques on first
    call.
"""

    frame = cv.resize(frame, size)

    if frame.shape[2] == 3:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        gray_frame = np.copy(frame)

    blurred_frame = cv.GaussianBlur(gray_frame, (7, 7), 0)

    canny_edges = cv.Canny(blurred_frame, 50, 100, apertureSize=3)

    lines = cv.HoughLinesP(canny_edges, 1, np.pi / 180, threshold=50,
                            minLineLength=50, maxLineGap=10)
    
    horizontal_lines = []

    if lines.any():
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if np.abs(y2-y1) < frame.shape[1]*0.05:  # Check if the line is horizontal
                horizontal_lines.append((y1, y2))


    top_line = min(horizontal_lines, key=lambda x: x[0]+x[1])
    bottom_line = max(horizontal_lines, key=lambda x: x[0]+x[1])

    if len(prev_tops) > 0 and len(prev_bots) > 0:
        avg_tops = int(np.mean(prev_tops))
        avg_bots = int(np.mean(prev_bots))

        if abs(avg_tops - top_line[1]) > frame.shape[0] * 0.05:
            top_line = (avg_tops, avg_tops)

        if abs(avg_bots - bottom_line[1]) > frame.shape[0] * 0.05:
            bottom_line = (avg_bots, avg_bots)
            print("ale chujnia")

    prev_tops.append((top_line[0] + top_line[1]) // 2)
    prev_bots.append((bottom_line[0] + bottom_line[1]) // 2)

    return top_line, bottom_line
