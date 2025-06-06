import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from utils import detect_movement
from alternative import Preprocess

filepath = "videos/nowy2.MP4"
signal = Preprocess().preprocess_video(filepath)
start, stop = detect_movement(signal, n_bkps=2)