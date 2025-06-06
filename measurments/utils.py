import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt


signal = np.load('nowy3.npy')

def detect_movement(signal: np.ndarray, n_bkps: int = 2) -> tuple:
    """
    Detect change points in the signal using the Kernel Change Point Detection algorithm.
    
    Parameters:
    - signal: np.ndarray - The input signal data.
    - n_bkps: int - The number of breakpoints to detect.
    
    Returns:
    - start: int - Start index of the first detected segment.
    - stop: int - End index of the last detected segment.
    - _ : int - Placeholder for additional information (not used).
    """
    
    algo = rpt.KernelCPD(kernel="linear").fit(signal)
    result = algo.predict(n_bkps=n_bkps)
    
    start, stop, _ = result[0], result[1], result[2]
    
    return start, stop
