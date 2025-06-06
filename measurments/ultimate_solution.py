import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt


signal = np.load('nowy1.npy')
result = rpt.Pelt(model="rbf").fit_predict(signal, pen=10)

rpt.display(signal, result, figsize=(10, 6))
plt.show()