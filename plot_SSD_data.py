import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter

data = pickle.load(file('SSD_data.pkl', 'rb'))

fig, ax = plt.subplots()

x = data[:, 0]
y = data[:, 1]
ax.plot(x, y, alpha=0.1)

yf = savgol_filter(y, 151, 3)
ax.plot(x, yf)

plt.show()
