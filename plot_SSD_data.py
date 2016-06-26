import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pickle.load(file('SSD_data.pkl', 'rb'))

fig, ax = plt.subplots()

x = data[:, 0]
y = data[:, 1]

ax.plot(x, y, alpha=0.1)

plt.show()