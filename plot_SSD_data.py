import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import pickle
from scipy.signal import savgol_filter
import time
import datetime as dt

data = pickle.load(file('SSD_data.pkl', 'rb'))

x = data[:, 0]
def tse2dt(ts_epoch):
    try:
        tdt = [dt.datetime.fromtimestamp(int(tse)) for tse in ts_epoch]
    except TypeError:
        tdt = dt.datetime.fromtimestamp(int(ts_epoch))
    return dates.date2num(tdt)
xdt = tse2dt(x)

y = data[:, 1]
yf = savgol_filter(y, 151, 3)

now = time.time()
periods = [7, 24, 3600]
labels = ["Week", "Day", "Hour"]
locators = [dates.DayLocator(), dates.HourLocator(interval=2), dates.MinuteLocator(interval=5)]
fmts = [dates.DateFormatter('%m/%d'), dates.DateFormatter('%H:%M'), dates.DateFormatter('%H:%M')]

fig, axs = plt.subplots(3, sharex=False, figsize=(8, 14))

for tp, (ax, label, locator, fmt) in enumerate(zip(axs, labels, locators, fmts)):
    ax.plot(xdt, y, alpha=0.1)
    ax.plot(xdt, yf)
    ax.set_xlabel(label, weight='bold')
    ax.set_xlim(tse2dt(now - np.prod(periods[tp:])), tse2dt(now))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)
    ticklabels = ax.get_xticklabels()
    plt.setp(ticklabels, rotation=30)

for ax in axs:
    ax.set_ylabel("Measurement", weight='bold')

plt.tight_layout()
plt.show()