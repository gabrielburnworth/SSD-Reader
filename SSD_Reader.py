"""Seven-segment Display Reader.

Reads a seven-segment display,
    plots the last minute of values,
    saves all values to file.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from picamera.array import PiRGBArray
from picamera import PiCamera
import sys, time, pickle

## Calibration
calibration_image = 0 # 1 to aid in calibration
# SSD location (from top left of entire image)
#  as percent (0 - 1) of image:
#                x     y   height width
SSD_location = [0.23, 0.26, 0.17, 0.19]
rotation = 1 # degrees
number_of_digits = 4
blur_value = 25
thresh_value = 27
threshold = 200 # region average pixel value segment detection limit
delay = 0.5 # delay after each record

## Debugging
debug = 0
debug_more = 1

showplot = 0; record = 0 # for use as module

def find_digits(x):
    """Define SSD digit regions."""
    digits = []; dk0 = 0.24 * (1. / (number_of_digits * 2))
    dk = dk0
    digit_percent = 1. / number_of_digits
    digit_width = digit_percent - dk * 2
    for dn in range(0, number_of_digits):
        edgeL, edgeR = 0.00, 0.00
        if dn == 0:
            edgeL = dk0
        if dn == (number_of_digits - 1):
            edgeR = dk0
        digit_x_begin = int(x * (dk - edgeL))
        digit_x_end = int(x * (dk + digit_width + edgeR))
        digits.append([digit_x_begin, digit_x_end])
        dk += digit_percent
    return digits

"""
Seven-segment display segment regions:

c1  c2        c3    c4
       -------        r1
      |   1   |      
 ---   -------   ---  r2
|   |           |   |
| 0 |           | 2 |
|   |           |   |
 ---   -------   ---  r3
      |   3   |      
 ---   -------   ---  r4
|   |           |   |
| 4 |           | 5 |
|   |           |   |
 ---   -------   ---  r5
      |   6   |      
       -------        r6
"""

def n(segment_values):
    """Return the displayed value of the SSD digit."""
    segment_values = ''.join([str(v) for v in segment_values])
    return {
       '1110111': 0,
       '0010010': 1,
       '0111101': 2,
       '0111011': 3,
       '1011010': 4,
       '1101011': 5,
       '1101111': 6,
       '0110010': 7,
       '1111111': 8,
       '1111011': 9,
    }.get(segment_values, 0)

def find_segments(digit, y):
    """Define SSD digit segment regions."""
    x1, x2 = digit[0], digit[1]
    #cv2.rectangle(annotated, (x1, 0), (x2, y), (255, 0, 0), 2)
    
    dx = x2 - x1
    c1 = x1
    c2 = int(x1 + dx * 0.2)
    c3 = int(x1 + dx * 0.8)
    c4 = x2
    r1 = 0
    r2 = int(y * 0.1)
    r3 = int(y * 0.44)
    r4 = int(y * 0.56)
    r5 = int(y * 0.9)
    r6 = y
    segment_locations = [
             [[c1, c2], [r2, r3]], # 0
             [[c2, c3], [r1, r2]], # 1
             [[c3, c4], [r2, r3]], # 2
             [[c2, c3], [r3, r4]], # 3
             [[c1, c2], [r4, r5]], # 4
             [[c3, c4], [r4, r5]], # 5
             [[c2, c3], [r5, r6]], # 6
            ]
    return segment_locations

def get_image():
    """Get image and crop it to SSD."""

    with PiCamera() as camera:
        camera.resolution = (1920, 1088)
        rawCapture = PiRGBArray(camera)
        time.sleep(0.1)
        camera.capture(rawCapture, format="bgr")
        image = rawCapture.array
    
    rows, cols, o = image.shape
    Mtrx = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
    rotated = cv2.warpAffine(image, Mtrx, (cols, rows))
    
    cx = int(SSD_location[0] * cols)
    cy = int(SSD_location[1] * rows)
    cxe = int(cx + SSD_location[2] * cols)
    cye = int(cy + SSD_location[3] * rows)
    SSD = rotated[cy:cye, cx:cxe]
    if calibration_image:
        cropbox_SSD = rotated.copy()
        cv2.rectangle(cropbox_SSD, (cx, cy), (cxe, cye), (255, 0, 0), 3)
        cv2.imshow("Image", cropbox_SSD)
        cv2.waitKey(0)
        cv2.destroyWindow("Image")
        if not debug:
            sys.exit()
    return SSD

def process_image(SSD):
    """Process SSD image."""
    gray = cv2.cvtColor(SSD, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 
          cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thresh_value, 2)
    return thresh

def find_number(SSD_images, segment_locations):
    """Determine digit value from segments.
       Get list of segment images for debug."""
    thresh = SSD_images[1]
    annotated = SSD_images[2]
    interpretation = SSD_images[3]
    umber = [0] * len(segment_locations)
    segment_images = []
    for s, segment_location in enumerate(segment_locations):
        x1_1, x2_1 = segment_location[0][0], segment_location[0][1]
        y1_1, y2_1 = segment_location[1][0], segment_location[1][1]
        segment_image = thresh[y1_1:y2_1, x1_1:x2_1]
        if segment_image.mean(axis=0).mean() < threshold:
           cv2.rectangle(interpretation, (x1_1, y1_1), 
                  (x2_1, y2_1), (0, 0, 0), -1) # draw interpreted segment
           umber[s] = 1
        segment_images.append(segment_image)
        cv2.rectangle(annotated, (x1_1, y1_1), 
                     (x2_1, y2_1), (255, 0, 0), 2) # draw segment region
    return n(umber), segment_images

def show_plot(SSD_images, reading, plot_images, action):
    """Show a plot of the last minute of values."""
    if action == "setup":
        plt.ion()
        fig = plt.figure()
        gs = GridSpec(2, 2)
        ax = fig.add_subplot(gs[:, 0])
        npminute = np.array(minute)
        line, = ax.plot(npminute[:, 0], npminute[:, 1], 'k-', lw=2)
        ax.set_ylim(0, 10000)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis("off")
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis("off")
        return line, fig, [ax, ax2, ax3]
    elif action == "plot":
        SSD = SSD_images[0]
        interpretation = SSD_images[3]
        minute.append([time.time(), reading])
        minute.pop(0)
        npminute = np.array(minute)
        plot_line.set_xdata(npminute[:, 0])
        plot_line.set_ydata(npminute[:, 1])
        plot_axes[0].set_xlim(time.time() - 60, time.time())
        plot_images[0].set_data(SSD)
        plot_images[1].set_data(interpretation)
        plot_fig.canvas.draw()

def debug_plots(segment_images):
    """Debug plots."""
    cv2.imshow("Interpretation", interpretation)
    cv2.imshow("Annotated", annotated)
    cv2.waitKey(0)
    if debug_more:
        for segment_images in all_segment_images:
            fig2 = plt.figure()
            def debug_plot_segment(plot_location, segment):
                segment_image = segment_images[segment]
                avg = segment_image.mean(axis=0).mean()
                axp = fig2.add_subplot(5, 3, plot_location)
                axp.set_title('{:.2f}'.format(avg))
                axp.axis('off')
                cmap = 'binary_r'
                if avg == 255: cmap = 'binary'
                axp.imshow(segment_image, cmap=cmap)
            debug_plot_segment(4, 0)
            debug_plot_segment(2, 1)
            debug_plot_segment(6, 2)
            debug_plot_segment(8, 3)
            debug_plot_segment(10, 4)
            debug_plot_segment(12, 5)
            debug_plot_segment(14, 6)
            plt.show()

def read_SSD():
    """Read SSD and record value."""
    global imgshown
    global plot_images
    global data
    SSD = get_image()
    thresh = process_image(SSD)
    annotated = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    interpretation = np.full_like(thresh, 255)
    SSD_images = [SSD, thresh, annotated, interpretation]

    y, x, o = SSD.shape
    digits = find_digits(x)
    number = [0] * number_of_digits
    
    all_segment_images = []
    for it, digit in enumerate(digits):
        segment_locations = find_segments(digit, y)
        number[it], segment_images = find_number(SSD_images, segment_locations)
        all_segment_images.append(segment_images)

    reading = int(''.join([str(val) for val in number]))

    if debug:
        debug_plots(segment_images)
    elif record:
        data = np.vstack((data, np.array([time.time(), reading])))
        pickle.dump(data, file('SSD_data.pkl', 'wb'))
    if showplot:
        if not imgshown:
            plotimage_SSD = plot_axes[1].imshow(SSD)
            plotimage_iSSD = plot_axes[2].imshow(interpretation, cmap='binary_r')
            plot_images = [plotimage_SSD, plotimage_iSSD]
            imgshown = True
        show_plot(SSD_images, reading, plot_images, "plot")

    if debug:
        cv2.imshow("SSD", SSD)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit()
    return reading

if __name__ == "__main__":
    ## Log/Plot Option
    showplot = 1 # show the SSD and a plot of the last minute of values
    record = 1

    if calibration_image or debug:
        showplot = 0
        calibration_image = 1

    if not debug and not calibration_image:
        try:
            data = pickle.load(file('SSD_data.pkl', 'rb'))
        except (IOError, EOFError):
            data = np.empty((0, 2))

    if showplot:
        minute = [[time.time() - i, 0] for i in range(0, 60)]
        plot_line, plot_fig, plot_axes = show_plot(0, 0, 0, "setup")
        imgshown = False
        plot_images = []
    else:
        imgshown = True

    ## Loop
    try:
        while 1:
            reading = read_SSD()
            print reading
            time.sleep(delay)

    except KeyboardInterrupt:
        sys.exit()
