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
#                x     y   width height
SSD_location = [0.27, 0.35, 0.155, 0.165]
rotation = 6 # degrees
number_of_digits = 4
blur_block_size = 25 # odd
threshold_block_size = 31 # odd
threshold_constant = 3
threshold = 110 # region average pixel value segment detection limit
morph_block_size = 8
delay = 0.5 # delay after each record

## Debugging
debug = 0
debug_more = 1

showplot = 0; record = 0 # for use as module

def find_digits(x):
    """Define SSD digit regions."""
    digits = []; dk = 0.24 * (1. / (number_of_digits * 2))
    digit_percent = 1. / number_of_digits
    digit_width = digit_percent - dk * 2
    for dn in range(0, number_of_digits):
        digit_x_begin = int(x * dk)
        digit_x_end = int(x * (dk + digit_width))
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

    SSDx1 = int(SSD_location[0] * cols)
    SSDy1 = int(SSD_location[1] * rows)
    SSDx2 = int(SSDx1 + SSD_location[2] * cols)
    SSDy2 = int(SSDy1 + SSD_location[3] * rows)
    SSD = rotated[SSDy1:SSDy2, SSDx1:SSDx2]
    if calibration_image:
        cropbox_SSD = rotated.copy()
        cv2.rectangle(cropbox_SSD, (SSDx1, SSDy1), (SSDx2, SSDy2), (255, 0, 0), 3)
        cv2.imshow("Image", cropbox_SSD)
        cv2.waitKey(0)
        cv2.destroyWindow("Image")
        if not debug:
            sys.exit()
    return SSD

def process_image(SSD):
    """Process SSD image."""
    gray = cv2.cvtColor(SSD, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_block_size, blur_block_size), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 
          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
          threshold_block_size, threshold_constant)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,
          np.ones((morph_block_size, morph_block_size), np.uint8))
    return thresh

def find_number(SSD_images, segment_locations):
    """Determine digit value from segments.
       Get list of segment images for debug."""
    thresh = SSD_images[1]
    annotated = SSD_images[2]
    region_view = SSD_images[3]
    interpretation = SSD_images[4]
    umber = ''
    segment_images = []
    for segment_location in segment_locations:
        x1, x2 = segment_location[0][0], segment_location[0][1]
        y1, y2 = segment_location[1][0], segment_location[1][1]
        segment_image = thresh[y1:y2, x1:x2]
        segment_average_pixel_value = segment_image.mean(axis=0).mean()
        if segment_average_pixel_value < threshold:
           region_view[y1:y2, x1:x2] = segment_image # show captured area
           cv2.rectangle(interpretation, (x1, y1),
                  (x2, y2), (0, 0, 0), -1) # draw interpreted segment
           umber += '1'
        else:
           umber += '0'
        segment_images.append(segment_image)
        cv2.rectangle(annotated, (x1, y1),
                     (x2, y2), (255, 0, 0), 2) # draw segment region
    return n(umber), segment_images

def show_plot(SSD_images, reading, minute, plot_images, action):
    """Show a plot of the last minute of values."""
    if action == "setup":
        plt.ion()
        fig = plt.figure()
        fig.canvas.set_window_title('SSD Reader')
        gs = GridSpec(3, 2)
        ax = fig.add_subplot(gs[:, 0])
        line, = ax.plot(minute[:, 0], minute[:, 1], 'k-', lw=2)
        ax.set_xlim(-60, 0)
        ax.set_xlabel("Time (seconds)", weight='bold')
        ax.set_ylabel("Reading", weight='bold')
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis("off")
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis("off")
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.axis("off")
        return line, fig, [ax, ax2, ax3, ax4]
    elif action == "plot":
        SSD = SSD_images[0]
        region_view = SSD_images[3]
        interpretation = SSD_images[4]
        # append new data and pop first record
        now = time.time()
        minute = np.append(minute, np.array([[now, reading]]), axis=0)
        minute = np.delete(minute, 0, axis=0)
        # convert to seconds from now for plotting
        pminute = np.copy(minute)
        for m in range(len(pminute)):
            pminute[m][0] -= now
        minute_time = pminute[:, 0]
        minute_readings = pminute[:, 1]
        # plot
        plot_line.set_xdata(minute_time)
        plot_line.set_ydata(minute_readings)
        plot_axes[0].set_ylim(np.min(minute_readings)
                              - abs(np.min(minute_readings) * 0.1),
                              np.max(minute_readings)
                              + abs(np.max(minute_readings) * 0.1))
        plot_images[0].set_data(SSD)
        plot_images[1].set_data(region_view)
        plot_images[2].set_data(interpretation)
        plot_fig.canvas.draw()
        return minute

def debug_plots(SSD_images, all_segment_images):
    """Debug plots: annotated SSD with regions, interpreted SSD.
       More debug: exploded view of each digit with average
       pixel values of segments."""
    annotated = SSD_images[2]
    interpretation = SSD_images[4]
    cv2.imshow("Annotated", annotated)
    cv2.imshow("Interpretation", interpretation)
    cv2.waitKey(0)
    if debug_more:
        for di, segment_images in enumerate(all_segment_images):
            fig2 = plt.figure()
            fig2.canvas.set_window_title('Digit {}'.format(di))
            def debug_plot_segment(plot_location, segment):
                segment_image = segment_images[segment]
                avg = segment_image.mean(axis=0).mean()
                axp = fig2.add_subplot(5, 3, plot_location)
                axp.set_title('{:.0f}'.format(avg))
                axp.axis('off')
                cmap = 'binary_r'
                if avg == 255: cmap = 'binary'
                axp.imshow(segment_image, cmap=cmap)
            segment_plot_locations = [4, 2, 6, 8, 10, 12, 14]
            for s, spl in enumerate(segment_plot_locations):
                debug_plot_segment(spl, s)
            plt.show()

def read_SSD():
    """Read SSD and record value."""
    global imgshown
    global plot_images
    global data
    global minute
    SSD = get_image()
    thresh = process_image(SSD)
    annotated = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    interpretation = np.full_like(thresh, 255)
    region_view = thresh.copy()
    region_view[thresh == 0] = 200 # to grey out black pixels outside regions
    SSD_images = [SSD, thresh, annotated, region_view, interpretation]

    y, x, o = SSD.shape
    digits = find_digits(x)
    reading = 0

    all_segment_images = []
    for it, digit in enumerate(digits):
        segment_locations = find_segments(digit, y)
        digit_value, segment_images = find_number(SSD_images, segment_locations)
        all_segment_images.append(segment_images)
        reading += digit_value * 10**(number_of_digits - 1 - it)

    if debug:
        debug_plots(SSD_images, all_segment_images)
    elif record:
        data = np.vstack((data, np.array([time.time(), reading])))
        pickle.dump(data, file('SSD_data.pkl', 'wb'))
    if showplot:
        if not imgshown:
            plotimage_SSD = plot_axes[1].imshow(SSD)
            plotimage_rvSSD = plot_axes[2].imshow(region_view, cmap='binary_r')
            plotimage_iSSD = plot_axes[3].imshow(interpretation, cmap='binary_r')
            plot_images = [plotimage_SSD, plotimage_rvSSD, plotimage_iSSD]
            imgshown = True
        minute = show_plot(SSD_images, reading, minute, plot_images, "plot")

    if debug:
        cv2.imshow("SSD", SSD)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit()
    return reading

if __name__ == "__main__":
    ## Log/Plot Option
    showplot = 1 # show the SSD and a plot of the last minute of values
    record = 1 # save the data to file

    if calibration_image or debug:
        showplot = 0
        calibration_image = 1

    if not debug and not calibration_image:
        try:
            data = pickle.load(file('SSD_data.pkl', 'rb'))
        except (IOError, EOFError):
            data = np.empty((0, 2))

    if showplot:
        minute = np.array( # line data for plot initialization
            [[time.time() + i, 0] for i in range(-int(60 / (delay + 1.9)), 0)])
        plot_line, plot_fig, plot_axes = show_plot(0, 0, minute, 0, "setup")
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
        time.sleep(0.5)
        print " Ended."
