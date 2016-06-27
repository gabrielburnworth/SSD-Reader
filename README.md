# SSD-Reader
Read a seven-segment display (SSD) using OpenCV and the Raspberry Pi Camera

__Install OpenCV and required modules on Raspbian:__
```
sudo apt-get install python-opencv python-numpy python-picamera python-matplotlib python-scipy
```

__Run the script:__

Run the script: `python SSD_Reader.py`

__Alternatively, use a python command line:__

After running `python SSD_Reader.py` with `calibration_image = True` 
and modifying calibration values to locate SSD:

```python
from SSD_Reader import read_SSD
read_SSD()
```

__Screenshot:__

![ssd_reader_plot](https://cloud.githubusercontent.com/assets/12681652/16398696/36f367d2-3c81-11e6-9c82-131abdbfd68e.png)

__Method:__

1. Take a photo with the Raspberry Pi Camera
2. Crop image to the seven-segment display (SSD)
3. Process and filter image to binary (black and white)
3. Calculate average pixel value over segment regions
4. Display segment regions whose average pixel values exceed a threshold
5. Convert to a number and return the value
6. (Optional) Plot the value over time along with the SSD and processing images
7. (Optional) Save the value with timestamp to file for later manipulation
