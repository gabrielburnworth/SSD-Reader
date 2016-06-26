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