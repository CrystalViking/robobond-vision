# robobond-vision
RoboBond-vision is a part of the RoboBond project, if you are interested in the robot documentation, please refer to the main RoboBond repository.
This repository contains code and setup instructions for dataset preparation, model training and model inference on the Raspberry Pi Zero 2W platform. 

## Features

The main feature is color signal recognition of LED towers mounted on RoboBond robots.


## Setup

##### Initial Setup
- You need a Raspberry Pi Zero 2W and a compatible MicroSD memory card 

1. Install a Raspberry Pi Imager from the [official website](https://www.raspberrypi.com/software/)

2. Install a Raspberry PI OS (Legacy, 32-bit) Lite image (Choose OS -> Raspberry Pi OS (other))
![](rpi_os_install.png)

3. Set up username, password, ssh options 

4. After successful installation, insert the MicroSD card into the device and boot the Raspberry Pi

5. (Optional)If you are using a Windows machine and experiencing problems when connecting via ssh, consider installing Bonjour Print Services from the [official website](https://support.apple.com/en-us/106380)

6. Connect to the Raspberry Pi via ssh

##### System configuration and library setup

1. Increase the SWAP size

>`# Stop Swap`\
`sudo nano /etc/dphys-swapfile`\
`CONF_SWAPSIZE=2048`\
`# Initialize Swap File`\
`sudo dphys-swapfile setup`\
`# Start Swap`\
`sudo dphys-swapfile swapon`



2. Install python 3.9.13

> `wget -qO - https://raw.githubusercontent.com/tvdsluijs/sh-python-installer/main/python.sh | sudo bash -s 3.9.13
`


3. Install the following libraries for the OpenCV support

>`sudo apt-get install libopenblas-dev`\
`sudo apt-get install libopenjp2-7`\
`sudo apt-get install ffmpeg libavcodec-dev`\
`sudo apt-get install libatlas-base-dev`


4. Install OpenCV

>`pip3 install opencv-python-headless==4.6.0.66`


5. Update f2py script PATH

> nano ~/.bashrc\
#add at the end of file:\
export PATH="$PATH:/home/missioncontrol/.local/bin"\
#save file\
source ~/.bashrc

6. Install onnxruntime from "wheels" folder

>`pip install onnxruntime-1.16.0-cp39-cp39-linux_armv7l.whl`


7. Install PIL library

>`pip install pillow`

8. Enable the Legacy camera support

>`sudo raspi-config -> interface options -> Legacy Camera -> enable -> yes -> Finish -> reboot` 

9. Install PiCamera library

>`sudo -H apt install python3-picamera`\
`sudo -H pip3 install --upgrade picamera[array]`


>(optional)`pip install picamera`



##### Run inference tests

`cd inference`\
`camera_inference.py`\
or\
`cascade_inference.py`



















