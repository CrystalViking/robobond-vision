from picamera import PiCamera
from time import sleep


camera = PiCamera()

sleep(5)
camera.capture('/home/username/pictures/image.jpg')

