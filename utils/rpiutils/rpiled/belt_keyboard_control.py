import time
from rpi_ws281x import PixelStrip, Color
import keyboard


# LED strip configuration:
LED_COUNT = 24  # Number of LED pixels.
LED_PIN = 21  # GPIO pin connected to the pixels (must support PWM).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10  # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 80  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False  # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0  # set to '1' for GPIOs 13, 19, 41, 45 or 53


# Create PixelStrip object with appropriate configuration.
strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)

# Intialize the library (must be called once before other functions).
strip.begin()


# Define the LED numbers for each belt
belts = [
    [15, 0, 16],
    [14, 1, 17],
    [13, 2, 18],
    [12, 3, 19],
    [11, 4, 20],
    [10, 5, 21],
    [9, 6, 22],
    [8, 7, 23]
]

color_dict = {
    "red": Color(255, 0, 0),
    "green": Color(0, 255, 0),
    "blue": Color(0, 0, 255),
    "yellow": Color(255, 255, 0),
    "purple": Color(128, 0, 128),
    "cyan": Color(0, 255, 255),
    "white": Color(255, 255, 255),
}

color_name = "blue"  # Default color
current_belt = 0  # The belt that is currently lit up

def light_up_belt(strip, belt):
    """Light up a specific belt."""
    for led in belt:
        strip.setPixelColor(led, color_dict[color_name])  # Set the specified pixel to the provided color.
    strip.show()  # Update the strip to match

try:
    # Light up the first belt initially
    light_up_belt(strip, belts[current_belt])
    while True:  # Keep the program running
        command = input("Enter command: ")
        if command == "k":
            # Turn off the current belt
            for led in belts[current_belt]:
                strip.setPixelColor(led, Color(0, 0, 0))
            # Move to the next belt
            current_belt = (current_belt + 1) % len(belts)
            # Light up the next belt
            light_up_belt(strip, belts[current_belt])
except KeyboardInterrupt:
    # If Ctrl+C is pressed, turn off all the LEDs and exit.
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, Color(0, 0, 0))
    strip.show()
