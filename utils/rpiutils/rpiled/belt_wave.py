import time
import colorsys
from rpi_ws281x import PixelStrip, Color

# LED strip configuration:
LED_COUNT = 24  # Number of LED pixels.
LED_PIN = 21  # GPIO pin connected to the pixels (must support PWM).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10  # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 10  # Set to 0 for darkest and 255 for brightest
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

try:
    hue_shift = 0.0
    step_size = 0.0001  # Smaller step size for smoother transition
    while True:
        for i, belt in enumerate(reversed(belts)):  # Iterate over belts in reverse order
            # Generate a gradient of hues across the belt.
            hue = ((i * step_size + hue_shift) % 1.0) * 360.0

            # Convert the hue to RGB.
            r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue/360.0, 1.0, 1.0)]
            for led in belt:  # Set color for each LED in the belt
                strip.setPixelColor(led, Color(r, g, b))

        # Update the strip to display the colors.
        strip.show()

        # Shift the starting point of the gradient.
        hue_shift = (hue_shift + step_size) % 1.0

        # Wait for a short period of time before updating again.
        time.sleep(0.1)
except KeyboardInterrupt:
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, Color(0,0,0))
    strip.show()