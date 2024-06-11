import cv2
import numpy as np

# Load the image
image = cv2.imread('./cv2_color_detection/blue_edit.jpg')

# Convert the image from BGR to LAB color space
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# Reshape the image to be a list of pixels
pixels = image_lab.reshape(-1, 3)

# Convert the pixels to float32 type
pixels = pixels.astype(np.float32)

# Calculate the positions of the centroids
start_y = 25
end_y = 160 - 35
padding = 2
num_centroids = 8
step = (end_y - start_y - 2 * padding) // (num_centroids - 1)
positions = [(160 // 2, start_y + padding + i * step) for i in range(num_centroids)]

# Get the color values at these positions
centers = np.array([image_lab[y, x] for x, y in positions], dtype=np.float32)

# Perform k-means clustering to find the most dominant colors
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixels, num_centroids, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
# Convert the centers to 8-bit values
centers = np.uint8(centers)

# Map the labels to the centers
segmented_image = centers[labels.flatten()]

# Reshape back to the original image
segmented_image = segmented_image.reshape(image.shape)

# Convert back to BGR color space
segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_Lab2BGR)

# Draw the centroids on the image
for x, y in positions:
    cv2.circle(segmented_image_bgr, (x, y), 3, (0, 255, 0), -1)


# Define the color ranges in BGR color space
color_ranges = {
    'blue': ([170, 70, 80], [230, 100, 100]),
    'red': ([345, 60, 60], [355, 100, 100]),
    'green': ([95, 85, 85], [135, 100, 100]),
    'purple': ([260, 70, 65], [275, 100, 100])
}
    #'red1': ([0, 100, 100], [10, 60, 60]),
    #'green': ([0, 100, 0], [100, 255, 100]),
    #
    #'purple': ([100, 0, 100], [255, 100, 255])

# Initialize the counter and list of detected colors
true_color_centroids = 0
detected_colors = []

# Draw the centroids on the image and detect the color
for i, (x, y) in enumerate(positions):
    cv2.circle(segmented_image_bgr, (x, y), 3, (0, 255, 0), -1)

    # Check the colors 15 pixels to the left and right of the centroid
    left = max(0, x - 10)
    right = min(image.shape[1] - 1, x + 10)
    region = segmented_image_bgr[y, left:right]

    max_count = 0
    detected_color = None

    # For each color, create a mask and count the number of pixels
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        # Convert lower and upper to 3D arrays
        lower_3d = np.full(region.shape, lower)
        upper_3d = np.full(region.shape, upper)

        mask = cv2.inRange(region, lower_3d, upper_3d)
        count = cv2.countNonZero(mask)

        # If this color has more pixels than the current maximum, update the detected color
        if count > max_count:
            max_count = count
            detected_color = color

    # If a color was detected, increment the counter and add to the list
    if detected_color is not None:
        true_color_centroids += 1
        detected_colors.append(detected_color)

    print(f'Centroid {i+1}: The detected color is {detected_color}')

print(f'The number of centroids in the "true color" area is {true_color_centroids}')
print(f'True colors detected at centroids: {detected_colors}')
