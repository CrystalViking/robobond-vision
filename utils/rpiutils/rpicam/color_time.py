import onnxruntime as ort
import numpy as np
import time
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from collections import Counter

# Load the image and preprocess it
def preprocess_image(image_path, input_size):
    # Load image with PIL
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size, Image.LANCZOS)
    
    # Convert RGB to BGR
    img_data = np.array(img).astype('float32')

    # Normalize to [0, 1]
    img_data /= 255.0

    # Change data layout from HWC to CHW
    img_data = np.transpose(img_data, (2, 0, 1))

    # Add batch dimension
    img_data = np.expand_dims(img_data, axis=0)

    return img_data

# Post-process the output
def postprocess_output(output, threshold=0.5):
    boxes = []
    scores = []
    class_indices = []

    # Process each detection
    for detection in output[0]:
        x_min, y_min, x_max, y_max, score, class_idx = detection
        if score > threshold:
            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(score)
            class_indices.append(int(class_idx))  # Convert class index to int
    
    return boxes, scores, class_indices

# Function to convert HSV color to a name
def get_color_name(hsv_color):
    h, s, v = hsv_color
    if 0 <= h <= 10 or 160 <= h <= 180:
        return 'red'
    elif 35 <= h <= 85:
        return 'green'
    elif 100 <= h <= 140:
        return 'blue'
    elif 130 <= h <= 160:
        return 'purple'
    else:
        return 'unknown'

# Color detection in bounding box using KMeans
def detect_colors_in_bbox(image_path, x_min, y_min, x_max, y_max, num_clusters=1):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the height of each vertical segment
    bbox_height = int(y_max - y_min)
    segment_height = bbox_height // 20

    # List to store found colors
    found_colors = set()

    for i in range(20):
        segment_y_min = y_min + i * segment_height
        segment_y_max = segment_y_min + segment_height

        # Ensure the segment does not exceed the bounding box
        if segment_y_max > y_max:
            segment_y_max = y_max

        # Crop the image to the vertical segment
        segment_image = hsv_image[int(segment_y_min):int(segment_y_max), int(x_min):int(x_max)]
        
        # Reshape the image to be a list of pixels
        pixels = segment_image.reshape(-1, 3)
        
        # Apply KMeans to find the most dominant colors
        try:
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_

            # Count the number of pixels in each cluster
            labels = kmeans.labels_
            label_counts = Counter(labels)
            
            # Get the most common colors
            most_common_colors = [dominant_colors[i] for i in label_counts.keys()]
        except ValueError:
            # Handle the case when there are fewer unique colors than clusters
            most_common_colors = [pixels[0]]

        # Convert the dominant colors to names
        for color in most_common_colors:
            color_name = get_color_name(color)
            if color_name != 'unknown':
                found_colors.add(color_name)

    return list(found_colors)

# Load the ONNX model
model_path = 'yolov10t_b16_e300_s320_opset17_quint8_dynamic.onnx'
session = ort.InferenceSession(model_path)

# Define input and output names
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# Preprocess the image
image_path = 'image4.jpg'
input_size = (320, 320)  # Example input size, adjust based on your model's requirement
img_data = preprocess_image(image_path, input_size)

# Measure inference time
start_time = time.time()
outputs = session.run(output_names, {input_name: img_data})
end_time = time.time()

# Calculate and print inference time
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")

# Extract the single output from the model
output = outputs[0]

class_names = ["blue", "green", "purple", "red", "robot_v1", "tower_positive"]

# Post-process the output
boxes, scores, class_indices = postprocess_output(output)

# Print the post-processed detections and check for colors
print("Filtered detections:")
if boxes:
    for box, score, class_idx in zip(boxes, scores, class_indices):
        class_name = class_names[int(class_idx)]  # Convert class index to class name
        print(f"Box: {box}, Score: {score}, Class: {class_name}")
        
        # Measure color detection time
        color_detection_start_time = time.time()
        colors_found = detect_colors_in_bbox(image_path, *box)
        color_detection_end_time = time.time()

        # Calculate and print color detection time
        color_detection_time = color_detection_end_time - color_detection_start_time
        print(f"Colors found in the bounding box: {colors_found}")
        print(f"Color detection time: {color_detection_time:.4f} seconds")
else:
    print("No detections above the threshold.")
