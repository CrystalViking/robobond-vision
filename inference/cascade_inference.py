
import cv2
import onnxruntime as ort
import numpy as np
import time
from PIL import Image, ImageOps, ImageDraw
from picamera import PiCamera
from time import sleep
from io import BytesIO

# Load Haar cascade classifier
cascade_path = 'cascade_2.xml'
haar_cascade = cv2.CascadeClassifier(cascade_path)

# Load the image and preprocess it for Haar cascade
def preprocess_image_for_haar(image_path):
    img = Image.open(image_path).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_cv, img

def preprocess_camera_image_for_haar(stream):
    img = Image.open(stream).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_cv, img

# Detect objects using Haar cascade
def detect_objects_with_haar(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    detected_objects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return detected_objects

# Post-process the output
def postprocess_output(output, threshold=0.3):
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

# Add padding to make the image 160x160 while maintaining aspect ratio
def pad_image_to_square(image, target_size=(160, 160)):
    padded_img = ImageOps.pad(image, target_size, method=Image.LANCZOS, color=(0, 0, 0))
    return padded_img

# Filter overlapping boxes based on IoU threshold
def filter_overlapping_boxes(boxes, class_indices, iou_threshold=0.9):
    filtered_boxes = []
    filtered_class_indices = []
    for i in range(len(boxes)):
        overlap = False
        for j in range(len(filtered_boxes)):
            if compute_iou(boxes[i], filtered_boxes[j]) > iou_threshold:
                overlap = True
                break
        if not overlap:
            filtered_boxes.append(boxes[i])
            filtered_class_indices.append(class_indices[i])
    return filtered_boxes, filtered_class_indices

# Load the second ONNX model
second_model_path = 'yolov10t_td_b16_e100_s160_opset17_quint8_dynamic.onnx'
print("Loading model 2")
second_session = ort.InferenceSession(second_model_path)
print("Model 2 loaded")

# Define input and output names for the second model
second_input_name = second_session.get_inputs()[0].name
second_output_names = [output.name for output in second_session.get_outputs()]

class_names_second_model = ["blue", "green", "purple", "red"]

# Take a picture
print("Starting camera")
stream = BytesIO()
camera = PiCamera()
camera.resolution = (320, 320)

print("Camera warmup")
sleep(2)
print("Camera ready")
camera.capture(stream, format='jpeg')
stream.seek(0)

# Preprocess the image for Haar cascade
img_cv, original_img = preprocess_camera_image_for_haar(stream)

# Detect objects using Haar cascade
detected_objects = detect_objects_with_haar(img_cv)

# Convert detected objects to bounding boxes
boxes = [[x, y, x + w, y + h] for (x, y, w, h) in detected_objects]

# Print the detections from the Haar cascade
print("Detections from the Haar cascade:")
if boxes:
    for box in boxes:
        print(f"Box: {box}")
else:
    print("No detections.")

# Process each bounding box and run inference on the second model
print("Second model detections:")
detected_classes = []
for box in boxes:
    x_min, y_min, x_max, y_max = map(int, box)
    
    # Crop the region
    cropped_img = original_img.crop((x_min, y_min, x_max, y_max))
    
    # Pad the cropped image to 160x160
    padded_img = pad_image_to_square(cropped_img, target_size=(160, 160))
    
    # Preprocess the padded image
    cropped_img_data = np.array(padded_img).astype('float32')
    cropped_img_data /= 255.0
    cropped_img_data = np.transpose(cropped_img_data, (2, 0, 1))
    cropped_img_data = np.expand_dims(cropped_img_data, axis=0)
    
    # Measure inference time for the second model
    start_time = time.time()
    second_outputs = second_session.run(second_output_names, {second_input_name: cropped_img_data})
    end_time = time.time()
    
    # Calculate and print inference time for the second model
    second_inference_time = end_time - start_time
    print(f"Second model inference time: {second_inference_time:.4f} seconds")

    # Extract the single output from the second model
    second_output = second_outputs[0]

    # Post-process the output
    second_boxes, _, second_class_indices = postprocess_output(second_output)

    # Filter overlapping boxes
    filtered_boxes, filtered_class_indices = filter_overlapping_boxes(second_boxes, second_class_indices)

    # Print detected class names for the second model
    for second_box, class_idx in zip(filtered_boxes, filtered_class_indices):
        class_name = class_names_second_model[class_idx]
        print(f"Detected class: {class_name}")
        detected_classes.append(class_name)

# Print the first recognized class of overlapping boxes
if detected_classes:
    pass
    #print(f"First recognized class of overlapping boxes: {detected_classes[0]}")
