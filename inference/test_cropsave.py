import onnxruntime as ort
import numpy as np
import time
from PIL import Image, ImageOps, ImageDraw

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

    return img_data, img

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

# Draw bounding box on the image
def draw_bounding_box(image, box, outline="red"):
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=outline, width=2)
    return image

# Compute Intersection over Union (IoU)
def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the coordinates of the intersection rectangle
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    # Calculate the area of the intersection rectangle
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Calculate the area of both bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

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

# Load the ONNX model
model_path = 'yolov10t_b16_e300_s320_opset17_quint8_dynamic.onnx'
session = ort.InferenceSession(model_path)

# Define input and output names
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# Preprocess the image
input_size = (320, 320)  # Example input size, adjust based on your model's requirement
img_data, original_img = preprocess_image('image4.jpg', input_size)

# Measure inference time for the first model
start_time = time.time()
outputs = session.run(output_names, {input_name: img_data})
end_time = time.time()

# Calculate and print inference time
inference_time = end_time - start_time
print(f"First model inference time: {inference_time:.4f} seconds")

# Extract the single output from the model
output = outputs[0]

class_names_first_model = ["blue", "green", "purple", "red", "robot_v1", "tower_positive"]

# Post-process the output
boxes, scores, class_indices = postprocess_output(output)

# Print the post-processed detections from the first model
print("Filtered detections from the first model:")
if boxes:
    for box, score, class_idx in zip(boxes, scores, class_indices):
        class_name = class_names_first_model[int(class_idx)]  # Convert class index to class name
        print(f"Box: {box}, Score: {score}, Class: {class_name}")
else:
    print("No detections above the threshold.")

# Load the second ONNX model
second_model_path = 'yolov10t_td_b16_e100_s160_opset17_quint8_dynamic.onnx'
second_session = ort.InferenceSession(second_model_path)

# Define input and output names for the second model
second_input_name = second_session.get_inputs()[0].name
second_output_names = [output.name for output in second_session.get_outputs()]

class_names_second_model = ["blue", "green", "purple", "red"]

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
        
        # Draw bounding box on the padded image
        padded_img_with_box = draw_bounding_box(padded_img, second_box)

        # Save the padded image with bounding box
        padded_img_with_box.save("cropped.jpg")

# Print the first recognized class of overlapping boxes
if detected_classes:
    pass
    #print(f"First recognized class of overlapping boxes: {detected_classes[0]}")
