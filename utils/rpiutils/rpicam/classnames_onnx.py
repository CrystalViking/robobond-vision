import onnxruntime as ort
import numpy as np
from PIL import Image

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

# Load the ONNX model
model_path = 'yolov10t_b16_e200_s640.onnx'
session = ort.InferenceSession(model_path)

# Define input and output names
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# Preprocess the image
input_size = (640, 640)  # Example input size, adjust based on your model's requirement
img_data = preprocess_image('image3.jpg', input_size)

# Run inference
outputs = session.run(output_names, {input_name: img_data})

# Extract the single output from the model
output = outputs[0]


class_names = ["blue", "green", "purple", "red", "robot_v1", "tower_positive"]

# Post-process the output
boxes, scores, class_indices = postprocess_output(output)

# Print the post-processed detections
print("Filtered detections:")
if boxes:
    for box, score, class_idx in zip(boxes, scores, class_indices):
        class_name = class_names[int(class_idx)]  # Convert class index to class name
        print(f"Box: {box}, Score: {score}, Class: {class_name}")
else:
    print("No detections above the threshold.")
