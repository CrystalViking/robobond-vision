import onnxruntime as ort
import numpy as np
import time
import cv2
import os
from PIL import Image, ImageOps, ImageDraw
from picamera import PiCamera
from time import sleep
from io import BytesIO

from utils.utils import image_preprocess, mkdir, post_process, visualize, format_input_tensor, get_output_tensor

class_names_ndt = ['blue', 'green', 'purple', 'red']

def print_bboxes_ndt(dets, origin_img, class_names, score_thres=0.3):
    all_box = []
    src_height, src_width = origin_img.shape[:2]
    h_ratio = src_height / 160
    w_ratio = src_width / 160

    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thres:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                x0 = int(x0 * w_ratio)
                y0 = int(y0 * h_ratio)
                x1 = int(x1 * w_ratio)
                y1 = int(y1 * h_ratio)

                all_box.append([label, x0, y0, x1, y1, score])

    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        print(f"Class: {class_names_ndt[label]}, Score: {score * 100}, Coordinates: ({x0}, {y0}), ({x1}, {y1})")

def inference(interpreter, origin_img, args, img_print=True):
    img = image_preprocess(origin_img, args.input_shape)
    ort_inputs = {interpreter.get_inputs()[0].name: img[None, :, :, :]}
    
    start_time = time.time()
    output = interpreter.run(None, ort_inputs)
    end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time: {inference_time} seconds")

    results = post_process(output[0], len(class_names_ndt), 7, args.input_shape)
    
    print_bboxes_ndt(results[0], origin_img, class_names_ndt, 0.35)

    if img_print:
        result_image = visualize(results[0], origin_img, class_names_ndt, 0.35)
        return result_image

def image_process(interpreter, args, img=None):
    # If img is None, read from args.input_path
    if img is None:
        img = Image.open(args.input_path)
    origin_img = np.array(img)
    origin_img = inference(interpreter, origin_img, args)
    mkdir(args.output_path)
    output_path = os.path.join(args.output_path, args.input_path.split("/")[-1])
    cv2.imwrite(output_path, origin_img)

def preprocess_camera_image(stream):
    img = Image.open(stream).convert('RGB')
    img_data = np.array(img).astype('float32')
    img_data /= 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    return img_data, img

def postprocess_output(output, threshold=0.3):
    boxes = []
    scores = []
    class_indices = []
    for detection in output[0]:
        x_min, y_min, x_max, y_max, score, class_idx = detection
        if score > threshold:
            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(score)
            class_indices.append(int(class_idx))
    return boxes, scores, class_indices

def preprocess_for_second_model(image, target_size=(160, 160)):
    image = image.resize(target_size, Image.LANCZOS)
    img_data = np.array(image).astype('float32')
    img_data /= 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

# Load the first ONNX model
model_path = 'train11_y10n_tde_yolo_orig_e200_b16_s320_opset17_quint8_dynamic.onnx'

print("Loading model 1")
session = ort.InferenceSession(model_path)
print("Model 1 loaded")

# Define input and output names
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

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

# Preprocess the image
input_size = (320, 320)  # Example input size, adjust based on your model's requirement
img_data, original_img = preprocess_camera_image(stream)

# Measure inference time for the first model
start_time = time.time()
outputs = session.run(output_names, {input_name: img_data})
end_time = time.time()

# Calculate and print inference time
inference_time = end_time - start_time
print(f"First model inference time: {inference_time:.4f} seconds")

# Extract the single output from the model
output = outputs[0]

class_names_first_model = ["tower_positive"]

# Post-process the output
boxes, scores, class_indices = postprocess_output(output)

# Print the post-processed detections from the first model
print("Filtered detections from the first model:")
if boxes:
    for box, score, class_idx in zip(boxes, scores, class_indices):
        class_name = class_names_first_model[int(class_idx)]
        print(f"Box: {box}, Score: {score}, Class: {class_name}")
else:
    print("No detections above the threshold.")

# Load the second ONNX model
class Args:
    pass

args = Args()
args.model = "smoltower_ndtpm_0_5_e500_b1280_quint8_dynamic.onnx"
args.mode = "image"
args.input_path = 'image9.jpg'
args.camid = 0
args.output_path = './onnx_predict/'
args.score_thr = 0.3
args.input_shape = "160,160"

args.input_shape = tuple(map(int, args.input_shape.split(',')))
interpreter = ort.InferenceSession(args.model)

print("Second model detections:")
for box in boxes:
    x_min, y_min, x_max, y_max = map(int, box)
    
    # Crop the region
    cropped_img = original_img.crop((x_min, y_min, x_max, y_max))
    
    # Process the cropped image with the second model
    cropped_img_data = preprocess_for_second_model(cropped_img, args.input_shape)
    ort_inputs = {interpreter.get_inputs()[0].name: cropped_img_data}
    
    start_time = time.time()
    output = interpreter.run(None, ort_inputs)
    end_time = time.time()
    
    inference_time = end_time - start_time
    print(f"Second model inference time: {inference_time} seconds")
    
    results = post_process(output[0], len(class_names_ndt), 7, args.input_shape)
    print_bboxes_ndt(results[0], np.array(cropped_img), class_names_ndt, 0.35)
