import onnxruntime as ort
import numpy as np
import time
from PIL import Image, ImageOps, ImageDraw
from picamera import PiCamera
from time import sleep
from io import BytesIO
from PIL import ImageOps
import cv2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



def pad_image_to_square(image, target_size=(160, 160)):
    padded_img = ImageOps.pad(image, target_size, method=Image.LANCZOS, color=(0, 0, 0))
    return padded_img

# Load the image and preprocess it
def preprocess_image(image_path, input_size):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size, Image.LANCZOS)
    
    img_data = np.array(img).astype('float32')
    img_data /= 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    return img_data, img

def preprocess_camera_image(stream):
    img = Image.open(stream).convert('RGB')
    img_data = np.array(img).astype('float32')
    img_data /= 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    return img_data, img

# Post-process the output
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

# Draw bounding box on the image
def draw_bounding_box(image, box, outline="red"):
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=outline, width=2)
    return image

# Compute Intersection over Union (IoU)
def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

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
model_path = 'train11_y10n_tde_yolo_orig_e200_b16_s320_opset17_quint8_dynamic.onnx'

print("Loading model 1")
session = ort.InferenceSession(model_path)
print("Model 1 loaded")

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

print("Starting camera")
stream = BytesIO()
camera = PiCamera()
camera.resolution = (320, 320)

print("Camera warmup")
sleep(2)
print("Camera ready")
camera.capture(stream, format='jpeg')
stream.seek(0)

input_size = (320, 320)
img_data, original_img = preprocess_camera_image(stream)

start_time = time.time()
outputs = session.run(output_names, {input_name: img_data})
end_time = time.time()

inference_time = end_time - start_time
print(f"First model inference time: {inference_time:.4f} seconds")

output = outputs[0]

class_names_first_model = ["tower_positive"]

boxes, scores, class_indices = postprocess_output(output)

print("Filtered detections from the first model:")
if boxes:
    for box, score, class_idx in zip(boxes, scores, class_indices):
        class_name = class_names_first_model[int(class_idx)]
        print(f"Box: {box}, Score: {score}, Class: {class_name}")
else:
    print("No detections above the threshold.")

if boxes:
    x_min, y_min, x_max, y_max = boxes[0]
    cropped_image = original_img.crop((x_min, y_min, x_max, y_max))
    padded_image = pad_image_to_square(cropped_image, target_size=(160, 160))
    cropped_image_np = np.array(padded_image)

    # Convert the image from PIL to OpenCV format (BGR)
    cropped_image_np = cv2.cvtColor(cropped_image_np, cv2.COLOR_RGB2BGR)

    start_color_detection = time.time()

    # Convert the image from BGR to LAB color space
    image_lab = cv2.cvtColor(cropped_image_np, cv2.COLOR_BGR2Lab)
    pixels = image_lab.reshape(-1, 3).astype(np.float32)

    # Calculate the positions of the centroids based on the size of the padded image (160x160)
    start_y = 25
    end_y = 160 - 35
    padding = 2
    num_centroids = 8
    step = (end_y - start_y - 2 * padding) // (num_centroids - 1)
    positions = [(160 // 2, start_y + padding + i * step) for i in range(num_centroids)]

    centers = np.array([image_lab[y, x] for x, y in positions], dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_centroids, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(cropped_image_np.shape)
    segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_Lab2BGR)

    color_ranges = {
        'blue': ([170, 70, 80], [230, 100, 100]),
        'red': ([345, 60, 60], [355, 100, 100]),
        'green': ([95, 85, 85], [135, 100, 100]),
        'purple': ([260, 70, 65], [275, 100, 100])
    }

    num_color_centroids = 0
    detected_colors = []

    for i, (x, y) in enumerate(positions):
        cv2.circle(segmented_image_bgr, (x, y), 3, (0, 255, 0), -1)
        left = max(0, x - 10)
        right = min(cropped_image_np.shape[1] - 1, x + 10)
        region = segmented_image_bgr[y, left:right]

        max_count = 0
        detected_color = None

        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)

            lower_3d = np.full(region.shape, lower)
            upper_3d = np.full(region.shape, upper)

            mask = cv2.inRange(region, lower_3d, upper_3d)
            count = cv2.countNonZero(mask)

            if count > max_count:
                max_count = count
                detected_color = color

        if detected_color is not None:
            num_color_centroids += 1
            detected_colors.append(detected_color)

        print(f'Centroid {i+1}: The detected color is {detected_color}')

    end_color_detection = time.time()
    color_detection_time = end_color_detection - start_color_detection
    print(f"Color detection time: {color_detection_time:.4f} seconds")

    print(f'Colors detected at centroids: {detected_colors}')