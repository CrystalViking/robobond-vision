import os
import random
import shutil
import xml.etree.ElementTree as ET
import cv2


folders = [] # Folders with images and annotations
dataset_dir = "dataset_name"
train_images_dir = os.path.join(dataset_dir, "train", "images")
train_labels_dir = os.path.join(dataset_dir, "train", "labels")
val_images_dir = os.path.join(dataset_dir, "val", "images")
val_labels_dir = os.path.join(dataset_dir, "val", "labels")

# Create the required directory structure
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

def convert_pascal_voc_to_yolo(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_annotations = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        xmlbox = obj.find('bndbox')
        x_min = int(xmlbox.find('xmin').text)
        y_min = int(xmlbox.find('ymin').text)
        x_max = int(xmlbox.find('xmax').text)
        y_max = int(xmlbox.find('ymax').text)

        # Convert to YOLO format
        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        yolo_annotations.append(f"{label} {x_center} {y_center} {width} {height}")

    return yolo_annotations

def process_folder(folder):
    images = [f for f in os.listdir(folder) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(images)
    train_split = int(0.8 * len(images))

    for i, image_name in enumerate(images):
        image_path = os.path.join(folder, image_name)
        xml_path = os.path.splitext(image_path)[0] + '.xml'

        # Load the image to get its dimensions
        img = cv2.imread(image_path)
        height, width, _ = img.shape

        yolo_annotations = convert_pascal_voc_to_yolo(xml_path, width, height)
        yolo_label = "\n".join(yolo_annotations)

        if i < train_split:
            dest_img_dir = train_images_dir
            dest_label_dir = train_labels_dir
        else:
            dest_img_dir = val_images_dir
            dest_label_dir = val_labels_dir

        shutil.copy(image_path, dest_img_dir)
        label_path = os.path.join(dest_label_dir, os.path.splitext(image_name)[0] + '.txt')

        with open(label_path, 'w') as f:
            f.write(yolo_label)

for folder in folders:
    process_folder(folder)
