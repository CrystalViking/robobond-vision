import os
import cv2
import xml.etree.ElementTree as ET

def resize_image_and_bbox(image_path, xml_path):
    # Load image
    img = cv2.imread(image_path)

    # Calculate aspect ratio and padding
    h, w = img.shape[:2]
    scale = min(160.0/w, 160.0/h)
    pad_x = max(160 - scale*w, 0) * 0.5
    pad_y = max(160 - scale*h, 0) * 0.5

    # Resize image and add padding
    img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    img = cv2.copyMakeBorder(img, int(pad_y), int(pad_y), int(pad_x), int(pad_x), cv2.BORDER_CONSTANT)

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Update size
    size = root.find('size')
    size.find('width').text = str(img.shape[1])
    size.find('height').text = str(img.shape[0])

    # Update bounding box
    for box in root.iter('bndbox'):
        box.find('xmin').text = str(int(int(box.find('xmin').text) * scale + pad_x))
        box.find('ymin').text = str(int(int(box.find('ymin').text) * scale + pad_y))
        box.find('xmax').text = str(int(int(box.find('xmax').text) * scale + pad_x))
        box.find('ymax').text = str(int(int(box.find('ymax').text) * scale + pad_y))

    # Write back XML
    tree.write(xml_path)

    # Save image
    cv2.imwrite(image_path, img)

# Get list of images
images = [f for f in os.listdir('towers_positive') if f.endswith('.jpg') or f.endswith('.png')]

# Apply resize function to each image
for image in images:
    resize_image_and_bbox('towers_positive/' + image, 'towers_positive/' + os.path.splitext(image)[0] + '.xml')