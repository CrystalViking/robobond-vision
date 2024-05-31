import os
import cv2

def resize_image_and_bbox(image_path):
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
    # Save image
    cv2.imwrite(image_path, img)

# Get list of images
images = [f for f in os.listdir('towers_negative') if f.endswith('.jpg') or f.endswith('.png')]

# Apply resize function to each image
for image in images:
    resize_image_and_bbox('towers_negative/' + image)