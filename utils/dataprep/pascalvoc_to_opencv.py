import xml.etree.ElementTree as ET
import glob
import os

output_file = open('positive_images_file.txt', 'w')

# Define the directories to process
directories = ['folder_with_positive_images']

for directory in directories:
    for xml_file in glob.glob(f'{directory}/*.xml'):  
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = os.path.join(directory, root.find('filename').text)
        boxes = []

        # Extract image dimensions from the 'size' tag
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        for object in root.findall('object'):
            bndbox = object.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            # convert to width and height
            width = xmax - xmin
            height = ymax - ymin

            # Check if the bounding box is within the dimensions of the image
            if xmin >= 0 and ymin >= 0 and xmax <= img_width and ymax <= img_height and width > 0 and height > 0:
                boxes.append(f"{xmin} {ymin} {width} {height}")

        output_line = f"{filename} {len(boxes)} {' '.join(boxes)}\n"
        output_file.write(output_line)

output_file.close()