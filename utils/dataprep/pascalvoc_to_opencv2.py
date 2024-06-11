import xml.etree.ElementTree as ET
import glob
import os

output_file = open('blue_from_pi.txt', 'w')

# Define the directories to process
directories = ['blue_from_pi_annotated']

for directory in directories:
    for xml_file in glob.glob(f'{directory}/*.xml'):  
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get the filename from the xml file path and append the image extension
        filename, _ = os.path.splitext(xml_file)
        filename = f'{filename}.jpg'  # or .png, .bmp, etc. depending on your images

        boxes = []
        for object in root.findall('object'):
            bndbox = object.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Calculate width and height of the bounding box
            width = xmax - xmin
            height = ymax - ymin

            boxes.append(f'{xmin} {ymin} {width} {height}')

        # Write the bounding box information to the output file
        output_file.write(f'{filename} {len(boxes)} {" ".join(boxes)}\n')

output_file.close()