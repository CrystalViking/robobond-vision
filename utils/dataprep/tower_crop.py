import os
import xml.etree.ElementTree as ET
from PIL import Image

# Define the input and output folder mappings
folders = {
    "folder1": "output_folder1",
    "folder2": "output_folder2",
    "folder3": "output_folder3",
    "folder4": "output_folder4",
    "folder5": "output_folder5",
    "folder6": "output_folder6",
}

# Create the output folders if they don't already exist
for folder in folders.values():
    if not os.path.exists(folder):
        os.makedirs(folder)

# Function to process each image and its corresponding annotation file
def process_images(folders):
    for input_folder, output_folder in folders.items():
        for file in os.listdir(input_folder):
            if file.endswith('.xml'):
                # Parse XML file to get the annotation data
                xml_path = os.path.join(input_folder, file)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Find the corresponding image file
                image_file = file.replace('.xml', '.jpg')
                image_path = os.path.join(input_folder, image_file)
                if not os.path.exists(image_path):
                    print(f"Image file {image_file} not found for annotation {file}")
                    continue
                
                # Open the image
                image = Image.open(image_path)
                
                # Iterate through all objects in the XML
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name == 'tower_positive':
                        # Get bounding box coordinates
                        bndbox = obj.find('bndbox')
                        xmin = int(bndbox.find('xmin').text)
                        ymin = int(bndbox.find('ymin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymax = int(bndbox.find('ymax').text)
                        
                        # Crop the image using the bounding box coordinates
                        cropped_image = image.crop((xmin, ymin, xmax, ymax))
                        
                        # Save the cropped image to the appropriate output folder
                        output_image_path = os.path.join(output_folder, image_file)
                        cropped_image.save(output_image_path)
                        print(f"Cropped image saved to {output_image_path}")

# Process the images and annotations
process_images(folders)
