import csv
import os
import xml.etree.ElementTree as ET

# Define the path to the dataset
dataset_path = "./positive_images_dataset"

# Get a list of all .xml files in the dataset directory
xml_files = [f for f in os.listdir(dataset_path) if f.endswith('.xml')]

# Open the .csv file in write mode
with open('dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["ImageID", "LabelName", "Xmin", "Ymin", "Xmax", "Ymax"])

    # For each .xml file
    for xml_file in xml_files:
        # Parse the .xml file
        tree = ET.parse(os.path.join(dataset_path, xml_file))
        root = tree.getroot()

        # Extract the necessary information
        ImageID = root.find('filename').text
        LabelName = root.find('object/name').text
        Xmin = root.find('object/bndbox/xmin').text
        Ymin = root.find('object/bndbox/ymin').text
        Xmax = root.find('object/bndbox/xmax').text
        Ymax = root.find('object/bndbox/ymax').text

        # Write the information to the .csv file
        writer.writerow([ImageID, LabelName, Xmin, Ymin, Xmax, Ymax])