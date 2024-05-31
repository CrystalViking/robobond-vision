import os
import re

# Specify the directory where the images are located
directory = 'blue_rotated'

# Define the regex pattern to match '.jpg' followed by any characters and ending with '.jpg'
pattern = re.compile(r'\.jpg(?=.*\.xml)')

# Loop over all files in the directory
for filename in os.listdir(directory):
    # Check if the filename contains '.jpg' in the middle
    if pattern.search(filename):
        # Remove the first occurrence of '.jpg' in the middle of the filename
        new_filename = pattern.sub('', filename, 1)
        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))