from PIL import Image
import os

directory = 'to_rotate/purple_rotated'  # replace with your directory

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # add any file type you need
        img = Image.open(os.path.join(directory, filename))
        img = img.rotate(-90, expand=True)  # rotate 90 degrees to the right
        name, ext = os.path.splitext(filename)
        img.save(os.path.join(directory, name + "_rotated.jpg"), quality=95)

print("All images have been rotated.")