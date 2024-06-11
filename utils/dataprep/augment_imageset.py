import os
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import xml.etree.ElementTree as ET
import numpy as np
import shutil

def adjust_bounding_box(bbox, aug, image_shape):
    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
    ], shape=image_shape)

    bbs_aug = aug.augment_bounding_boxes([bbs])[0]
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
    if len(bbs_aug.bounding_boxes) == 0:
        return None
    else:
        bbox_aug = bbs_aug.bounding_boxes[0]
        return [bbox_aug.x1, bbox_aug.y1, bbox_aug.x2, bbox_aug.y2]

def create_augmented_annotation(original_annotation_path, augmented_annotation_path, aug, image_shape):
    tree = ET.parse(original_annotation_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        xmlbox = obj.find('bndbox')
        bbox = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
        
        new_bbox = adjust_bounding_box(bbox, aug, image_shape)
        if new_bbox is None:
            root.remove(obj)
        else:
            xmlbox.find('xmin').text = str(int(new_bbox[0]))
            xmlbox.find('ymin').text = str(int(new_bbox[1]))
            xmlbox.find('xmax').text = str(int(new_bbox[2]))
            xmlbox.find('ymax').text = str(int(new_bbox[3]))

    tree.write(augmented_annotation_path)

def augment_images_and_annotations(folder_path):
    augmented_folder = os.path.join(folder_path, "augmented")
    if not os.path.exists(augmented_folder):
        os.makedirs(augmented_folder)

    seq_rotate = iaa.Affine(rotate=(-10, 10))
    seq_color = iaa.Sequential([
        iaa.AddToHueAndSaturation((-20, 20)),
        iaa.Multiply((0.8, 1.2), per_channel=0.5)
    ])
    seq_noise = iaa.AdditiveGaussianNoise(scale=(10, 60))

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            annotation_path = os.path.splitext(image_path)[0] + ".xml"

            if not os.path.exists(annotation_path):
                continue

            image = cv2.imread(image_path)
            image_shape = image.shape

            basename, ext = os.path.splitext(filename)

            # Rotation
            for angle in [-10, 10]:
                aug = iaa.Affine(rotate=angle)
                rotated_image = aug(image=image)
                rotated_filename = f"{basename}_rotate{angle}{ext}"
                rotated_image_path = os.path.join(augmented_folder, rotated_filename)
                cv2.imwrite(rotated_image_path, rotated_image)

                rotated_annotation_path = os.path.join(augmented_folder, f"{basename}_rotate{angle}.xml")
                create_augmented_annotation(annotation_path, rotated_annotation_path, aug, image_shape)

            # Color jitter
            jittered_image = seq_color(image=image)
            jittered_filename = f"{basename}_colorjitter{ext}"
            jittered_image_path = os.path.join(augmented_folder, jittered_filename)
            cv2.imwrite(jittered_image_path, jittered_image)

            jittered_annotation_path = os.path.join(augmented_folder, f"{basename}_colorjitter.xml")
            shutil.copy(annotation_path, jittered_annotation_path)

            # Adding noise
            noised_image = seq_noise(image=image)
            noised_filename = f"{basename}_noise{ext}"
            noised_image_path = os.path.join(augmented_folder, noised_filename)
            cv2.imwrite(noised_image_path, noised_image)

            noised_annotation_path = os.path.join(augmented_folder, f"{basename}_noise.xml")
            shutil.copy(annotation_path, noised_annotation_path)

if __name__ == "__main__":
    folder_path = "tower_blue_rotated"
    augment_images_and_annotations(folder_path)
