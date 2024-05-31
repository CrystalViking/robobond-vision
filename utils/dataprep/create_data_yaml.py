import os
import yaml
import glob

# Paths
dataset_dir = "larger_dataset"
train_images_dir = os.path.join(dataset_dir, "train", "images")
train_labels_dir = os.path.join(dataset_dir, "train", "labels")
val_images_dir = os.path.join(dataset_dir, "val", "images")
val_labels_dir = os.path.join(dataset_dir, "val", "labels")

# Function to gather all labels from the dataset
def gather_labels(labels_dir):
    labels = set()
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(label_file, 'r') as file:
            for line in file:
                label = line.strip().split()[0]
                labels.add(label)
    return sorted(labels)

# Function to create data.yaml
def create_data_yaml(labels, yaml_path):
    label_map = {label: idx for idx, label in enumerate(labels)}

    data_yaml = {
        'train': os.path.join(dataset_dir, 'train', 'images'),
        'val': os.path.join(dataset_dir, 'val', 'images'),
        'nc': len(labels),
        'names': [label for label in labels]
    }

    with open(yaml_path, 'w') as file:
        yaml.dump(data_yaml, file, default_flow_style=False)

    return label_map

# Convert labels in annotation files to integers
def convert_labels_to_integers(labels_dir, label_map):
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        new_lines = []
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                label = parts[0]
                new_label = label_map[label]
                new_line = " ".join([str(new_label)] + parts[1:])
                new_lines.append(new_line)
        
        with open(label_file, 'w') as file:
            file.write("\n".join(new_lines))

# Main script
if __name__ == "__main__":
    train_labels = gather_labels(train_labels_dir)
    val_labels = gather_labels(val_labels_dir)
    
    # Combine and deduplicate labels from both train and val sets
    all_labels = sorted(set(train_labels + val_labels))
    
    # Create data.yaml and get label_map
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    label_map = create_data_yaml(all_labels, yaml_path)
    
    # Convert labels in both train and val sets
    convert_labels_to_integers(train_labels_dir, label_map)
    convert_labels_to_integers(val_labels_dir, label_map)
    
    print(f"data.yaml created at {yaml_path}")
    print(f"Label mapping: {label_map}")
