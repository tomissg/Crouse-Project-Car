import os
import glob
import random
import cv2
import numpy as np
import imgaug.augmenters as iaa

# Define the snow augmentation pipeline using imgaug
snow_augmenter = iaa.Sequential([
    iaa.Multiply((1.3, 1.7)),  # Increase brightness
    iaa.Sometimes(1, iaa.Grayscale(alpha=(0.6, 0.9))),  # Slight desaturation to mimic snow
    iaa.imgcorruptlike.Snow(severity=2)  # severity can be adjusted from 1 to 5
])

# Define the dataset directories
data_directories = [
    r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\VisDrone2019-DET-train\VisDrone2019-DET-train\images',
    r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\VisDrone2019-DET-val\VisDrone2019-DET-val\images',
    r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\VisDrone2019-DET-test-dev\images'
]

label_directories = [
    r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\VisDrone2019-DET-train\VisDrone2019-DET-train\annotations',
    r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\VisDrone2019-DET-val\VisDrone2019-DET-val\annotations',
    r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\VisDrone2019-DET-test-dev\annotations'
]

aug_data_usage = ['train', 'val', 'test']
num_to_keep = {'train': 200, 'val': 40, 'test': 40}

# Path to the augmented data directory
aug_dir = 'aug_data_2'

# Create the main aug_data directory if it doesn't exist
os.makedirs(aug_dir, exist_ok=True)

for usage, data_dir, labels_dir in zip(aug_data_usage, data_directories, label_directories):
    # Create directories for images and annotations
    os.makedirs(os.path.join(aug_dir, usage, 'images'), exist_ok=True)
    os.makedirs(os.path.join(aug_dir, usage, 'labels'), exist_ok=True)

    output_dir_images = os.path.join(aug_dir, usage, 'images')
    output_dir_labels = os.path.join(aug_dir, usage, 'labels')

    # Get all image files in the current directory
    all_image_paths = glob.glob(os.path.join(data_dir, '*.jpg'))

    # Filter images that have corresponding labels with label 4
    selected_image_paths = []
    for image_path in all_image_paths:
        image_filename = os.path.basename(image_path)
        annotation_file = os.path.join(labels_dir, image_filename.replace('.jpg', '.txt'))
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
            if any(len(line.strip().split(',')) >= 6 and line.strip().split(',')[5] == '4' for line in lines):
                selected_image_paths.append(image_path)

    # Randomly select a subset of images
    if len(selected_image_paths) > num_to_keep[usage]:
        selected_image_paths = random.sample(selected_image_paths, num_to_keep[usage])

    # Initialize scaling factor list for this usage
    scaling_factors = []

    # Iterate over the selected images
    for image_path in selected_image_paths:
        # Load the image
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        # Compute the scaling factors
        scale_x = 1024 / image_width
        scale_y = 1024 / image_height
        scaling_factors.append((scale_x, scale_y))

        # Convert the image to RGB for augmentation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply the snow effect
        augmented_image = snow_augmenter(image=image_rgb)

        # Resize the augmented image to 640x640
        resized_image = cv2.resize(augmented_image, (1024, 1024))

        # Convert the augmented image back to BGR for saving with OpenCV
        resized_image_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

        # Save the augmented image to the output directory
        image_filename = os.path.basename(image_path)
        output_path_image = os.path.join(output_dir_images, image_filename)
        cv2.imwrite(output_path_image, resized_image_bgr)

        # Get the corresponding annotation file path
        annotation_file = os.path.join(labels_dir, image_filename.replace('.jpg', '.txt'))

        # Check if the annotation file exists
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                lines = f.readlines()

            # Iterate over each line in the annotation file
            for line in lines:
                parts = line.strip().split(',')

                # Check if the line corresponds to label 4
                if len(parts) >= 6 and parts[5] == '4':
                    bbox_left = float(parts[0]) * scale_x/1024
                    bbox_top = float(parts[1]) * scale_y/1024
                    bbox_width = float(parts[2]) * scale_x/1024
                    bbox_height = float(parts[3]) * scale_y/1024

                    yolo_annotation = f"0 {bbox_left} {bbox_top} {bbox_width} {bbox_height}\n"
                    output_path_label = os.path.join(output_dir_labels, image_filename.replace('.jpg', '.txt'))

                    with open(output_path_label, 'a') as label_file:
                        label_file.write(yolo_annotation)

    # Save the scaling factors to a text file in the root directory
    with open(os.path.join(aug_dir, usage, f'scale_factors.txt'), 'w') as f:
        for scale_x, scale_y in scaling_factors:
            f.write(f"{scale_x} {scale_y}\n")

    print(f"Augmentation complete for {usage}. Augmented images and annotations are saved.")

print("All augmentations complete.")