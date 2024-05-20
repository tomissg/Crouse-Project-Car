import os
import glob
import cv2
import shutil
import numpy as np
import imgaug.augmenters as iaa

# Define the snow augmentation pipeline using imgaug
snow_augmenter = iaa.Sequential([
    iaa.imgcorruptlike.Snow(severity=5)  # severity can be adjusted from 1 to 5
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

# Path to the augmented data directory
aug_dir = 'aug_data'

# Create the main aug_data directory if it doesn't exist
os.makedirs(aug_dir, exist_ok=True)

for usage in aug_data_usage:
    # Create directories for images and annotations
    os.makedirs(os.path.join(aug_dir, usage, 'images'), exist_ok=True)
    os.makedirs(os.path.join(aug_dir, usage, 'labels'), exist_ok=True)

    # Iterate over each image in the data directory
    for data_dir, labels_dir in zip(data_directories, label_directories):
        output_dir_images = os.path.join(aug_dir, usage, 'images')
        output_dir_labels = os.path.join(aug_dir, usage, 'labels')

        # Initialize scaling factor list for this usage
        scaling_factors = []

        for image_path in glob.glob(os.path.join(data_dir, '*.jpg')):
            # Load the image
            image = cv2.imread(image_path)
            image_height, image_width = image.shape[:2]

            # Compute the scaling factors
            scale_x = 640 / image_width
            scale_y = 640 / image_height
            scaling_factors.append((scale_x, scale_y))

            # Convert the image to RGB for augmentation
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply the snow effect
            augmented_image = snow_augmenter(image=image_rgb)

            # Resize the augmented image to 640x640
            resized_image = cv2.resize(augmented_image, (640, 640))

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
                        bbox_left = float(parts[0]) * scale_x
                        bbox_top = float(parts[1]) * scale_y
                        bbox_width = float(parts[2]) * scale_x
                        bbox_height = float(parts[3]) * scale_y

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