import os
import random


def keep_n_samples(image_dir, label_dir, num_to_keep):
    # Get list of images and corresponding labels
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]  # Adjust the extension if needed
    label_files = [f.replace('.jpg', '.txt') for f in image_files]  # Adjust the extension if needed

    # Check if number of images is greater than the number to keep
    if len(image_files) > num_to_keep:
        # Randomly select images to keep
        selected_images = random.sample(image_files, num_to_keep)
        selected_labels = [f.replace('.jpg', '.txt') for f in selected_images]

        # Determine images and labels to delete
        images_to_delete = set(image_files) - set(selected_images)
        labels_to_delete = set(label_files) - set(selected_labels)

        # Delete the images
        for img in images_to_delete:
            os.remove(os.path.join(image_dir, img))

        # Delete the labels
        for lbl in labels_to_delete:
            os.remove(os.path.join(label_dir, lbl))

        print(f"Kept {num_to_keep} images and their corresponding annotations in {image_dir}. Deleted the rest.")
    else:
        print(
            f"The directory {image_dir} contains {len(image_files)} images, which is less than or equal to {num_to_keep}. No deletion necessary.")


# Directories for train, val, and test sets
datasets = {
    'train': {
        'images': r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\aug_data\train\images',
        'labels': r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\aug_data\train\labels',
        'num_to_keep': 1000
    },
    'val': {
        'images': r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\aug_data\val\images',
        'labels': r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\aug_data\val\labels',
        'num_to_keep': 200
    },
    'test': {
        'images': r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\aug_data\test\images',
        'labels': r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\aug_data\test\labels',
        'num_to_keep': 200
    }
}

# Process each dataset
for dataset, paths in datasets.items():
    keep_n_samples(paths['images'], paths['labels'], paths['num_to_keep'])