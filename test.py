import cv2
import matplotlib.pyplot as plt

# Function to plot bounding boxes on an image
def plot_bounding_boxes(image_path, annotations_path):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB for Matplotlib

    # Read bounding box annotations from the text file
    with open(annotations_path, 'r') as f:
        annotations = f.readlines()

    bbox_left, bbox_top, bbox_width, bbox_height = [], [], [], []
    for annotation in annotations:
        parts = annotation.strip().split()
        if len(parts) >= 4:
            bbox_left.append(float(parts[1]))
            bbox_top.append(float(parts[2]))
            bbox_width.append(float(parts[3]))
            bbox_height.append(float(parts[4]))

    # Plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    # Plot the bounding boxes
    for left, top, width, height in zip(bbox_left, bbox_top, bbox_width, bbox_height):
        # Calculate the coordinates of the bounding box
        x1 = left
        y1 = top
        x2 = left + width
        y2 = top + height
        print([x1, x2, x2, x1, x1])
        # Plot the bounding box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='red', linewidth=2)

    # Show the plot
    plt.axis('off')
    plt.show()

# Example usage
image_path = r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\aug_data\val\images\0000002_00005_d_0000014.jpg'
annotations_path = r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\aug_data\val\labels\0000002_00005_d_0000014.txt'

plot_bounding_boxes(image_path, annotations_path)