import yaml

# Define object categories
object_categories = {
    0: 'car'
}

# Define paths
train_path = r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\aug_data\train'
val_path = r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\aug_data\val'

# Create data dictionary
data = {
    'train': train_path,
    'val': val_path,
    'nc': len(object_categories),
    'names': list(object_categories.values())
}

# Specify the path for the YAML file
yaml_file_path = 'data.yaml'

# Write dictionary to YAML file
with open(yaml_file_path, 'w') as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)