import os
from PIL import Image
from torchvision import transforms
import random

# Paths
input_root = "dataset"
output_root = "augmented_dataset"

# Output image size
image_size = (64, 64)

# Number of augmented versions per original
copies_per_image = 5

# Define the full transform pipeline
base_transform = transforms.Compose([
    transforms.Grayscale(),                        # Convert to grayscale
    transforms.Resize(image_size),                 # Resize to 64x64
])

augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])

# Ensure output folders exist
for label in ['left', 'right', 'no_arrow']:
    os.makedirs(os.path.join(output_root, label), exist_ok=True)

# Process each image
for label in ['left', 'right', 'no_arrow']:
    input_dir = os.path.join(input_root, label)
    output_dir = os.path.join(output_root, label)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_dir, fname)
        image = Image.open(img_path).convert('RGB')

        # Apply base transforms (grayscale + resize)
        base_image = base_transform(image)

        # Save the processed original image
        base_image.save(os.path.join(output_dir, fname))

        # Generate augmentations
        for i in range(copies_per_image):
            aug_image = augmentation(base_image)
            base_name, ext = os.path.splitext(fname)
            aug_name = f"{base_name}_aug{i}{ext}"
            aug_image.save(os.path.join(output_dir, aug_name))

print("Augmentation complete.")
