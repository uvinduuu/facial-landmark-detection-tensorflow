import os
import cv2
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

def process_mat_files(image_dir, mat_dir, output_train, output_val, train_ratio=0.8):
    """
    Extract landmarks from .mat files and save them to training and validation text files.
    Args:
        image_dir: Directory containing the images.
        mat_dir: Directory containing the .mat files.
        output_train: Path to save the training landmarks file.
        output_val: Path to save the validation landmarks file.
        train_ratio: Proportion of the dataset to use for training (default is 0.8).
    """
    all_data = []

    for mat_file in os.listdir(mat_dir):
        if mat_file.endswith('.mat'):
            mat_path = os.path.join(mat_dir, mat_file)
            data = loadmat(mat_path)

            # Extract landmarks
            pt2d = data['pt2d']
            landmarks = np.transpose(pt2d).flatten()  # Flatten to [x1, y1, ..., x68, y68]

            # Get the corresponding image name
            image_name = os.path.splitext(mat_file)[0] + ".jpg"
            image_path = os.path.join(image_dir, image_name)

            # Check if the image exists
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            # Read image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                print(f"Invalid image: {image_path}")
                continue

            height, width = image.shape[:2]

            # Normalize landmarks
            landmarks[0::2] /= width  # Normalize x-coordinates
            landmarks[1::2] /= height  # Normalize y-coordinates

            # Append data as a tuple (image_name, landmarks)
            all_data.append((image_name, landmarks))

    # Split data into training and validation sets
    train_data, val_data = train_test_split(all_data, test_size=1 - train_ratio, random_state=42)

    # Write training data
    with open(output_train, 'w') as f_train:
        for image_name, landmarks in train_data:
            f_train.write(f"{image_name} " + " ".join(map(str, landmarks)) + "\n")

    # Write validation data
    with open(output_val, 'w') as f_val:
        for image_name, landmarks in val_data:
            f_val.write(f"{image_name} " + " ".join(map(str, landmarks)) + "\n")

    print(f"Training landmarks saved to {output_train}")
    print(f"Validation landmarks saved to {output_val}")

# Example usage
image_dir = r"D:\FLD\archive\300W-LP\300W_LP\AFW"
mat_dir = r"D:\FLD\archive\300W-LP\300W_LP\AFW"
output_train = r"D:\FLD\FLD\landmarks_train.txt"
output_val = r"D:\FLD\FLD\landmarks_val.txt"

process_mat_files(image_dir, mat_dir, output_train, output_val)
