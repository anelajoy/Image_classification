#Preprocess.py
import os
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMAGE_FOLDER = "Training"
# Use a raw string or forward slashes so the backslash isn't read as escape character for Windows paths
TEST_FOLDER = "Testing/Test"

def load_and_preprocess_image(image_id, target_size=(224, 224)):

    """Generate filename from ID and load the image."""

    filename = f"{str(image_id)}.png"  # Convert ID to string and append .png

    img_path = os.path.join(IMAGE_FOLDER, filename)  # Construct full path

    if not os.path.exists(img_path):  # Check if the file exists
        print(f"Warning: Image {img_path} not found!")  # Debugging info
        return np.zeros((target_size[0], target_size[1], 3))  # Return blank image


    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize image
    return img_array

def load_data(csv_path, n_per_class=400):
    """Load CSV file, select images, and preprocess them."""
    df = pd.read_csv(csv_path)
    
    # Ensure ID is treated as a string (to match filename format)
    df["ID"] = df["ID"].astype(str)
    
    # Guard against asking for more images than exist in the dataset
    n_normal = (df["Disease_Risk"] == 0).sum()
    n_abnormal = (df["Disease_Risk"] == 1).sum()
    n = min(n_per_class, n_normal, n_abnormal)
    if n < n_per_class:
        print(f"Warning: only {n} images per class available"
              f" (Normal: {n_normal}, Abnormal: {n_abnormal}). Using {n}.")
        
    # Filter normal and abnormal images
    normal_images = df[df["Disease_Risk"] == 0].sample(n=n, random_state=42)
    abnormal_images = df[df["Disease_Risk"] == 1].sample(n=n, random_state=42)
    
     # Combine and shuffle the selected images
    selected_df = pd.concat([normal_images, abnormal_images]).sample(frac=1, random_state=42)
 
    # Load and preprocess images
    selected_df["image_data"] = selected_df["ID"].apply(load_and_preprocess_image)
 
    # Stack image arrays into a NumPy array
    X = np.stack(selected_df["image_data"].values)
 
    # Extract labels
    y = selected_df["Disease_Risk"].astype(int).values
 
    return X, y