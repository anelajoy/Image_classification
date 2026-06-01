#predict.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

# Load trained model once
model = load_model("abnormal_detection_model.h5")

def preprocess_image(image_path, target_size=(224, 224)):
    # Load a single image and preprocess it the same way as in training
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize image to [0, 1]
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    score = prediction[0][0] #scalar sigmoid output
    return "Abnormal" if score > 0.5 else "Normal"

# Example usage
#print(predict_image("images12.jpeg"))

def predict_and_validate(model, image_folder, label_csv, target_size=(224, 224)):
    # Load ground truth from CSV
    df = pd.read_csv(label_csv)
    df["ID"] = df["ID"].astype(str)  # Ensure ID is string for filename matching
    df["filename"] = df["ID"].apply(lambda x: f"{x}.png")  # Generate filename from ID
    
    image_data = []
    true_labels = []
    image_names = []
    
    for _, row in df.iterrows():
        filename = row["filename"]
        label = row["Disease_Risk"]
        image_path = os.path.join(image_folder, filename)
        
        if os.path.exists(image_path):
            image_data.append(preprocess_image(image_path, target_size))
            true_labels.append(label)
            image_names.append(filename)
        else:
            print(f"Warning: Image {image_path} not found in {image_folder}!")
            
    if not image_data:
        raise ValueError("No valid images found for prediction. Check the path and filenames")
    
    # Convert to numpy arrays and predict
    X = np.array(image_data)
    predictions = model.predict(X, batch_size=32)
    predicted_labels = (predictions > 0.5).astype(int).flatten()
    
    #Evaluation
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=["Normal", "Abnormal"]))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))
    
    results_df = pd.DataFrame({
        "filename": image_names,
        "true_label": true_labels,
        "predicted_label": predicted_labels
    })
    return results_df

if __name__ == "__main__":
    results_df = predict_and_validate(model, r"Testing/Test", "RFMiD_Testing_Labels.csv")
    results_df.to_csv("prediction_results.csv", index=False)
    print("Prediction results saved to prediction_results.csv")
    print("Prediction and validation complete.")