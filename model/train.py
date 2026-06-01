#train.py
from preprocess import load_data
from model import create_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
X, y = load_data("RFMiD_Training_Labels (1).csv")

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta = 0.001,
    patience=5, 
    restore_best_weights=True,
)

# Create and train model
model = create_model()
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stopping],
)

# Save model
model.save("abnormal_detection_model.h5")
print("Model saved as abnormal_detection_model.h5")