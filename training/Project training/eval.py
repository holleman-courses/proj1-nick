import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set up directories for the dataset (same as during training)
data_dir = 'Dataset'  # Path to the directory containing 'Negative' and 'Positive'

# Image size to match the input size of the model
IMG_SIZE = (224, 224)

# Create ImageDataGenerator for preprocessing
eval_datagen = ImageDataGenerator(rescale=1./255)  # Same rescaling as during training

# Load the evaluation data using flow_from_directory
eval_data = eval_datagen.flow_from_directory(
    data_dir,  # Main directory containing 'Negative' and 'Positive'
    target_size=IMG_SIZE,
    batch_size=2,  # Same batch size as during training
    class_mode='binary',  # Binary classification
    shuffle=False,  # Don't shuffle for evaluation
    
)

# Load the trained model from the saved .h5 file
model = tf.keras.models.load_model('better_96_col.h5')
print("Model loaded successfully.")
model.summary()


# Function to convert grayscale images to RGB
def grayscale_to_rgb(grayscale_images):
    # Repeat the grayscale images across the 3 color channels (R, G, B)
    return np.repeat(grayscale_images, 3, axis=-1)

# Evaluate the model on the evaluation dataset
loss, accuracy = model.evaluate(eval_data, steps=eval_data.samples // eval_data.batch_size)

# Print the evaluation results
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# True labels (0 or 1)
y_true = eval_data.classes

# Get predicted probabilities
y_pred_prob = model.predict(eval_data, steps=eval_data.samples // eval_data.batch_size, verbose=1)

# Convert probabilities to binary labels (0 or 1)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Extract TP, TN, FP, FN from the confusion matrix
TN, FP, FN, TP = cm.ravel()

# Plotting the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Optionally: Print TP, TN, FP, FN
print(f'True Positive (TP): {TP}')
print(f'True Negative (TN): {TN}')
print(f'False Positive (FP): {FP}')
print(f'False Negative (FN): {FN}')
