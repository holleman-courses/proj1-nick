import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up directories for the dataset (same as during training)
data_dir = 'Dataset_reduced'  # Path to the directory containing 'Negative' and 'Positive'

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
    shuffle=False  # Don't shuffle for evaluation
)

# Load the trained model from the saved .h5 file
model = tf.keras.models.load_model('trained_model.h5')
print("Model loaded successfully.")

# Evaluate the model on the evaluation dataset
loss, accuracy = model.evaluate(eval_data, steps=eval_data.samples // eval_data.batch_size)

# Print the evaluation results
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")
