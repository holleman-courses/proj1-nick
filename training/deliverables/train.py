import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Set the directory for images (combined negative and positive in one folder)
data_dir = 'Dataset_reduced'  # Path to the main directory containing both "Negative" and "Positive" directories

# Verify the data directory exists
print(f"Data directory exists: {os.path.exists(data_dir)}")

# Image size for resizing
IMG_SIZE = (224, 224)

# Create ImageDataGenerator for preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize images to [0, 1]
    validation_split=0.2  # 20% of data will be used for validation
)

# Load the data using the generator (both training and validation data)
train_data = train_datagen.flow_from_directory(
    data_dir,  # Main directory containing 'Negative' and 'Positive'
    target_size=IMG_SIZE,
    batch_size=2,  # Small batch size for small dataset
    class_mode='binary',
    shuffle=True,
    subset='training'  # Specify that this is for training data
)

# Validation data using the same generator
val_data = train_datagen.flow_from_directory(
    data_dir,  # Main directory containing 'Negative' and 'Positive'
    target_size=IMG_SIZE,
    batch_size=2,  # Same batch size as training
    class_mode='binary',
    shuffle=True,
    subset='validation'  # Specify that this is for validation data
)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model with a small learning rate for a small dataset
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)

# Train the model using the training data and validation data
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,  # Number of steps per epoch
    epochs=10,  # Number of epochs
    validation_data=val_data,  # Validation data
    validation_steps=val_data.samples // val_data.batch_size  # Number of validation steps
)

# Save the trained model
model.save('trained_model.h5')
print("Model trained and saved as 'trained_model.h5'.")

# Print model summary to see the parameter count
model.summary()

# Print training and validation accuracy
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]  # Handle validation accuracy if available

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%" if val_accuracy != 'N/A' else "No validation accuracy available.")
