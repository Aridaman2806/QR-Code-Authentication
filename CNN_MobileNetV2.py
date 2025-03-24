# CNN_Model.py
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Helper function to save classification report as PDF
def save_classification_report_to_pdf(report, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.text(0.1, 0.9, report, fontsize=10, family='monospace', verticalalignment='top')
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f"Classification report saved to '{filename}'.")

# Define paths to the directories containing the images
first_print_dir = "Assignment Data\First Print"
second_print_dir = "Assignment Data\Second Print"

# Load the images
first_images = [cv2.imread(os.path.join(first_print_dir, f)) for f in os.listdir(first_print_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
second_images = [cv2.imread(os.path.join(second_print_dir, f)) for f in os.listdir(second_print_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Filter out any None values
first_images = [img for img in first_images if img is not None]
second_images = [img for img in second_images if img is not None]

# Check if any images were loaded
if not first_images or not second_images:
    raise ValueError("No images were loaded. Check the directory paths and ensure they contain images.")

# Preprocess images (resize to 128x128, normalize)
images = [cv2.resize(img, (128, 128)) / 255.0 for img in first_images + second_images]
X = np.array(images)
y = np.array([0] * len(first_images) + [1] * len(second_images))  # 0=first, 1=second

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation (reverted to original intensity)
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build the model using MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Unfreeze the last 10 layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Reduced from 20 to 10
    layer.trainable = False

# Add a custom head with L2 regularization
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # Added L2 regularization
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy', metrics=['accuracy'])

# Add a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,  # Reduced from 30 to 20
    validation_data=(X_test, y_test),
    callbacks=[lr_scheduler],
    verbose=1
)

# Make predictions on the test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["First Print", "Second Print"])
conf_matrix = confusion_matrix(y_test, y_pred)

# Save the evaluation metrics to a text file
with open("outputs/cnn_model_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(class_report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix))

# Save the classification report as a PDF
save_classification_report_to_pdf(class_report, "outputs/cnn_model_classification_report.pdf")

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(class_report)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nMetrics saved to 'outputs/cnn_model_metrics.txt'.")

# Save the trained model
model.save("outputs/cnn_model.h5")
print("Trained CNN model saved to 'outputs/cnn_model.h5'.")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig("outputs/cnn_training_history.png")
plt.show()