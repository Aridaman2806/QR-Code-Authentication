# Traditional_Model.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Ensure the outputs directory exists
if not os.path.exists("outputs"):
    raise FileNotFoundError("The 'outputs' directory does not exist. Run Feature_Extraction.py first to generate features and labels.")

# Load features and labels
try:
    features = np.load("outputs/features.npy")
    labels = np.load("outputs/labels.npy")
except FileNotFoundError:
    raise FileNotFoundError("Features or labels file not found in 'outputs/'. Run Feature_Extraction.py first.")

# Verify the shape of the loaded data
print(f"Loaded features shape: {features.shape}")
print(f"Loaded labels shape: {labels.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["First Print", "Second Print"])
conf_matrix = confusion_matrix(y_test, y_pred)

# Save the evaluation metrics to a file
with open("outputs/traditional_model_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(class_report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix))

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(class_report)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nMetrics saved to 'outputs/traditional_model_metrics.txt'.")

# Optionally, save the trained model for future use
import joblib
joblib.dump(model, "outputs/random_forest_model.pkl")
print("Trained Random Forest model saved to 'outputs/random_forest_model.pkl'.")