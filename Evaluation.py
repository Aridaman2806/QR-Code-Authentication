# Evaluation.py
import os
import matplotlib.pyplot as plt

# Check if the required files exist
if not os.path.exists("outputs/traditional_model_metrics.txt"):
    raise FileNotFoundError("Traditional model metrics file not found. Run Traditional_Model.py first.")
if not os.path.exists("outputs/cnn_model_metrics.txt"):
    raise FileNotFoundError("CNN model metrics file not found. Run CNN_Model.py first.")

# Load the metrics from both models
with open("outputs/traditional_model_metrics.txt", "r") as f:
    traditional_metrics = f.read()

with open("outputs/cnn_model_metrics.txt", "r") as f:
    cnn_metrics = f.read()

# Extract accuracy values for plotting
traditional_accuracy = float(traditional_metrics.split("Accuracy: ")[1].split("\n")[0])
cnn_accuracy = float(cnn_metrics.split("Accuracy: ")[1].split("\n")[0])

# Print the comparison
print("=== Traditional Model (Random Forest) Metrics ===")
print(traditional_metrics)
print("\n=== CNN Model Metrics ===")
print(cnn_metrics)

# Reference the classification report PDFs
print("\nClassification reports have been saved as PDFs:")
print("- Traditional Model: outputs/traditional_model_classification_report.pdf")
print("- CNN Model: outputs/cnn_model_classification_report.pdf")

# Reference the CNN training history plot
if os.path.exists("outputs/cnn_training_history.png"):
    print("\nCNN training history plot saved to: outputs/cnn_training_history.png")
else:
    print("\nCNN training history plot not found. Run CNN_Model.py to generate it.")

# Plot a bar chart comparing accuracies
plt.figure(figsize=(6, 4))
models = ['Random Forest', 'CNN']
accuracies = [traditional_accuracy, cnn_accuracy]
plt.bar(models, accuracies, color=['blue', 'orange'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
plt.savefig("outputs/model_accuracy_comparison.png")
plt.show()
print("\nAccuracy comparison plot saved to: outputs/model_accuracy_comparison.png")