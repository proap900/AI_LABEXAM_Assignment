import csv
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# --- Load dataset from CSV ---
def load_dataset_from_csv(filename="binary_data.csv"):
    data = []
    if not os.path.exists(filename):
        print("CSV not found, using hardcoded fallback data.")
        return [
            [165, 60, 1], [170, 65, 1], [160, 55, 0], [175, 70, 1],
            [155, 50, 0], [168, 62, 1], [162, 58, 0], [172, 68, 1],
            [158, 53, 0], [167, 61, 1]
        ]
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            data.append([float(row[0]), float(row[1]), int(row[2])])
    return data

# --- Euclidean distance ---
def euclidean(p1, p2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

# --- kNN classification ---
def knn_classify(train_data, test_point, k):
    distances = [(euclidean(row[:-1], test_point), row[-1]) for row in train_data]
    distances.sort()
    k_labels = [label for (_, label) in distances[:k]]
    return Counter(k_labels).most_common(1)[0][0]

# --- Main Execution ---
def main():
    data = load_dataset_from_csv()
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    k = 3

    # Prompt user test input
    try:
        h = float(input("Enter height (cm): "))
        w = float(input("Enter weight (kg): "))
        test_point = [h, w]
        predicted = knn_classify(train_data, test_point, k)
        print(f"\n🔍 Prediction for ({h}, {w}) → Class: {'Pass ✅' if predicted == 1 else 'Fail ❌'}")
    except:
        print("⚠️ Invalid input. Skipping custom test point.")
        test_point = None
        predicted = None

    # Evaluate
    y_true = [row[-1] for row in test_data]
    y_pred = [knn_classify(train_data, row[:-1], k) for row in test_data]
    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(test_data)
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=["Fail", "Pass"], output_dict=True)
    df_report = pd.DataFrame(report_dict).T.round(2)

    # --- Prepare Plot ---
    fig, axs = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f"kNN (k={k}) — Accuracy: {acc*100:.2f}%", fontsize=16)

    # Subplot 1: Confusion Matrix
    axs[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axs[0].set_title("Confusion Matrix", fontsize=14)
    axs[0].set_xticks([0, 1])
    axs[0].set_yticks([0, 1])
    axs[0].set_xticklabels(["Fail", "Pass"])
    axs[0].set_yticklabels(["Fail", "Pass"])
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("True")
    for i in range(2):
        for j in range(2):
            axs[0].text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=12)

    # Subplot 2: Classification Report Table
    axs[1].axis('off')
    axs[1].set_title("Classification Report (Table)", fontsize=14)
    table = axs[1].table(cellText=df_report.values,
                         rowLabels=df_report.index,
                         colLabels=df_report.columns,
                         cellLoc='center',
                         loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Subplot 3: Decision Boundary
    x_min, x_max = min([r[0] for r in train_data]) - 5, max([r[0] for r in train_data]) + 5
    y_min, y_max = min([r[1] for r in train_data]) - 5, max([r[1] for r in train_data]) + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    zz = np.array([[knn_classify(train_data, [x, y], k) for x, y in zip(xrow, yrow)]
                   for xrow, yrow in zip(xx, yy)])
    axs[2].contourf(xx, yy, zz, alpha=0.3, cmap=plt.cm.coolwarm)

    # Train points
    for row in train_data:
        label = row[2]
        color = 'blue' if label == 1 else 'red'
        axs[2].scatter(row[0], row[1], c=color, marker='o', s=70, edgecolor='black')

    # Test points
    for row in test_data:
        label = row[2]
        color = 'blue' if label == 1 else 'red'
        axs[2].scatter(row[0], row[1], c=color, marker='x', s=90, edgecolor='black')

    # User Input
    if test_point:
        axs[2].scatter(test_point[0], test_point[1], color='purple', s=150, marker='D', edgecolor='black')
        axs[2].text(test_point[0] + 1, test_point[1], f"{'Pass ✅' if predicted == 1 else 'Fail ❌'}", fontsize=12, color='black')

    axs[2].set_title("Decision Boundary + Data", fontsize=14)
    axs[2].set_xlabel("Height (cm)")
    axs[2].set_ylabel("Weight (kg)")

    # Clean Legend
    axs[2].legend(["Train Pass (O)", "Train Fail (O)", "Test Pass (X)", "Test Fail (X)", "Your Input (D)"],
                  loc='upper left', fontsize=9)
    axs[2].grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()