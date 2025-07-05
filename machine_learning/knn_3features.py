import csv
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# --- Load dataset ---
def load_dataset(filename="multi_class_data.csv"):
    data = []
    if not os.path.exists(filename):
        print("CSV not found, using fallback data.")
        return [
            [5.1, 3.5, 1.4, 0], [4.9, 3.0, 1.3, 0], [5.0, 3.4, 1.5, 0],
            [7.0, 3.2, 4.7, 1], [6.4, 3.2, 4.5, 1], [6.9, 3.1, 4.9, 1],
            [5.5, 2.3, 4.0, 2], [6.5, 2.8, 4.6, 2], [5.7, 2.8, 4.1, 2],
            [6.3, 3.3, 6.0, 2], [5.8, 2.7, 5.1, 2], [6.1, 3.0, 4.8, 2]
        ]
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([float(row[0]), float(row[1]), float(row[2]), int(row[3])])
    return data

# --- Distance + kNN ---
def euclidean(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def knn_classify(train_data, test_point, k):
    distances = [(euclidean(row[:-1], test_point), row[-1]) for row in train_data]
    distances.sort()
    top_k = [label for _, label in distances[:k]]
    return Counter(top_k).most_common(1)[0][0]

# --- Main logic ---
def main():
    data = load_dataset()
    k = 5
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)

    # Input test point
    try:
        f1 = float(input("Enter sepal length: "))
        f2 = float(input("Enter sepal width: "))
        f3 = float(input("Enter petal length: "))
        test_point = [f1, f2, f3]
        pred = knn_classify(train_data, test_point, k)
        print(f"\n🔍 Predicted class: {pred}")
    except:
        print("⚠️ Invalid input.")
        return

    # --- Plotting only scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("kNN Classification — 3 Features", fontsize=14)

    color_map = {0: 'red', 1: 'green', 2: 'blue'}
    markers = {'train': 'o', 'test': 'x'}

    # Train points
    for row in train_data:
        label = row[-1]
        ax.scatter(row[0], row[1], s=row[2]*25, c=color_map[label], marker=markers['train'], alpha=0.6, label=f"Train C{label}")

    # Test points
    for row in test_data:
        label = row[-1]
        ax.scatter(row[0], row[1], s=row[2]*25, c=color_map[label], marker=markers['test'], alpha=0.9, label=f"Test C{label}")

    # User input
    ax.scatter(test_point[0], test_point[1], s=test_point[2]*30, c='purple', marker='D', edgecolor='black', label="Your Input")
    ax.text(test_point[0]+0.1, test_point[1], f"Pred: {pred}", fontsize=12)

    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Sepal Width")
    ax.grid(True)

    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()