import numpy as np
import itertools

# Step activation for logic gates
def step(x):
    return 1 if x >= 0 else 0

# Linear activation for regression
def linear(x):
    return x

# Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ----------------------
# LOGIC GATES
# ----------------------
def generate_truth_table(n, gate_type='AND'):
    """Generates truth table inputs and outputs for n-input AND/OR gates."""
    inputs = list(itertools.product([0, 1], repeat=n))
    if gate_type == 'AND':
        outputs = [int(all(x)) for x in inputs]
    elif gate_type == 'OR':
        outputs = [int(any(x)) for x in inputs]
    else:
        raise ValueError("Invalid gate_type: choose 'AND' or 'OR'")
    return np.array(inputs), np.array(outputs)

def train_perceptron_gate(X, y, lr=0.1, epochs=100, label=""):
    """Trains a perceptron with step activation for logic gates."""
    weights = np.zeros(X.shape[1])
    bias = 0
    print(f"\n🚀 Training Perceptron for {label} Gate")
    for epoch in range(epochs):
        errors = 0
        print(f"Epoch {epoch+1}:")
        for xi, target in zip(X, y):
            output = step(np.dot(xi, weights) + bias)
            error = target - output
            if error != 0:
                weights += lr * error * xi
                bias += lr * error
                errors += 1
            print(f"  Input: {xi} Target: {target} -> Pred: {output} | Weights: {weights} Bias: {bias}")
        if errors == 0:
            break
    print(f"✅ Final Weights for {label} Gate: {weights}, Bias: {bias}\n")
    return weights, bias

# ----------------------
# LINEAR FUNCTION (REGRESSION)
# ----------------------
def generate_linear_data(n_features=3, n_samples=10, w=None, bias=5):
    """Generates data based on y = w1*x1 + w2*x2 + ... + wn*xn + b"""
    X = np.random.rand(n_samples, n_features)
    true_w = w if w is not None else np.random.uniform(-1, 1, size=n_features)
    y = np.dot(X, true_w) + bias
    return X, y, true_w

def train_linear_perceptron(X, y, lr=0.01, epochs=100):
    """Trains a linear regression model (perceptron with linear activation)."""
    weights = np.zeros(X.shape[1])
    bias = 0
    for epoch in range(epochs):
        y_pred = np.dot(X, weights) + bias
        error = y - y_pred
        weights += lr * np.dot(X.T, error)
        bias += lr * error.sum()
        print(f"Epoch {epoch+1}: MSE = {mse(y, y_pred):.4f}")
    print(f"✅ Final Weights: {weights}, Bias: {bias}\n")
    return weights, bias

# ----------------------
# MAIN TESTS
# ----------------------
if __name__ == "__main__":
    print("==============================")
    print("🔹 2-Input Logic Gates")
    print("==============================")
    for gate in ['AND', 'OR']:
        X, y = generate_truth_table(2, gate)
        w, b = train_perceptron_gate(X, y, label=f"2-Input {gate.upper()}")

    print("==============================")
    print("🔹 n-Input Logic Gates (n=3,4)")
    print("==============================")
    for n in [3, 4]:
        for gate in ['AND', 'OR']:
            X, y = generate_truth_table(n, gate)
            w, b = train_perceptron_gate(X, y, label=f"{n}-Input {gate.upper()}")

    print("==============================")
    print("🔹 Linear Regression with 3 Features")
    print("==============================")
    X, y, true_w = generate_linear_data(n_features=3)
    w, b = train_linear_perceptron(X, y)
    print(f"🎯 True Weights: {true_w}\n")

    print("==============================")
    print("🔹 Linear Regression with n=4,5 Features")
    print("==============================")
    for n in [4, 5]:
        print(f"\n--- n = {n} Features ---")
        X, y, true_w = generate_linear_data(n_features=n)
        w, b = train_linear_perceptron(X, y)
        print(f"🎯 True Weights: {true_w}\n")