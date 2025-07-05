import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
def load_data(csv_path=None, fallback_data=None):
    if csv_path and Path(csv_path).is_file():
        print(f"📂 Loading data from: {csv_path}")
        return pd.read_csv(csv_path)
    elif fallback_data:
        print("⚠️ CSV not found. Using fallback data.")
        return pd.DataFrame(fallback_data)
    else:
        raise FileNotFoundError("CSV not found and no fallback data provided.")

# ---------------------------
def train_and_plot(df, feature_cols, target_col):
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Evaluation Metrics:")
    print(f"  ➤ Mean Squared Error (MSE): {mse:.4f}")
    print(f"  ➤ R-squared (R2 Score): {r2:.4f}")
    print(f"  ➤ Coefficients: {model.coef_}")
    print(f"  ➤ Intercept: {model.intercept_}")

    # ---------- Visualization ----------
    if X.shape[1] == 1:
        plt.scatter(X_test, y_test, color='blue', label='Actual')
        plt.plot(X_test, y_pred, color='red', label='Prediction')
        plt.xlabel(feature_cols[0])
        plt.ylabel(target_col)
        plt.title("Linear Regression (1 Feature)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif X.shape[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='blue', label='Actual', s=40)

        if len(X_test) >= 3:
            try:
                ax.plot_trisurf(X_test[:, 0], X_test[:, 1], y_pred, color='red', alpha=0.5)
            except Exception as e:
                print("⚠️ Plotting trisurf failed. Showing scatter only.")
        else:
            print("⚠️ Not enough points to plot trisurf. Showing scatter only.")

        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel(feature_cols[1])
        ax.set_zlabel(target_col)
        ax.set_title("Linear Regression (2 Features)")
        plt.tight_layout()
        plt.show()

    elif X.shape[1] >= 3:
        print("🔢 Data has more than 2 features. No 3D plot shown.")

# ---------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ✅ Replace these with your actual CSV filenames if needed
    file_2f = os.path.join(script_dir, "data_regression_2f.csv")
    file_3f = os.path.join(script_dir, "data_regression_3f.csv")

    fallback_data_2f = {
        'height': [160, 165, 170, 175, 180, 185, 190, 195, 200, 205],
        'weight': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        'score':  [30, 35, 40, 50, 60, 65, 70, 75, 80, 85]
    }

    fallback_data_3f = {
        'x1': np.random.rand(10),
        'x2': np.random.rand(10),
        'x3': np.random.rand(10),
        'y': 2 * np.random.rand(10) + 3 * np.random.rand(10) - 1 * np.random.rand(10) + 5
    }

    print("\n===== Linear Regression with 2 Features =====")
    df2 = load_data(file_2f, fallback_data_2f)
    train_and_plot(df2, ['height', 'weight'], 'score')

    print("\n===== Linear Regression with 3 Features =====")
    df3 = load_data(file_3f, fallback_data_3f)
    train_and_plot(df3, ['x1', 'x2', 'x3'], 'y')