import pandas as pd
import matplotlib.pyplot as plt
from code.model import run_models

# Load dataset
data = pd.read_csv("dataset/housing.csv")

# Convert categorical variables
data = pd.get_dummies(data, drop_first=True)

# Define features and target
X = data.drop("price", axis=1)
y = data["price"] / 10000000

print(data.head())
print("Dataset shape:", data.shape)
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Run all regression models
results = run_models(X, y)

# Extract results
mse_lr = results["Linear"]
mse_ridge = results["Ridge"]
mse_lasso = results["LASSO"]
best_mse = results["Dynamic"]

# Print summary
print("\nModel Performance Summary")
print("--------------------------")
print("Linear Regression MSE:", mse_lr)
print("Ridge Regression MSE:", mse_ridge)
print("LASSO Regression MSE:", mse_lasso)
print("Dynamic LASSO Best MSE:", best_mse)

# Plot comparison graph
models = ["Linear", "Ridge", "LASSO", "Dynamic LASSO"]
mse_values = [mse_lr, mse_ridge, mse_lasso, best_mse]

plt.figure(figsize=(8,5))
plt.bar(models, mse_values)

plt.title("Model Performance Comparison")
plt.ylabel("Mean Squared Error(lakhs)")
plt.xlabel("Regression Models")

# Save graph
plt.savefig("graph/model_comparison.png")

# Display graph
plt.show()