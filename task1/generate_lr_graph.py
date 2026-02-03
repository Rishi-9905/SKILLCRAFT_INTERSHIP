import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'c:/Users/rishi/OneDrive/Desktop/skillcraft/train (1).csv'
df = pd.read_csv(file_path)

# Select relevant columns
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']
df_subset = df[features]

# Prepare data
X = df_subset[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df_subset['SalePrice']

# Split the data (80% train, 20% validation) - Consistent with notebook
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_val)

# Generate Graph
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2) # Diagonal line
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.grid(True)

# Save the plot
output_path = 'c:/Users/rishi/OneDrive/Desktop/skillcraft/linear_regression_graph.png'
plt.savefig(output_path)
print(f"Graph saved to {output_path}")
