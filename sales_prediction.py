# 1. Import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 2. Load Dataset from internet
url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
df = pd.read_csv(url)

# Remove the extra first column
df = df.iloc[:, 1:]

print("Dataset Loaded Successfully")
print(df.head())

# 3. Create Pairplot and save it
sns.pairplot(df)
plt.suptitle("Pairplot: TV vs Radio vs Newspaper vs Sales", y=1.02)
plt.savefig("pairplot.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Split data into X and y
# Note: column names are small letters
X = df[['TV', 'radio', 'newspaper']]
y = df['sales']

# Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create and Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6. Predict on Test data
y_pred = model.predict(X_test)

# 7. Calculate Accuracy
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n================ RESULTS ================")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print("=========================================")

# 8. Plot Actual vs Predicted and save it
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.8, color='royalblue', edgecolor='black')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Sales", fontsize=12)
plt.ylabel("Predicted Sales", fontsize=12)
plt.title("Actual vs Predicted Sales", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig("result_graph.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nDone! Check left side 'Files' for pairplot.png and result_graph.png")
