import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

df = fetch_california_housing(as_frame=True).frame

# Plot 1: price vs income
plt.figure()
plt.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.2)
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("Price vs Income")
plt.savefig("price_vs_income.png")

# Plot 2: price vs location
plt.figure()
plt.scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"], cmap="viridis", s=5)
plt.colorbar(label="House Price")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Price by Location")
plt.savefig("price_by_location.png")

# Correlation with price
print(df.corr()["MedHouseVal"].sort_values(ascending=False))

# 80/20 split
X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

plt.show()