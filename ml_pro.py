import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Load the dataset
file_path = "data/updated_purchase_data.csv"
data = pd.read_csv(file_path)

# Data Preparation
data["Purchase Date"] = pd.to_datetime(data["Purchase Date"])
data["Year"] = data["Purchase Date"].dt.year
data["Month"] = data["Purchase Date"].dt.month
data["Day"] = data["Purchase Date"].dt.day

# Aggregate spending data
category_features = data.groupby("Category").agg(
    total_spending=("Price", "sum"),
    average_spending=("Price", "mean"),
    purchase_count=("Price", "count")
).reset_index()

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(category_features[["total_spending", "average_spending", "purchase_count"]])

# Compute Distance Matrix
distance_matrix = pairwise_distances(normalized_features, metric="euclidean")

# Apply MDS
mds = MDS(n_components=2, random_state=42, dissimilarity="precomputed")
mds_features = mds.fit_transform(distance_matrix)

# Add MDS Results to DataFrame
category_features["MDS1"] = mds_features[:, 0]
category_features["MDS2"] = mds_features[:, 1]

# Visualize MDS Results
plt.figure(figsize=(10, 6))
plt.scatter(category_features["MDS1"], category_features["MDS2"], c="skyblue", edgecolor="k", s=100)
for i, row in category_features.iterrows():
    plt.text(row["MDS1"] + 0.02, row["MDS2"], row["Category"], fontsize=9)
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.title("MDS Visualization of Spending Patterns")
# plt.show()

# save plot to data folder
plt.savefig("data/mds_visualization.png")

# Add Spending Insights
def generate_budgeting_tip(row):
    # Thresholds
    high_spending_threshold = 15500  # 25% of total spending
    frequent_purchase_threshold = 75

    if row["total_spending"] > high_spending_threshold:
        return "High spending; consider reducing in this category."
    elif row["purchase_count"] > frequent_purchase_threshold:
        return "Frequent purchases; look for bulk discounts."
    elif row["total_spending"] < 3100:  # 5% of total spending
        return "Spending seems reasonable in this category."
    else:
        return "Monitor spending; ensure it aligns with your budget."


category_features["Budgeting_Tip"] = category_features.apply(generate_budgeting_tip, axis=1)

# Display the updated dataset with budgeting tips
print("Spending Patterns with Budgeting Tips:")
print(category_features)

# Save the results to a CSV file
category_features.to_csv("data/mds_spending_patterns_with_tips.csv", index=False)
print("The MDS spending patterns with budgeting tips have been saved to 'data/mds_spending_patterns_with_tips.csv'.")
