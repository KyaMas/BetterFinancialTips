import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
file_path = "data/updated_purchase_data.csv"
data = pd.read_csv(file_path)

# Data Preparation
# Extract features from Purchase Date
data["Purchase Date"] = pd.to_datetime(data["Purchase Date"])
data["Year"] = data["Purchase Date"].dt.year
data["Month"] = data["Purchase Date"].dt.month
data["Day"] = data["Purchase Date"].dt.day

# Encode the Category column and transform
label_encoder = LabelEncoder()
data["Category_Encoded"] = label_encoder.fit_transform(data["Category"])

# Aggregate spending data
category_features = data.groupby("Category").agg(
    total_spending=("Price", "sum"),
    average_spending=("Price", "mean"),
    purchase_count=("Price", "count")
).reset_index()

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(category_features[["total_spending", "average_spending", "purchase_count"]])

# Clustering
# Use KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
category_features["Cluster"] = kmeans.fit_predict(normalized_features)

# Interpret Clusters
# Define cluster labels based on observed patterns
cluster_descriptions = {
    0: "High-Value, Infrequent Purchases",
    1: "Moderate, Balanced Spending",
    2: "Frequent, Low-Value Purchases"
}
category_features["Cluster_Label"] = category_features["Cluster"].map(cluster_descriptions)

# Reduce dimension to fit PCA
# Use PCA to visualize clusters
pca = PCA(n_components=2)
pca_features = pca.fit_transform(normalized_features)

# Add PCA results to the dataset
category_features["PCA1"] = pca_features[:, 0]
category_features["PCA2"] = pca_features[:, 1]

# Visualize clusters
# Plot clusters with descriptive labels
plt.figure(figsize=(10, 6))
for cluster, label in cluster_descriptions.items():
    cluster_data = category_features[category_features["Cluster"] == cluster]
    plt.scatter(cluster_data["PCA1"], cluster_data["PCA2"], label=f"{label} (Cluster {cluster})")

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("Clusters of Spending Habits with Descriptions")
plt.legend()
plt.show()

# Display descriptive statistics for each cluster
cluster_summary = category_features.groupby("Cluster_Label").agg(
    total_spending=("total_spending", "mean"),
    average_spending=("average_spending", "mean"),
    purchase_count=("purchase_count", "mean")
).reset_index()

# Display spending trends
print("Cluster Spending Summary with Descriptions:")
print(cluster_summary)

# Save the cluster summary to a CSV file
cluster_summary.to_csv("data/cluster_spending_summary.csv", index=False)
print("Cluster summary has been saved to 'cluster_spending_summary.csv'.")

# Analyze Spending Trends by Category
spending_trends = category_features.copy()

# Add insights based on thresholds (might use GPT api here to generate insights)
spending_trends["Spending_Level"] = pd.cut(
    spending_trends["total_spending"],
    bins=[0, 1000, 5000, float("inf")],  # Define spending ranges
    labels=["Low Spending", "Moderate Spending", "High Spending"]
)

spending_trends["Frequency_Level"] = pd.cut(
    spending_trends["purchase_count"],
    bins=[0, 10, 50, float("inf")],  # Define frequency ranges
    labels=["Low Frequency", "Moderate Frequency", "High Frequency"]
)

# Add a column for budgeting tips
def generate_budgeting_tip(row):
    if row["Spending_Level"] == "High Spending":
        return "Consider reducing purchases in this category or finding alternatives."
    elif row["Spending_Level"] == "Moderate Spending" and row["Frequency_Level"] == "High Frequency":
        return "Look for bulk discounts or subscriptions to save money."
    elif row["Spending_Level"] == "Low Spending" and row["Frequency_Level"] == "Low Frequency":
        return "This category is well-managed; no action needed."
    else:
        return "Monitor spending to ensure it aligns with your budget."

spending_trends["Budgeting_Tip"] = spending_trends.apply(generate_budgeting_tip, axis=1)

# Display the spending trends in the console
print("Spending Trends with Budgeting Tips:")
print(spending_trends)

# Save the data to a CSV file
spending_trends.to_csv("data/spending_trends_with_budgeting_tips.csv", index=False)
print("The spending trends with budgeting tips have been saved to 'spending_trends_with_budgeting_tips.csv'.")

# Visualize trends
# Bar chart for total spending by category
plt.figure(figsize=(12, 6))
plt.bar(spending_trends["Category"], spending_trends["total_spending"], color="skyblue")
plt.xlabel("Category")
plt.ylabel("Total Spending ($)")
plt.title("Total Spending by Category")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Pie chart for spending levels
spending_levels = spending_trends["Spending_Level"].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(spending_levels, labels=spending_levels.index, autopct="%1.1f%%", startangle=140, colors=["lightcoral", "gold", "lightgreen"])
plt.title("Distribution of Spending Levels Across Categories")
plt.show()
