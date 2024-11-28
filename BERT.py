from transformers import pipeline
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the dataset
file_path = "uploads/raw_data.csv"
data = pd.read_csv(file_path)

# Define predefined categories
categories = ["Electronics", "Clothing", "Household", "Entertainment", "Necessities", "Sports"]

# Initialize Hugging Face's zero-shot classification pipeline with DistilBERT
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

# Function to classify product names
def classify_product(product_name):
    result = classifier(
        product_name,
        candidate_labels=categories
    )
    # Return the highest scoring category
    return result["labels"][0]

# Apply classification to the "Product Name" column
data["Category"] = data["Product Name"].apply(classify_product)

# Save the categorized data
output_file_path = "data/updated_purchase_data.csv"
data.to_csv(output_file_path, index=False)

# Print the first few rows of the updated dataset
print(data.head())

# Path to the output file
print(f"Classified data saved to: {output_file_path}")

# Generate a wordcloud of the review text
text = " ".join(review for review in data['Product Name'])
wordcloud = WordCloud(background_color="white").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('./static/images/review_wordcloud.png')