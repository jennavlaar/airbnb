import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gzip
import shutil

compressed_file_path = "data/listings.csv.gz"
extracted_file_path = "data/newzealandlistings.csv"

if not os.path.exists(extracted_file_path):
    print("Extracting compressed file...")
    os.makedirs("data", exist_ok=True)  # Ensure the data directory exists
    with gzip.open(compressed_file_path, 'rb') as f_in:
        with open(extracted_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"File extracted to {extracted_file_path}")

# Load dataset
print("Loading dataset...")
df = pd.read_csv(extracted_file_path)

print("Dataset Overview:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop columns with less than 10% non-null values
threshold = 0.1 * len(df)
df = df.dropna(thresh=threshold, axis=1)

# Standardize column names
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Clean and convert 'price' column
if 'price' in df.columns:
    df['price'] = df['price'].str.replace('[\$,]', '', regex=True).astype(float)

# Fill missing values
# Example: Fill categorical data with "unknown"
if 'host_response_time' in df.columns:
    df['host_response_time'].fillna('unknown', inplace=True)

# Fill numerical data with median
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Drop rows with missing critical data
if 'price' in df.columns and 'property_type' in df.columns:
    df = df.dropna(subset=['price', 'property_type'])

# Example: Convert date columns to datetime (if applicable)
date_columns = [col for col in df.columns if "date" in col]
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Create output directories if they don't exist
os.makedirs("output/visualizations", exist_ok=True)

# Save cleaned data
cleaned_file_path = "output/cleaned_data.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}")

sns.set_theme(style="whitegrid")

# Price Distribution Histogram
if 'price' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="price", kde=True, color="blue")
    plt.title("Price Distribution")
    plt.savefig("output/visualizations/price_distribution.png")
    plt.close()

# Listings by Region Bar Chart
if 'region_name' in df.columns:
    region_counts = df['region_name'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=region_counts.index, y=region_counts.values, palette="viridis")
    plt.title("Listings by Region")
    plt.xticks(rotation=45)
    plt.savefig("output/visualizations/listings_by_region.png")
    plt.close()

print("Visualizations saved to output/visualizations/")

# Basic Analysis Example
if 'price' in df.columns:
    print("\nPrice Analysis:")
    print(f"Average Price: {df['price'].mean():.2f}")
    print(f"Median Price: {df['price'].median():.2f}")
