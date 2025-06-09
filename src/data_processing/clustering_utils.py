import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # Import PCA
import logging
from tqdm import tqdm  # For progress bar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
logging.info("Loading the dataset...")
data_path = "dataset/processed/IPIP_NEO_120_processed.csv"
df_original = pd.read_csv(data_path)

# Features
logging.info("Selecting relevant features...")
features = [f"i{i}" for i in range(1, 121)]
features.append("sex")
features.append("age")
df_features = df_original[features].copy()

# Step 1: Preprocess the data
logging.info("Scaling the data...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

# Step 2: Apply clustering techniques
with tqdm(total=2, desc="Clustering Progress", unit="step") as pbar:
    logging.info("Applying K-Means clustering...")
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_data)
    logging.info("K-Means clustering completed.")
    pbar.update(1)

    # Step 3: Reduce dimensions using PCA
    logging.info("Reducing dimensions using PCA...")
    reducer = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    pca_embeddings = reducer.fit_transform(scaled_data)
    logging.info("PCA dimensionality reduction completed.")
    pbar.update(1)

# Attach kmeans labels and PCA components to df_original
logging.info("Attaching KMeans labels and PCA components to the original dataframe...")
df_original['Cluster'] = kmeans_labels
df_original['PCA1'] = pca_embeddings[:, 0]
df_original['PCA2'] = pca_embeddings[:, 1]

# Step 4: Plot the clusters
def plot_clusters(embeddings, labels, title):
    logging.info(f"Plotting clusters: {title}...")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='Spectral', s=10)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title(title)
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.show()
    logging.info(f"Plotting completed: {title}.")

# Plot K-Means clusters
#plot_clusters(pca_embeddings, kmeans_labels, "K-Means Clusters")

# Sample from each cluster to create train and test sets
logging.info("Sampling from each cluster to create train and test sets...")
train_set = []
test_set = []

for cluster_label in np.unique(kmeans_labels):
    cluster_indices = np.where(kmeans_labels == cluster_label)[0]
    cluster_cases = df_original.iloc[cluster_indices]

    # Ensure there are enough points to sample
    if len(cluster_cases) >= 1250:
        train_sample = cluster_cases.sample(n=1250, random_state=42)
        test_sample = cluster_cases.drop(train_sample.index).sample(n=1250, random_state=42)

        train_set.append(train_sample)
        test_set.append(test_sample)
    else:
        logging.warning(f"Cluster {cluster_label} has less than 50 points. Skipping...")

# Combine train and test sets
train_set = pd.concat(train_set)
test_set = pd.concat(test_set)

# Export train and test sets
logging.info("Exporting train and test sets...")
train_set.to_csv("dataset/train_set.csv", index=False)
test_set.to_csv("dataset/test_set.csv", index=False)
logging.info("Export completed.")