import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Step 1: Load data
with open('ids.txt', 'r') as f:
    ids = [line.strip() for line in f.readlines()]

with open('texts.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f.readlines()]

with open('items.json', 'r') as f:
    ground_truth = json.load(f)

# Step 2: Preprocessing and feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# Step 3: Dimensionality Reduction using Truncated SVD (Latent Semantic Analysis)
svd = TruncatedSVD(n_components=100)  # Reduce to 100 components (adjust as needed)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# Step 4: Find Neighbors using Nearest Neighbors with Cosine Distance
nbrs = NearestNeighbors(n_neighbors=6, metric='cosine').fit(tfidf_reduced)  # 6 because we need to exclude self
distances, indices = nbrs.kneighbors(tfidf_reduced)

# Step 5: Generate top 5 similar items (excluding the item itself)
predictions = {}
for idx, neighbors in enumerate(indices):
    # Exclude the first neighbor since it's the item itself
    predictions[ids[idx]] = [ids[neighbor_idx] for neighbor_idx in neighbors[1:6]]

# Step 6: Evaluate performance using Intersection Score
intersection_scores = []
for sample_id, true_neighbors in ground_truth.items():
    predicted_neighbors = predictions.get(sample_id, [])
    intersection = len(set(true_neighbors) & set(predicted_neighbors))
    intersection_scores.append(intersection)

# Step 7: Calculate statistics and visualize results
intersection_scores_series = pd.Series(intersection_scores)
print(intersection_scores_series.describe())

# Step 8: Plot histogram and boxplot
plt.figure(figsize=(14, 6))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(intersection_scores, bins=6, edgecolor='black')
plt.title('Histogram of Intersection Scores')
plt.xlabel('Intersection Score')
plt.ylabel('Frequency')

# Box plot
plt.subplot(1, 2, 2)
plt.boxplot(intersection_scores)
plt.title('Boxplot of Intersection Scores')
plt.ylabel('Intersection Score')

plt.tight_layout()
plt.show()

# Step 9: Save predictions for test data
def predict_top_5_for_test(test_ids_file, test_texts_file, output_file):
    # Load test data
    with open(test_ids_file, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]

    with open(test_texts_file, 'r', encoding='utf-8') as f:
        test_texts = [line.strip() for line in f.readlines()]

    # Transform test data using the same vectorizer and SVD model
    test_tfidf = vectorizer.transform(test_texts)
    test_reduced = svd.transform(test_tfidf)

    # Find neighbors for test data
    test_distances, test_indices = nbrs.kneighbors(test_reduced)

    # Generate top 5 similar items (excluding the item itself)
    test_predictions = {}
    for idx, neighbors in enumerate(test_indices):
        test_predictions[test_ids[idx]] = [ids[neighbor_idx] for neighbor_idx in neighbors[1:6]]

    # Save results
    with open(output_file, 'w') as f:
        json.dump(test_predictions, f)

# Example of using the test function
# predict_top_5_for_test('test_ids.txt', 'test_texts.txt', 'test_predictions.json')
