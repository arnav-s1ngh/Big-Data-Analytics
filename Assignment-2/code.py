import json
import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
logging.info("Loading data files...")

with open('ids.txt', 'r') as f:
    ids = [line.strip() for line in f.readlines()]

with open('texts.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f.readlines()]

with open('items.json', 'r') as f:
    ground_truth = json.load(f)

# Feature Extraction
logging.info("Extracting features using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(texts)

# Dimensionality Reduction
logging.info("Reducing dimensionality with Truncated SVD...")
svd = TruncatedSVD(n_components=300, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# Nearest Neighbors Model
logging.info("Using Nearest Neighbors with Cosine distance...")
nbrs = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute').fit(tfidf_reduced)
distances, indices = nbrs.kneighbors(tfidf_reduced)

# Generate predictions
logging.info("Generating predictions...")
predictions = {}
for idx, neighbors in enumerate(indices):
    predictions[ids[idx]] = [ids[neighbor_idx] for neighbor_idx in neighbors[1:6]]

# Evaluate performance
logging.info("Evaluating performance...")
intersection_scores = []
for sample_id, true_neighbors in ground_truth.items():
    predicted_neighbors = predictions.get(sample_id, [])
    intersection = len(set(true_neighbors) & set(predicted_neighbors))
    intersection_scores.append(intersection)

# Statistics and visualization
logging.info("Calculating and displaying statistics...")
intersection_scores_series = pd.Series(intersection_scores)
print(intersection_scores_series.describe())

# Histogram and Boxplot
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

# Predict function for test data
def predict_top_5_for_test(test_ids_file, test_texts_file, output_file):
    logging.info("Loading test data and generating predictions...")
    with open(test_ids_file, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]

    with open(test_texts_file, 'r', encoding='utf-8') as f:
        test_texts = [line.strip() for line in f.readlines()]

    # Transform test data
    test_tfidf = vectorizer.transform(test_texts)
    test_reduced = svd.transform(test_tfidf)

    # Neighbors for test data
    test_distances, test_indices = nbrs.kneighbors(test_reduced)

    # Generate top 5 similar items (excluding self)
    test_predictions = {}
    for idx, neighbors in enumerate(test_indices):
        test_predictions[test_ids[idx]] = [ids[neighbor_idx] for neighbor_idx in neighbors[1:6]]

    # Save predictions
    with open(output_file, 'w') as f:
        json.dump(test_predictions, f)

# Example of using the test function
# predict_top_5_for_test('test_ids.txt', 'test_texts.txt', 'test_predictions.json')

