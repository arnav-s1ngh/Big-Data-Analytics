import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from datasketch import MinHash, MinHashLSH
import matplotlib.pyplot as plt

# Step 1: Load data
with open('ids.txt', 'r') as f:
    ids = [line.strip() for line in f.readlines()]

with open('texts.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f.readlines()]

with open('items.json', 'r') as f:
    ground_truth = json.load(f)

# Step 2: Preprocessing and feature extraction
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# Step 3: LSH Setup
lsh = MinHashLSH(threshold=0.5, num_perm=128)
minhashes = {}


def create_minhash(text):
    minhash = MinHash(num_perm=128)
    for word in text.split():
        minhash.update(word.encode('utf8'))
    return minhash


# Populate LSH
for idx, text in enumerate(texts):
    minhash = create_minhash(text)
    lsh.insert(ids[idx], minhash)
    minhashes[ids[idx]] = minhash

# Step 4: Find Neighbors and predict
predictions = {}

for idx, text in enumerate(texts):
    minhash = minhashes[ids[idx]]
    result = lsh.query(minhash)
    result = [r for r in result if r != ids[idx]]  # remove self from neighbors
    predictions[ids[idx]] = result[:5]  # top 5 predictions

# Step 5: Evaluate performance
intersection_scores = []

for sample_id, true_neighbors in ground_truth.items():
    predicted_neighbors = predictions.get(sample_id, [])
    intersection = len(set(true_neighbors) & set(predicted_neighbors))
    intersection_scores.append(intersection)

# Step 6: Calculate statistics
intersection_scores_series = pd.Series(intersection_scores)
print(intersection_scores_series.describe())

# Step 7: Plot histogram and boxplot
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


# # Step 8: Save predictions for test data
# def predict_top_5_for_test(test_ids_file, test_texts_file, output_file):
#     # Load test data
#     with open(test_ids_file, 'r') as f:
#         test_ids = [line.strip() for line in f.readlines()]
#
#     with open(test_texts_file, 'r', encoding='utf-8') as f:
#         test_texts = [line.strip() for line in f.readlines()]
#
#     test_predictions = {}
#     for idx, text in enumerate(test_texts):
#         minhash = create_minhash(text)
#         result = lsh.query(minhash)
#         result = [r for r in result if r != test_ids[idx]]
#         test_predictions[test_ids[idx]] = result[:5]
#
#     # Save results
#     with open(output_file, 'w') as f:
#         json.dump(test_predictions, f)
#
#
# # Example of using the test function
# predict_top_5_for_test('test_ids.txt', 'test_texts.txt', 'test_predictions.json')
