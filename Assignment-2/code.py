import pandas as pd
import json
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from datasketch import MinHash, MinHashLSH
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Loading data...")
# Load data
with open("ids.txt", "r") as f:
    ids = [line.strip() for line in f]
with open("texts.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f]
with open("items.json", "r") as f:
    ground_truth = json.load(f)
logging.info("Data loaded successfully.")

# Vectorize text data
logging.info("Vectorizing text data with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=3000)  # Increased feature space
tfidf_matrix = vectorizer.fit_transform(texts)
logging.info("TF-IDF vectorization complete.")

# Dimensionality reduction
logging.info("Reducing dimensions with TruncatedSVD...")
svd = TruncatedSVD(n_components=500)  # Increased components for better capture
reduced_matrix = svd.fit_transform(tfidf_matrix)
logging.info("Dimensionality reduction complete.")

# Initialize LSH
logging.info("Setting up LSH with Jaccard similarity...")
lsh = MinHashLSH(threshold=0.3, num_perm=256)  # Lowered threshold, increased permutations
minhashes = {}

logging.info("Creating MinHash signatures and inserting into LSH...")
for idx, vec in enumerate(reduced_matrix):
    m = MinHash(num_perm=256)
    for val in vec:
        m.update(str(val).encode('utf8'))
    minhashes[ids[idx]] = m
    lsh.insert(ids[idx], m)
    if idx % 1000 == 0:
        logging.info(f"Processed {idx}/{len(ids)} items for MinHash insertion.")

logging.info("MinHash signatures created and inserted into LSH.")

# Retrieve top 5 similar items
logging.info("Retrieving top 5 similar items for each data sample using Jaccard similarity...")
predictions = {}
for idx, id_ in enumerate(ids):
    m = minhashes[id_]
    candidates = lsh.query(m)

    # Calculate Jaccard similarity on candidates and select top 5
    if candidates:
        candidate_similarities = []
        for candidate_id in candidates:
            if candidate_id != id_:
                candidate_m = minhashes[candidate_id]
                jaccard_similarity = m.jaccard(candidate_m)
                candidate_similarities.append((candidate_id, jaccard_similarity))

        # Sort by Jaccard similarity and take top 5
        top5 = sorted(candidate_similarities, key=lambda x: x[1], reverse=True)[:5]
        predictions[id_] = [candidate_id for candidate_id, _ in top5]
    else:
        predictions[id_] = []

    if idx % 1000 == 0:
        logging.info(f"Processed {idx}/{len(ids)} samples for similarity retrieval.")

logging.info("Similarity retrieval complete for all samples.")

# Evaluation
logging.info("Evaluating model performance...")
scores = []
for id_, predicted_ids in predictions.items():
    true_ids = set(ground_truth.get(id_, []))
    predicted_ids = set(predicted_ids)
    score = len(true_ids.intersection(predicted_ids))
    scores.append(score)

# Convert scores to pandas series for easy analysis
score_series = pd.Series(scores)
logging.info("Model evaluation complete.")
logging.info(f"Score Statistics:\n{score_series.describe()}")

# Plot Histogram
logging.info("Plotting histogram of intersection scores...")
plt.hist(scores, bins=5, range=(0, 5))
plt.xlabel("Intersection Score")
plt.ylabel("Frequency")
plt.title("Distribution of Intersection Scores")
plt.show()

# Plot Box Plot
logging.info("Plotting box plot of intersection scores...")
plt.boxplot(scores)
plt.title("Box Plot of Intersection Scores")
plt.show()

logging.info("Script execution complete.")


