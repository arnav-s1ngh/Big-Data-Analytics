import pandas as pd
import json
from datasketch import MinHash,MinHashLSH
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

with open("ids.txt","r") as f:
    ids=[line.strip() for line in f]
with open("texts.txt","r",encoding="utf-8") as f:
    texts=[line.strip() for line in f]
with open("items.json","r") as f:
    ground_truth=json.load(f)

vectorizer=TfidfVectorizer(max_features=5000)
tfidf_matrix=vectorizer.fit_transform(texts)
#to ensure that the matrix is not sparse
svd=TruncatedSVD(n_components=500)
reduced_matrix=svd.fit_transform(tfidf_matrix)
lsh=MinHashLSH(threshold=0.4,num_perm=256)
minhashes={}

for id_curr,vec in enumerate(reduced_matrix):
    mhash=MinHash(num_perm=256)
    for val in vec:
        mhash.update(str(val).encode('utf8'))
    minhashes[ids[id_curr]]=mhash
    lsh.insert(ids[id_curr],mhash)

predictions={}
for id_curr,id_actual in enumerate(ids):
    mhash=minhashes[id_actual]
    candidates=lsh.query(mhash)
    if candidates is not None and len(candidates)>0:
        candidate_similarities=[]
        for candidate_id in candidates:
            if candidate_id!=id_actual:
                candidate_m=minhashes[candidate_id]
                jaccard_similarity=mhash.jaccard(candidate_m)
                candidate_similarities.append((candidate_id,jaccard_similarity))
        toplist=sorted(candidate_similarities,key=lambda x:x[1])
        toplist.reverse()
        toplist=toplist[:5]
        predictions[id_actual]=[candidate_id for candidate_id,_ in toplist]
    else:
        predictions[id_actual]=[]
scores=[]
for id_actual,predicted_ids in predictions.items():
    true_ids=set(ground_truth.get(id_actual,[]))
    predicted_ids=set(predicted_ids)
    score=len(true_ids.intersection(predicted_ids))
    scores.append(score)
score_series=pd.Series(scores)
print(score_series.describe())
plt.hist(scores,bins=5,range=(0,5))
plt.show()
plt.boxplot(scores)
plt.show()

# with open("test_ids.txt","r") as f:
#     test_ids=[line.strip() for line in f]
# with open("test_texts.txt","r",encoding="utf-8") as f:
#     test_texts=[line.strip() for line in f]
# test_tfidf_matrix=vectorizer.transform(test_texts)
# test_reduced_matrix=svd.transform(test_tfidf_matrix)
# test_predictions={}
# for id_curr,test_id in enumerate(test_ids):
#     mhash=MinHash(num_perm=256)
#     for val in test_reduced_matrix[id_curr]:
#         mhash.update(str(val).encode('utf8'))
#     candidates=lsh.query(mhash)
#     candidate_similarities=[]
#     if candidates:
#         for candidate_id in candidates:
#             candidate_m=minhashes[candidate_id]
#             jaccard_similarity=mhash.jaccard(candidate_m)
#             candidate_similarities.append((candidate_id,jaccard_similarity))
#         toplist=sorted(candidate_similarities,key=lambda x:x[1])
#         toplist.reverse()
#         toplist=toplist[:5]
#         test_predictions[test_id]=[candidate_id for candidate_id,_ in toplist]
#     else:
#         test_predictions[test_id]=[]
#print(test_predictions)
