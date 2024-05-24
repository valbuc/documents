import numpy as np


def get_keyword_similarities(keyword_embedding: dict):
    embeddings = np.array(list(keyword_embedding.values()))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    cosine_similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    keyword_pairs_with_similarity = []

    keywords = list(keyword_embedding.keys())

    for i in range(len(keywords)):
        for j in range(i+1, len(keywords)): 
            similarity = cosine_similarity_matrix[i, j]
            keyword_pairs_with_similarity.append((keywords[i], keywords[j], similarity))

    keyword_pairs_with_similarity.sort(key=lambda x: x[2], reverse=True)

    return keyword_pairs_with_similarity


def get_pairs_to_merge(pairs: list, similarity_threshold: float = 0.7):
    pairs_to_merge = []
    merged = []

    for keyword_pair in pairs:
        if (
          keyword_pair[2] > similarity_threshold
        ) and (
          keyword_pair[0] not in merged
        ) and (
          keyword_pair[1] not in merged
        ):
            pairs_to_merge.append(keyword_pair)
            merged.append(keyword_pair[0])
            merged.append(keyword_pair[1])
    return pairs_to_merge
