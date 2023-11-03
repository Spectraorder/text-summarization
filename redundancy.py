from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def detect_redundancy(sentences):
    # Vectorize the sentences using CountVectorizer
    vectorizer = CountVectorizer().fit_transform(sentences)

    # Calculate cosine similarity between sentences
    cosine_matrix = cosine_similarity(vectorizer)

    # Identify redundant sentences based on cosine similarity
    redundant_sentences = set()
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if cosine_matrix[i][j] > 0.8:  # Adjust the threshold as needed
                redundant_sentences.add(sentences[j])  # Mark sentence j as redundant

    # Remove redundant sentences from the original list of sentences
    output = [sentence for sentence in sentences if sentence not in redundant_sentences]

    return output


# Example usage
input_sentences = [
    "TextRank is an extractive text summarization algorithm.",
    "TextRank algorithm is based on graph theory.",
    "TextRank is using graph theory for its implementation",
    "Graph-based algorithms are used for ranking sentences.",
    "Redundancy detection helps in improving the quality of summaries.",
    "TextRank uses graph-based algorithms for ranking sentences.",
    "Redundancy detection is crucial for improving summary quality."
]

non_redundant_sentences = detect_redundancy(input_sentences)
print("Non-redundant Sentences:")
print(len(non_redundant_sentences))
print(non_redundant_sentences)
