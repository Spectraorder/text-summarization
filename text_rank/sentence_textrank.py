# import numpy as np
# import networkx as nx
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# import pandas as pd
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# nltk.download('punkt')
# nltk.download('stopwords')

# class data_stream:
#     def __init__(self, article: str, highlight: str, id: str) -> None:
#         self.article = article
#         self.highlight = highlight
#         self.id = id
        
#     def __str__(self) -> str:
#         return self.article
        
# def read_data(path: str) -> list:
#     with open(path, 'r') as f:
#         df = pd.read_csv(path)
#     return [data_stream(*row) for row in df.itertuples(index=False)]

def load_glove_embeddings(glove_file):
    embeddings_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

# Example of loading GloVe embeddings
glove_file = 'glove_6B/glove.6B.100d.txt'  # Replace with your GloVe file path
word_embeddings = load_glove_embeddings(glove_file)

# def textrank_sentence_extraction(text, num_sentences=5):
#     # Tokenize the text into sentences
#     sentences = sent_tokenize(text)

#     # Preprocess sentences (tokenize, lower, remove stopwords)
#     stop_words = set(stopwords.words('english'))
#     preprocessed_sentences = [[word.lower() for word in word_tokenize(sent) if word.isalpha() and word.lower() not in stop_words] for sent in sentences]

#     # Vectorize sentences (using simple average of word embeddings)
#     # Assuming word_embeddings is a dict mapping words to their embeddings
#     sentence_vectors = [np.mean([word_embeddings.get(word, np.zeros((100,))) for word in sent], axis=0) for sent in preprocessed_sentences]

#     # Build the similarity matrix
#     similarity_matrix = np.zeros((len(sentences), len(sentences)))
#     for i in range(len(sentences)):
#         for j in range(len(sentences)):
#             if i != j:
#                 similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, -1), sentence_vectors[j].reshape(1, -1))[0,0]

#     # Apply PageRank to the similarity matrix
#     nx_graph = nx.from_numpy_array(similarity_matrix)
#     scores = nx.pagerank(nx_graph)

#     # Sort sentences by score and select the top ones for summary
#     ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
#     summary_sentences = [s for _, s in ranked_sentences[:num_sentences]]

#     return ' '.join(summary_sentences)

# # Example usage
# text = """
# LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. 
# Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, 
# the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, 
# suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. 
# "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," 
# currently six places below his number one movie on the UK box office chart. 
# Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. 
# "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. 
# Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. 
# "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" 
# is breaking records on both sides of the Atlantic and he will reprise the role in the last two films.  Watch I-Reporter give her review of Potter's latest » . 
# There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. 
# He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." 
# Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . 
# Copyright 2007 Reuters. All rights reserved.
# This material may not be published, broadcast, rewritten, or redistributed.
# """
# summary = textrank_sentence_extraction(text)
# print(summary)

def textrank_sentence_extraction(text, num_sentences=5):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)

    # Preprocess sentences
    stop_words = set(stopwords.words('english'))
    preprocessed_sentences = [[word.lower() for word in word_tokenize(sent) if word.isalpha() and word.lower() not in stop_words] for sent in sentences]

    # Vectorize sentences
    # Assuming word_embeddings is a dictionary of word embeddings
    sentence_vectors = [np.mean([word_embeddings.get(word, np.zeros((100,))) for word in sent], axis=0) for sent in preprocessed_sentences]

    # Build similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, -1), sentence_vectors[j].reshape(1, -1))[0,0]

    # PageRank algorithm
    pagerank_scores = np.array([1] * len(sentences))
    damping_factor = 0.85
    threshold = 1e-5  # convergence threshold

    for _ in range(100):
        prev_pagerank_scores = np.copy(pagerank_scores)
        for i in range(len(sentences)):
            inbound_links = similarity_matrix[:,i]
            pagerank_scores[i] = (1 - damping_factor) + damping_factor * sum(inbound_links * prev_pagerank_scores / np.sum(similarity_matrix, axis=0))
        
        if np.abs(prev_pagerank_scores - pagerank_scores).sum() < threshold:
            break

    # Extract top N sentences as summary
    top_sentence_indices = pagerank_scores.argsort()[-num_sentences:][::-1]
    summary = ' '.join([sentences[i] for i in top_sentence_indices])

    return summary

# Example usage
text = """
LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. 
Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, 
the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, 
suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. 
"The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," 
currently six places below his number one movie on the UK box office chart. 
Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. 
"Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. 
Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. 
"But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" 
is breaking records on both sides of the Atlantic and he will reprise the role in the last two films.  Watch I-Reporter give her review of Potter's latest » . 
There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. 
He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." 
Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . 
Copyright 2007 Reuters. All rights reserved.
This material may not be published, broadcast, rewritten, or redistributed.
"""
summary = textrank_sentence_extraction(text)
print(summary)
