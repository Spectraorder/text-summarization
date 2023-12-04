import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from config import *


class data_stream:
    def __init__(self, article: str, highlight: str, id: str, keywords: list = None, use_keywords=True) -> None:
        self.article = article
        self.highlight = highlight
        self.id = id
        if use_keywords:
            self.keywords = keywords if keywords else extract_keywords(article)

    def __str__(self) -> str:
        return self.article

    def get_highlight(self) -> str:
        return self.highlight

def extract_keywords(text: str) -> list:
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

    freq_dist = FreqDist(filtered_tokens)
    top_keywords = [word for word, freq in freq_dist.most_common(MOST_COMMON)]  # Adjust 5 to the desired number of keywords

    return top_keywords

def read_data(path: str, highlight = False) -> list:
    #df = pd.read_csv(path)
    #data_streams = []
    #for row in df.itertuples(index=False):
        #keywords = extract_keywords(row.article)
        #data_streams.append(data_stream(*row, keywords=keywords))

    #return data_streams
    with open(path, 'r') as f:
        df = pd.read_csv(path)
    if highlight:
        return [data.get_highlight() for data in (data_stream(*row) for row in df.itertuples(index=False))]
    else:
        return [data_stream(*row) for row in df.itertuples(index=False)]
