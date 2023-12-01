import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist


class data_stream:
    def __init__(self, article: str, highlight: str, id: str, keywords: list = None) -> None:
        self.article = article
        self.highlight = highlight
        self.id = id
        self.keywords = keywords if keywords else extract_keywords(article)

    def __str__(self) -> str:
        return self.article


def read_data(path: str) -> list:
    # with open(path, 'r') as f:
    #     df = pd.read_csv(path)
    # return [data_stream(*row) for row in df.itertuples(index=False)]
    df = pd.read_csv(path)
    data_streams = []
    for row in df.itertuples(index=False):
        keywords = extract_keywords(row.article)
        data_streams.append(data_stream(*row, keywords=keywords))

    return data_streams


def extract_keywords(text: str) -> list:
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

    freq_dist = FreqDist(filtered_tokens)
    top_keywords = [word for word, freq in freq_dist.most_common(5)]  # Adjust 5 to the desired number of keywords

    return top_keywords
