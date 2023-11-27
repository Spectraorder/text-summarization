from TextRank import textRank
import pandas as pd

class data_stream:
    def __init__(self, article: str, highlight: str, id: str) -> None:
        self.article = article
        self.highlight = highlight
        self.id = id

    def __str__(self) -> str:
        return self.article


def read_data(path: str) -> list:
    with open(path, 'r') as f:
        df = pd.read_csv(path)
    return [data_stream(*row) for row in df.itertuples(index=False)]


data = read_data("CNNML_tiny.csv")
T = textRank.TextRank(str(data[1]), pr_config={'alpha': 0.85, 'max_iter': 100})
print(T.get_n_sentences(3))
print(T.get_n_keywords(10))
