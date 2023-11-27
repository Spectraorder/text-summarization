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