import pandas as pd


class data_stream:
    def __init__(self, article: str, highlight: str, id: str) -> None:
        self.article = article
        self.highlight = highlight
        self.id = id

    def __str__(self) -> str:
        return self.article

    def get_highlight(self) -> str:
        return self.highlight


def read_data(path: str, highlight=False) -> list:
    # df = pd.read_csv(path)
    # data_streams = []
    # for row in df.itertuples(index=False):
    #     keywords = extract_keywords(row.article)
    #     data_streams.append(data_stream(*row, keywords=keywords))
    #
    # return data_streams
    with open(path, 'r') as f:
        df = pd.read_csv(path)
    if highlight:
        return [data.get_highlight() for data in (data_stream(*row) for row in df.itertuples(index=False))]
    else:
        return [data_stream(*row) for row in df.itertuples(index=False)]
