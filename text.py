from text_rank.text_rank import *

data = read_data("small_datasets/CNNML_tiny.csv")
T = TextRank(str(data[1]), pr_config={'alpha': 0.85, 'max_iter': 100})
print(T.get_n_sentences(3))
print(T.get_n_keywords(10))
