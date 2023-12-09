from text_rank.evaluation import *
from text_rank.langauge_model import language_model
from text_rank.config import *

# fine tune the number of keywords
if __name__ == "__main__":
    data = read_data("small_datasets/CNNML_tiny.csv")
    # T = TextRank(str(data[1]), pr_config={'alpha': 0.85, 'max_iter': 100})
    # print(T.get_n_sentences(3))
    # print(T.get_n_keywords(10))
    model = language_model()

    # reference_translation = "the cat is on the mat"
    # candidate_translation = "the cat sat on the mat"
    # rouge = get_rouge(reference_translation, candidate_translation)
    # blue = get_blue(reference_translation, candidate_translation)
    #
    # print("Rouge Score:", rouge)
    # print("Blue Score:", blue)
    #
    # f1 = get_f_measure(rouge, blue)
    # print("F1 Measure:", f1)

    for j in range(3, 5):
        for k in range(4, 7):
            get_evaluation("./tiny_CNN_DM/test_dataset.csv", j, k, language_model=model)
