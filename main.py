from text_rank.evaluation import *
from text_rank.langauge_model import language_model

# fine tune the number of keywords
if __name__ == "__main__":
    data = read_data("small_datasets/CNNML_tiny.csv")
    T = TextRank(str(data[1]), pr_config={'alpha': 0.85, 'max_iter': 100})
    print(T.get_n_sentences(3))
    print(T.get_n_keywords(10))
    model = language_model("lidiya/bart-base-samsum", "./bart_cnn_dailymail_finetuned")

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

    for k in range(3, 8):
        for j in range(1, 4):
            get_evaluation("./tiny_CNN_DM/test_dataset.csv", j, k, use_langauge_model=True, language_model=model)
