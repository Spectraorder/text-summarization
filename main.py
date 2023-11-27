from text_rank.text_rank import *
from text_rank.dataset import read_data
from text_rank.evaluation import *

if __name__ == "__main__":
    data = read_data("small_datasets/CNNML_tiny.csv")
    T = TextRank(str(data[1]), pr_config={'alpha': 0.85, 'max_iter': 100})
    print(T.get_n_sentences(3))
    print(T.get_n_keywords(10))

    # Example usage
    reference_translation = "the cat is on the mat"
    candidate_translation = "the cat sat on the mat"
    bleu_score = calculate_bleu(candidate_translation, reference_translation)
    print("BLEU Score:", bleu_score)

    # Example usage
    reference = "the cat is on the mat"
    candidate = "the cat sat on the mat"
    rouge_score = rouge_n(get_ngrams(candidate, 2), get_ngrams(reference, 2), 2)
    print("ROUGE-2 Score:", rouge_score)