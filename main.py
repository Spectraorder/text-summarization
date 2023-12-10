from text_rank.evaluation import *
from text_rank.langauge_model import language_model
from text_rank.config import *
import csv

# fine tune the number of keywords
if __name__ == "__main__":
    data = read_data("small_datasets/CNNML_tiny.csv")
    # T = TextRank(str(data[1]), pr_config={'alpha': 0.85, 'max_iter': 100})
    model = language_model()
    results_list = []

    for j in range(1, 6):
        row_results = []
        for k in range(1, 6):
            # get_evaluation("./tiny_CNN_DM/test_dataset.csv", j, k, language_model=model)
            for w in range(2, 6):
                WINDOW_SIZE = w
                rouge, bleu, f1 = get_evaluation("./tiny_CNN_DM/test_dataset.csv", j, k, model)
                rouge = str(round(rouge, 4))
                bleu = str(round(bleu, 4))
                f1 = str(round(f1, 4))
                row_results.append("rouge: " + rouge + "\nbleu: " + bleu + "\nf1: " + f1)
            results_list.append(row_results)

    with open("output.csv", mode='w', newline='') as csv_file:
        fieldnames = ['Rouge', 'Bleu', 'F1']
        writer = csv.writer(csv_file)

        writer.writerow([''] + [f'{k} keywords' for k in range(3, 8)])

        for i, row in enumerate(results_list, start=1):
            writer.writerow([f'{i} sentences'] + [result for result in row])
