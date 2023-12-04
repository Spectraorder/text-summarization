import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from text_rank.dataset import read_data
from text_rank.text_rank import TextRank
import tqdm

rouge = Rouge()
def get_rouge(label, pred, type='rouge-1'):
    if type not in ['rouge-1', 'rouge-2', 'rouge-l']:
        type = 'rouge-1'
    return rouge.get_scores(pred, label)[0][type]['r']

def get_blue(label, pred, weights=(1, 0, 0, 0)):
    return sentence_bleu([label], pred, weights)

def get_f_measure(rouge, blue):
    if rouge == 0 or blue == 0:
        return 0
    return 2.0 / (1.0 / rouge + 1.0 / blue)

def get_evaluation(path, numOfSentences, numOfKeywords, use_langauge_model=False, language_model=None):
    data = read_data(path)
    label = read_data(path, highlight=True)
    rouge_score = []
    blue_score = []
    f1_score = []
    for i in tqdm.tqdm(range(len(data))):
        text = str(data[i])
        if use_langauge_model:
            model_summary = language_model.generate_summarisation(text)
        T = TextRank(model_summary if use_langauge_model else text, pr_config={'alpha': 0.85, 'max_iter': 100})
        keywords = [word[0] for word in T.get_n_keywords(numOfKeywords)]
        sentences = [sen[0] for sen in T.get_n_sentences(20)]
        summary = [sen for sen in sentences if any(word in sen for word in keywords)][:numOfSentences]
        summary = ' '.join(summary)
        target = str(label[i])
        rouge = get_rouge(target, summary)
        blue = get_blue(target, summary)
        f1 = get_f_measure(rouge, blue)
        rouge_score.append(rouge)
        blue_score.append(blue)
        f1_score.append(f1)
    print('Sentence Extraction Evaluation: ' + str(numOfKeywords) + ' keywords ' + str(numOfSentences) + ' sentences')
    print('Average Rouge: ' + str(np.mean(rouge_score)))
    print('Average BLEU: ' + str(np.mean(blue_score)))
    print('Average F1: ' + str(np.mean(f1_score)))