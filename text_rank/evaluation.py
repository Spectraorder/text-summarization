import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from text_rank.dataset import read_data
from text_rank.text_rank import TextRank
import tqdm
from .config import *

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


def get_evaluation(path, numOfSentences, numOfKeywords, window_size, language_model=None):
    with open(LOG_NAME, "a", encoding="utf-8") as f:
        data = read_data(path)
        label = read_data(path, highlight=True)
        rouge_score = []
        blue_score = []
        f1_score = []
        for i in tqdm.tqdm(range(len(data))):
            text = str(data[i])
            if USE_LANGUAGE_MODEL_BEFORE_EXT:
                summary = language_model.generate_summarisation(text, True)
            if USE_TEXTRANK:
                T = TextRank(summary if USE_LANGUAGE_MODEL_BEFORE_EXT else text,
                             pr_config={'alpha': 0.85, 'max_iter': 100}, windows=window_size)
                keywords = [word[0] for word in T.get_n_keywords(numOfKeywords)]
                sentences = [sen[0] for sen in T.get_n_sentences(20)]
                summary = [sen for sen in sentences if any(word in sen for word in keywords)][:numOfSentences]
                summary = ' '.join(summary)
            if USE_LANGUAGE_MODEL_AFTER_EXT:
                summary = language_model.generate_summarisation(summary, False)
            target = str(label[i])
            rouge = get_rouge(target, summary)
            blue = get_blue(target, summary)
            f1 = get_f_measure(rouge, blue)
            rouge_score.append(rouge)
            blue_score.append(blue)
            f1_score.append(f1)
        print(
            'Sentence Extraction Evaluation: ' + str(numOfKeywords) + ' keywords ' + str(numOfSentences) + ' sentences')
        print('Average Rouge: ' + str(np.mean(rouge_score)))
        print('Average BLEU: ' + str(np.mean(blue_score)))
        print('Average F1: ' + str(np.mean(f1_score)))
        f.write(f"sentences,keywords,window,LM_before,LM_after,filter,textrank,alpha,max_iter,BLEU,ROUGE,F1\n")
        f.write(
            f"{numOfSentences},{numOfKeywords},{window_size}，{USE_LANGUAGE_MODEL_BEFORE_EXT},{USE_LANGUAGE_MODEL_AFTER_EXT},{FILTER},{USE_TEXTRANK},{ALPHA},{MAX_ITER},{str(np.mean(rouge_score))},{str(np.mean(blue_score))},{str(np.mean(f1_score))}\n")
        return np.mean(rouge_score), np.mean(blue_score), np.mean(f1_score)
