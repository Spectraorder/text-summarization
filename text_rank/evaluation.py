from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

rouge = Rouge()
def get_rouge(label, pred, type='rouge-1'):
    if type not in ['rouge-1', 'rouge-2', 'rouge-l']:
        type = 'rouge-1'
    return rouge.get_scores(pred, label)[0][type]['r']

def get_blue(label, pred, weights=(1, 0, 0, 0)):
    return sentence_bleu(pred, label, weights)

def get_f_measure(rouge, blue):
    return 1.0 / (1.0 / rouge + 1.0 / blue)