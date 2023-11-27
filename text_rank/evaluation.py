import math
from collections import Counter
from nltk import ngrams

def BLEU(candidate, reference, n):
    candidate_ngrams = Counter(ngrams(candidate.split(), n))
    reference_ngrams = Counter(ngrams(reference.split(), n))
    overlap = candidate_ngrams & reference_ngrams
    return sum(overlap.values()) / max(1, sum(candidate_ngrams.values()))

def calculate_bleu(candidate, reference):
    precisions = [BLEU(candidate, reference, n) for n in range(1, 5)]
    p_geom_mean = math.exp(sum(math.log(p) for p in precisions if p) / 4)
    bp = min(1, len(candidate.split()) / len(reference.split()))
    return bp * p_geom_mean

def rouge_n(evaluated_ngrams, reference_ngrams, n):
    evaluated_ngrams = Counter(evaluated_ngrams)
    reference_ngrams = Counter(reference_ngrams)
    overlap_ngrams = evaluated_ngrams & reference_ngrams
    return sum(overlap_ngrams.values()) / max(1, sum(reference_ngrams.values()))

def get_ngrams(text, n):
    words = text.split()
    return zip(*[words[i:] for i in range(n)])