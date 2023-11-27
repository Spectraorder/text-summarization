import os
from nltk.tokenize import word_tokenize
import nltk
import re
import string
import numpy as np
import math
import pandas as pd

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
punc = string.punctuation
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class data_stream:
    def __init__(self, article: str, highlight: str, id: str) -> None:
        self.article = article
        self.highlight = highlight
        self.id = id

    def __str__(self) -> str:
        return self.article


def read_data(path: str) -> list:
    with open(path, 'r') as f:
        df = pd.read_csv(path)
    return [data_stream(*row) for row in df.itertuples(index=False)]

def words_info(words_list):
    """
    :param words_list:
    :return:
    """
    word_index = dict()
    index_word = dict()
    word_num = 0

    for index_s, words in enumerate(words_list):
        for index_w, word in enumerate(words):
            word_index[word] = word_num
            index_word[word_num] = word
            word_num += 1

    return word_index, index_word, word_num


def sents_info(sents_list):

    return dict(zip(range(len(sents_list)), sents_list))


def word_adj_matrix(words_pro, windows, word_num, word_index):
    """
    Adjacency Matrix
    :param windows:
    :return:
    """
    def _word_combine(words, window):
        """
        Keyword arguments:
        :param window:
        """
        if window < 2: window = 2

        for x in range(1, window):
            if x >= len(words):
                break
            words2 = words[x:]
            res = zip(words, words2)
            for r in res:
                yield r

    matrix = np.zeros((word_num, word_num))

    for words in words_pro:
        for w1, w2 in _word_combine(words, windows):
            if w1 in word_index and w2 in word_index:
                index1 = word_index.get(w1)
                index2 = word_index.get(w2)
                matrix[index1][index2] = 1.0
                matrix[index2][index1] = 1.0

    return matrix


def sent_adj_matrix(words_pro):

    def _get_similarity(word_ls1, word_ls2):
        co_occur_num = len(set(word_ls1) & set(word_ls2))
        if abs(co_occur_num) <= 1e-12:
            return 0.

        if len(word_ls1) == 0 or len(word_ls2) == 0:
            return 0.

        denominator = math.log(float(len(word_ls1))) + math.log(float(len(word_ls2)))

        if abs(denominator) < 1e-12:
            return 0.

        return co_occur_num / denominator

    sentences_num = len(words_pro)
    matrix = np.zeros((sentences_num, sentences_num))

    for x in range(sentences_num):
        for y in range(x, sentences_num):
            s = _get_similarity(words_pro[x], words_pro[y])
            matrix[x, y] = s
            matrix[y, x] = s
    return matrix


def cal_score(ad_matrix, alpha=0.85, max_iter=100):

    N = len(ad_matrix)
    ad_sum = ad_matrix.sum(axis=0).astype(float)
    ad_sum[ad_sum == 0.0] = 0.001
    ad_matrix = ad_matrix / ad_sum
    pr = np.full([N, 1], 1 / N)

    for _ in range(max_iter):
        pr = np.dot(ad_matrix, pr) * alpha + (1 - alpha)
    pr = pr / pr.sum()

    scores = dict(zip(range(len(pr)), [i[0] for i in pr]))

    return scores


def get_sorted_items(scores, index_items):
    """
    :param scores:
    :param index_items:
    :return: list[tuple]
    """
    items_scores = dict()
    for index, score in scores.items():
        items_scores[index_items.get(index)] = score
    sorted_items = sorted(items_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_items



def read_static_data():
    module_path = os.path.dirname(__file__)
    STOPWORDS = list()
    with open(module_path+'/statics/stopwords.txt', 'r', encoding='utf-8') as f:
        for w in f.readlines():
            STOPWORDS.append(w.strip())

    PROPERTY_FILTER = list()
    with open(module_path+'/statics/pos.txt', 'r', encoding='utf-8') as f:
        for p in f.readlines():
            PROPERTY_FILTER.append(p.strip())

    return STOPWORDS, PROPERTY_FILTER

STOPWORDS, PROPERTY_FILTER = read_static_data()

class Text:
    def __init__(self, text, use_property, no_stopwords):
        """

        :param text:
        :param use_property: 是否根据词性进行筛选
        :param no_stopwords: 是否去停用词
        """

        if not isinstance(text, str):
            raise ValueError('text type must be str!')
        elif text is None:
            raise ValueError('text should not be none!')

        self.sents = self._sentence_split(text)
        self.words_pro = self._get_words(self.sents, use_property, no_stopwords)


    def _get_words(self, sents, use_property, no_stopwords):

        words = list()

        if len(sents) < 1:
            return None

        for s in sents:
            cut_s = word_tokenize(s)
            cut_s = nltk.pos_tag(cut_s)
            if use_property:
                cut_s = [w for w in cut_s if w[1] in PROPERTY_FILTER]
            else:
                cut_s = [w for w in cut_s]

            cut_s = self._clean_words(cut_s)
            if no_stopwords:
                cut_s = [w.strip() for w in cut_s if w.strip() not in STOPWORDS]
            words.append(cut_s)

        return words

    @staticmethod
    def _sentence_split(text):
        sents = sent_tokenizer.tokenize(text)
        return sents

    @staticmethod
    def _clean_words(sent):
        w_ls = [w[0].strip() for w in sent if w[1] != 'x']
        w_ls = [word for word in w_ls if len(word) > 0]
        return w_ls


class TextRank(Text):
    def __init__(self,
                 text=None, windows=2,
                 use_property=True, no_stopwords=True,
                 pr_config={'alpha': 0.85, 'max_iter': 100}):
        super(TextRank, self).__init__(text, use_property, no_stopwords)
        print(text)
        self.pr_config = pr_config
        self.windows = windows
        self.word_index, self.index_word, self.word_num = words_info(self.words_pro)
        self.sorted_words = self._score_items(is_word=True)
        self.sorted_sents = self._score_items(is_word=False)

    def _build_adjacency_matrix(self, is_word=True):

        if is_word:
            adj_matrix = word_adj_matrix(self.words_pro,
                                               self.windows,
                                               self.word_num,
                                               self.word_index)
        else:
            adj_matrix = sent_adj_matrix(self.words_pro)

        return adj_matrix

    def _score_items(self, is_word=True):
        if is_word:
            adj_matrix = self._build_adjacency_matrix(is_word=is_word)
            scores = cal_score(adj_matrix, **self.pr_config)
            sorted_items = get_sorted_items(scores, self.index_word)
        else:
            adj_matrix = self._build_adjacency_matrix(is_word=is_word)
            scores = cal_score(adj_matrix, **self.pr_config)
            index_sent = dict(zip(range(len(self.sents)), self.sents))
            sorted_items = get_sorted_items(scores, index_sent)

        return sorted_items

    def get_n_keywords(self, N):
        return self.sorted_words[:N]

    def get_n_sentences(self, N):
        return self.sorted_sents[:N]