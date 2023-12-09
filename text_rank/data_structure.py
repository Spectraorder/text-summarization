import nltk
from nltk.tokenize import word_tokenize
from .meta import read_static_data
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from .config import *

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
STOPWORDS, PROPERTY_FILTER = read_static_data()

def words_info(words_list):
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

class Text:
    def __init__(self, text, use_property, no_stopwords):
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
        if FILTER:
            all_tokens = [word.lower() for sent in sents for word in word_tokenize(sent) if word.isalnum()]

            stop_words = set(stopwords.words('english'))
            filtered_tokens = [word for word in all_tokens if word not in stop_words]

            freq_dist = FreqDist(filtered_tokens)
            top_keywords = [word for word, freq in freq_dist.most_common(MOST_COMMON)]
            filtered_sents = [sent for sent in sents if any(keyword in sent.lower() for keyword in top_keywords)]

            return filtered_sents
        return sents

    @staticmethod
    def _clean_words(sent):
        w_ls = [w[0].strip() for w in sent if w[1] != 'x']
        w_ls = [word for word in w_ls if len(word) > 0]
        return w_ls
