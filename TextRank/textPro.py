# import jieba.posseg as pseg
from nltk.tokenize import word_tokenize
import nltk
import re
from .const import STOPWORDS, PROPERTY_FILTER
import string
from nltk.tokenize import WordPunctTokenizer

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
SENTENCE_SP_PATTERN = re.compile(r"\.|！|\!|？|\?|;")
punc = string.punctuation
nltk.download('averaged_perceptron_tagger')

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
        # sents = [i.strip() for i in SENTENCE_SP_PATTERN.split(text) if i != '' ]
        return sents

    @staticmethod
    def _clean_words(sent):
        w_ls = [w[0].strip() for w in sent if w[1] != 'x']
        w_ls = [word for word in w_ls if len(word) > 0]
        return w_ls

