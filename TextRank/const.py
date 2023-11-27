import os

module_path = os.path.dirname(__file__)

STOPWORDS = list()
with open(module_path+'/statics/stopwords.txt', 'r', encoding='utf-8') as f:
    for w in f.readlines():
        STOPWORDS.append(w.strip())

PROPERTY_FILTER = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']



