import os
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