MOST_COMMON = 6
ALPHA = 0.85
FILTER = True
# FILTER_ARTICLE = True
# USE_KEYWORD = False
USE_LANGUAGE_MODEL_BEFORE_EXT = True
USE_LANGUAGE_MODEL_AFTER_EXT = False
LM_ET_B = False
LM_ET_A = True
MIN_MODEL_GEN_B = 200
MAX_MODEL_GEN_B = 300
MIN_MODEL_GEN_A = 50
MAX_MODEL_GEN_A = 70
USE_TEXTRANK = True
ALPHA = 0.85
MAX_ITER = 100
M_NAME = "lidiya/bart-base-samsum"
M_WEIGHT = "./bart_cnn_dailymail_finetuned"
LOG_NAME = "result.csv"
# M_NAME = "lidiya/bart-large-xsum-samsum"
# M_WEIGHT = "./bart_large"

# M_NAME = "Falconsai/text_summarization"
# M_WEIGHT = "./t5_finetuned"