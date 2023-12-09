from transformers import BartTokenizerFast, BartForConditionalGeneration
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from text_rank.evaluation import *
import torch
from .config import *

class language_model:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prefix = "summarize: "
        self.weight_path = M_WEIGHT
        self.model_name = M_NAME
        self.tokenizer = BartTokenizerFast.from_pretrained(M_NAME)
        self.model = BartForConditionalGeneration.from_pretrained(self.weight_path)
        self.model.to(self.device)

    def generate_summarisation(self, text: str, is_before: bool):
        prompt_text = self.prefix + str(text)
        inputs = self.tokenizer([prompt_text], max_length=1024, return_tensors='pt', truncation=True)
        inputs.to(self.device)
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, min_length=MIN_MODEL_GEN_B if is_before else MIN_MODEL_GEN_A,
                                          max_length=MAX_MODEL_GEN_B if is_before else MAX_MODEL_GEN_A,
                                     early_stopping=LM_ET_B if is_before else LM_ET_A)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

