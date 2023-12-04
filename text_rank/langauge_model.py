from transformers import BartTokenizerFast, BartForConditionalGeneration
from text_rank.evaluation import *
import torch

class language_model:
    def __init__(self, model_name: str, weight_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prefix = "summarize: "
        self.weight_path = weight_path
        self.model_name = model_name
        self.tokenizer = BartTokenizerFast.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.weight_path)
        self.model.to(self.device)

    def generate_summarisation(self, text: str):
        prompt_text = self.prefix + str(text)
        inputs = self.tokenizer([prompt_text], max_length=1024, return_tensors='pt', truncation=True)
        inputs.to(self.device)
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, min_length=100, max_length=200,
                                     early_stopping=False)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

