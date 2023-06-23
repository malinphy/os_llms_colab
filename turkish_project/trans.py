# !pip install transformers -q
# !pip install sentencepiece -q
# !pip install sacremoses -q

import os 
import pandas as pd 
import numpy as np

from transformers import AutoTokenizer, MarianMTModel

def tr2eng(input_text):
    src = "tr"  # source language
    trg = "en"  # target language
    model_name_tr2eng = f"Helsinki-NLP/opus-mt-{src}-{trg}"
    model_tr2eng = MarianMTModel.from_pretrained(model_name_tr2eng)
    tokenizer_tr2eng = AutoTokenizer.from_pretrained(model_name_tr2eng)
    batch = tokenizer_tr2eng([input_text], return_tensors="pt")
    generated_ids = model_tr2eng.generate(**batch)
    eng_text = tokenizer_tr2eng.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(eng_text)
    return eng_text



tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-tr")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-tr")
from transformers import MarianMTModel, MarianTokenizer
def eng2tr(english_text):
    model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer(english_text, return_tensors="pt", padding=True))
    # print(translated)
    decoded = []
    for t in translated:
        decoded.append(tokenizer.decode(t, skip_special_tokens=True) )
    print(decoded)
