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

# def eng2tr(input_text):
#     trg = "tr"  # source language
#     src = "en"  # target language
#     model_name_eng2tr = f"Helsinki-NLP/opus-mt-{src}-{trg}"
#     model_eng2tr = MarianMTModel.from_pretrained(model_name_eng2tr)
#     tokenizer_eng2tr = AutoTokenizer.from_pretrained(model_name_eng2tr)
#     batch = tokenizer_eng2tr([input_text], return_tensors="pt")
#     generated_ids = model_eng2tr.generate(**batch)
#     turkish_text = tokenizer_eng2tr.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     print(turkish_text)
#     return turkish_text    