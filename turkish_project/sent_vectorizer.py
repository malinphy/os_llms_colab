import os 
import pandas as pd 
import numpy as np 
import re
# from langchain.vectorstores import FAISS
# import faiss

def cleaner(x):
    return re.sub('<pad>','',x)

from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('all-MiniLM-L6-v2')
def vectorizer(input_sent):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_emb = model.encode(input_sent)
    return  input_emb