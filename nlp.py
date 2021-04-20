# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:28:00 2021

@author: laramos
"""
import nltk
nltk.download('punkt')
import pandas as pd

df = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\data_python\DiaryRecord.csv")
dutch_tokenizer = nltk.data.load('tokenizers/punkt/dutch.pickle')

txt = df.Title.iloc[2]

nltk_tokens = nltk.sent_tokenize(txt)
print (nltk_tokens)
nltk_tokens = nltk.word_tokenize(txt)
print (nltk_tokens)

import torch
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
tokens = roberta.encode('Hello world!')

robert = torch.hub.load('pytorch/fairseq',r"\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\RobBERT-base.pt")
tokens = robert.encode('Hello world!')


import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    '../',
    checkpoint_file='checkpoints/checkpoint_best.pt',
    data_name_or_path="./data"
)

roberta = RobertaModel.from_pretrained(r'\\amc.intra\users\L\laramos\home\Desktop\Postdoc eHealth\NLP',checkpoint_file="RobBERT-base.pt")

roberta.eval()

s = nltk.word_tokenize(txt)
tokens_swapped = nltk.word_tokenize(txt)

s = roberta.encode(txt)
