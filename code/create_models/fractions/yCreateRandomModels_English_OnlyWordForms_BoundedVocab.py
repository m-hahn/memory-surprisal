# /u/scr/mhahn/deps/memory-need-neural-wordforms/search-Czech_yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab.py_model_761359559_RANDOM_BY_TYPE.txt


import subprocess
import random

from math import exp
import sys

trainingSize = sys.argv[1]
dropout1 = 0.1
emb_dim = 200
lstm_dim = 128
layers = 1


learning_rate = 0.1
dropout2 = 0.25
batch_size = 2
sequence_length = 20
input_noising = 0.1

language = 'English'

for model in ["GROUND", "RANDOM_BY_TYPE", "REAL_REAL"]:
  assert model in ["RANDOM_MODEL", "REAL_REAL", "RANDOM_BY_TYPE", "GROUND"], model

  for _ in range(10 if model == "RANDOM_BY_TYPE" else 5):
   command = ['./python27', 'yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_Fraction.py', language, language, dropout1, emb_dim, lstm_dim, layers, learning_rate, model, dropout2, batch_size,input_noising,  sequence_length, trainingSize]
   command = map(str,command)
   subprocess.call(command)

