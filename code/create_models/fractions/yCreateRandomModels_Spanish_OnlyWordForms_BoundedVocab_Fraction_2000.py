# /u/scr/mhahn/deps/memory-need-neural-wordforms-fractions/search-Spanish_yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_Fraction.py_model_2000_259363391_RANDOM_BY_TYPE.txt


import subprocess
import random

from math import exp
import sys

prescribedID = None
dropout1 = 0.45
emb_dim = 200
lstm_dim = 128
layers = 1


learning_rate = 0.05
dropout2 = 0.0
batch_size = 2
sequence_length = 20
input_noising = 0.05

fraction = 2000
language = 'Spanish'

for model in ["GROUND", "RANDOM_BY_TYPE", "REAL_REAL"]:
 for _ in range(10 if model == "RANDOM_BY_TYPE" else 5):
   command = ['./python27', 'yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_Fraction.py', language, language, dropout1, emb_dim, lstm_dim, layers, learning_rate, model, dropout2, batch_size,input_noising,  sequence_length, 'None', 'GPU0', 'False', fraction]
   command = map(str,command)
   subprocess.call(command)
quit()
