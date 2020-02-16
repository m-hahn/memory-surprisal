# /u/scr/mhahn/deps/memory-need-neural-wordforms/search-English_yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab.py_model_350445637_RANDOM_BY_TYPE.txt


import subprocess
import random

from math import exp
import sys

language = sys.argv[1]
model = sys.argv[2]

dropout1 = 0.15
emb_dim = 150
lstm_dim = 1024
layers = 2


learning_rate = 0.1
dropout2 = 0.35
batch_size = 2
sequence_length = 20
input_noising = 0.0


command = ['./python27', 'yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_NP.py', language, language, dropout1, emb_dim, lstm_dim, layers, learning_rate, model, dropout2, batch_size,input_noising,  sequence_length]
command = map(str,command)
subprocess.call(command)

