# /u/scr/mhahn/deps/memory-need-neural-wordforms/search-Lithuanian_2.4_yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_V.py_model_948113700_RANDOM_BY_TYPE.txt


import subprocess
import random

from math import exp
import sys

model = sys.argv[1]
if len(sys.argv) > 2:
  prescribedID = sys.argv[2]
else:
  prescribedID = None
assert model in ['REAL_REAL', 'RANDOM_BY_TYPE', 'GROUND'], model
dropout1 = 0.4
emb_dim = 50
lstm_dim = 256
layers = 2


learning_rate = 0.001
dropout2 = 0.25
batch_size = 2
sequence_length = 20
input_noising = 0.45

language = 'Lithuanian_2.4'

for i in range(100):
   command = ['./python27', 'yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_V.py', language, language, dropout1, emb_dim, lstm_dim, layers, learning_rate, model, dropout2, batch_size,input_noising,  sequence_length]
   if prescribedID is not None:
     command.append(prescribedID)
   command = map(str,command)
   subprocess.call(command)
   if prescribedID is not None:
      break
quit()
