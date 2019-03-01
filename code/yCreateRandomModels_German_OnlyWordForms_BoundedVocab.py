# /u/scr/mhahn/deps/memory-need-neural-wordforms/search-German_yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab.py_model_833265572_RANDOM_BY_TYPE.txt


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
dropout1 = 0.3
emb_dim = 100
lstm_dim = 512
layers = 1


learning_rate = 0.05
dropout2 = 0.0
batch_size = 8
sequence_length = 20
input_noising = 0.0

language = 'German'

for i in range(100):
   command = ['./python27', 'yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab.py', language, language, dropout1, emb_dim, lstm_dim, layers, learning_rate, model, dropout2, batch_size,input_noising,  sequence_length]
   if prescribedID is not None:
     command.append(prescribedID)
   command = map(str,command)
   subprocess.call(command)
   if prescribedID is not None:
      break
quit()
