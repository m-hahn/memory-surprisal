import subprocess
import random

from math import exp
import sys

model = sys.argv[1]
if len(sys.argv) > 2:
  prescribedID = sys.argv[2]
else:
  prescribedID = None

assert model in ["REAL_REAL", "RANDOM_BY_TYPE_BRANCHING_ENT", "GROUND"], model

# /u/scr/mhahn/deps/memory-need-neural-wordforms/search-Russian_yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_ByBranchingEntropy.py_model_55489296_RANDOM_BY_TYPE_BRANCHING_ENT.txt
# based on part of the first one:
# /u/scr/mhahn/deps/memory-need-neural-wordforms/search-Russian_yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_ByBranchingEntropy.py_model_773655319_RANDOM_BY_TYPE_BRANCHING_ENT.txt
# independent, less good results
# /u/scr/mhahn/deps/memory-need-neural-wordforms/search-Russian_yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_ByBranchingEntropy.py_model_833001977_RANDOM_BY_TYPE_BRANCHING_ENT.txt


dropout1 = 0.4
emb_dim = 150
lstm_dim = 256
layers = 3

# -6.907755278982137
#>>> log(b)
#-2.995732273553991

learning_rate = 0.05
dropout2 = 0.2
batch_size = 2
sequence_length = 20
input_noising = 0.0

for i in range(100):
   command = ["./python27", "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_ByBranchingEntropy.py", "Russian", "Russian", dropout1, emb_dim, lstm_dim, layers, learning_rate, model, dropout2, batch_size,input_noising,  sequence_length]
   if prescribedID is not None:
     command.append(prescribedID)
   command = map(str,command)
   subprocess.call(command)
   if prescribedID is not None:
      break
