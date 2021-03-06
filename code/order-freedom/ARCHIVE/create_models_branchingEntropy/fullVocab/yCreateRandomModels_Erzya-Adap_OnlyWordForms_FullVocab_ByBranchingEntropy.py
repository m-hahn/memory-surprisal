# /u/scr/mhahn/deps/memory-need-neural-wordforms/search-Erzya-Adap_yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_FullVocab_ByBranchingEntropy.py_model_150649378_RANDOM_BY_TYPE_BRANCHING_ENT.txt



import subprocess
import random

from math import exp
import sys

model = sys.argv[1]
if len(sys.argv) > 2:
  prescribedID = sys.argv[2]
else:
  prescribedID = None

assert model in ["RANDOM_MODEL", "REAL_REAL", "RANDOM_BY_TYPE_BRANCHING_ENT", "GROUND"], model


dropout1 = 0.4
emb_dim = 200
lstm_dim = 128
layers = 1

# -6.907755278982137
#>>> log(b)
#-2.995732273553991

learning_rate = 0.05
dropout2 = 0.1
batch_size = 2
sequence_length = 20
input_noising = 0.4

language = "Erzya-Adap"

for i in range(100):
   command = ["./python27", "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_FullVocab_ByBranchingEntropy.py", language, language, dropout1, emb_dim, lstm_dim, layers, learning_rate, model, dropout2, batch_size,input_noising,  sequence_length]
   if prescribedID is not None:
     command.append(prescribedID)
   command = map(str,command)
   subprocess.call(command)
   if prescribedID is not None:
      break
