
import subprocess
import os
import random



BASE_DIR = "/u/scr/mhahn/deps/memory-need-neural-wordforms_infostruc/"


from math import exp
import sys



dropout1 = 0.05
emb_dim = 200
lstm_dim = 256
layers = 3


learning_rate = 0.05
dropout2 = 0.25
batch_size = 2
sequence_length = 20
input_noising = 0.0




language = "Czech-PDT"
if True:
   MODEL_TYPE = random.choice(["REAL_REAL", "RANDOM_INFOSTRUC", "RANDOM_BY_TYPE"]) # "GROUND", "GROUND_INFOSTRUC", 
   filenames = [x for x in os.listdir(BASE_DIR) if language in x and MODEL_TYPE in x]
   existing = 0
   for name in filenames:
       with open(BASE_DIR+name, "r") as inFile:
           if paramsString in inFile.read():
               existing += 1


   print(language, MODEL_TYPE, existing)
   if existing < {"REAL_REAL" : 5, "RANDOM_BY_TYPE" : 20, "RANDOM_INFOSTRUC" : 20, "GROUND" : 5, "GROUND_INFOSTRUC" : 5}[MODEL_TYPE]:
      command = ['./python27', 'yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_Infostruc.py', language, language, dropout1, emb_dim, lstm_dim, layers, learning_rate, MODEL_TYPE, dropout2, batch_size,input_noising,  sequence_length]
      print(command)
      result = subprocess.call([str(x) for x in command])



