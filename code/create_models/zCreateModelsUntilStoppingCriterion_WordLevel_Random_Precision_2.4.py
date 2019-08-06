
import subprocess
import random
from math import exp
import random
import sys
import numpy as np
import os
import subprocess
import time


real_type = sys.argv[1] if len(sys.argv) > 1 else "REAL_REAL"

languages = []
languages += ["Wolof_2.4", "Welsh-Adap_2.4", "Lithuanian_2.4"]
	


satisfiedLanguages = set()

myID = random.randint(0,1000000000)




#languages = {"GROUND" : languages[::], "RANDOM_BY_TYPE" : languages[::], "RWAL_REAL" : languages[::]}

perGPU = [0]

#version = "yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab.py"

path = "/u/scr/mhahn/deps/memory-need-neural-wordforms/"

while len(languages) > 0:
    language = random.choice(languages)

    print(language)
    if "yCreateRandomModels_"+language+"_OnlyWordForms_BoundedVocab.py" not in os.listdir("/u/scr/mhahn/CODE/deps/"):
       print("yCreateRandomModels_"+language+"_OnlyWordForms_BoundedVocab.py doesn't exist")
       languages.remove(language)
       continue
#    if "Adap" in language:
 #      languages.remove(language)
  #     continue
    print("Looking for next language "+language)
    

    subprocess.check_output(["./python27", "yAnalyzeNeuralMorphFinalSaveLast_OnlyWordForms_BoundedVocab_Adjusted.py", "2", "20", language, "True", "True"])


    stats = (subprocess.check_output(["./python27", "tradeoffStats_OnlyWordForms_BoundedVocab.py", language, real_type])).strip().split("\t")
    print(stats)
    if int(stats[1]) > 100 and int(stats[2]) > 300:
        languages.remove(language)
        print("Enough for now")
        continue
    if int(stats[1]) < 10:
         typ = real_type
         print("Not enough REAL_REAL")
    elif int(stats[2]) < 10:
         typ = "RANDOM_BY_TYPE"
         print("Not enough RANDOM_BY_TYPE")
    else:
       commandCurve = ["./python27", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_Adjusted.py", language, "RANDOM_BY_TYPE", real_type]
       inCurve = [x.split("\t") for x in subprocess.check_output(commandCurve).strip().split("\n")]
   #    print(inCurve)
       distanceCurve = float(inCurve[1][0])
   
       print(inCurve)
       print("DISTANCE CURVE", distanceCurve)
       if abs(distanceCurve) < 0.15:
         languages.remove(language)
         continue
       print("Not enough precision")
   #    quit()
       typ = random.choice([real_type, "RANDOM_BY_TYPE", "RANDOM_BY_TYPE"])

    command = map(str,["./python27", "yCreateRandomModels_"+language+"_OnlyWordForms_BoundedVocab.py", typ, random.randint(0,10000000)]) #, "GPU"+str(gpu)
    print " ".join(command)
   
    #my_env["CUDA_VISIBLE_DEVICES"] = str(gpus[gpu])
    subprocess.call(command)



