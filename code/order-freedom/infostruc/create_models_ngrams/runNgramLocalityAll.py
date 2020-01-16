
import subprocess
import os
import random

version = "yWithMorphologySequentialStreamDropoutDev_Ngrams_Log_Infostruc.py"
argumentNames = ["alpha", "gamma", "delta", "cutoff"]


BASE_DIR = "/u/scr/mhahn/deps/memory-need-ngrams/"

language = "Czech-PDT"
if True:
    filenames = ([x for x in os.listdir(BASE_DIR) if x.startswith("search-"+"Czech"+"_yWithMo") and len(open(BASE_DIR+x, "r").read().split("\n"))>=30])
    with open(BASE_DIR+filenames[0], "r") as inFile:
        params = next(inFile).strip().split("\t")[2:]
        assert len(params) == 4
        params[-1] = 20
    params2 = []
    for i in range(len(params)):
      params2.append("--"+argumentNames[i])
      params2.append(params[i])
    paramsString = " ".join([str(x) for x in params2])
#    command = ["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "yHyperParamSearchGPUs_CorPost_Automated_OnlyWordForms_Slurm_Ngrams.py", language, "1", "2", "NONE", "RANDOM_BY_TYPE", "0.02", "30"]
    for MODEL_TYPE in ["GROUND", "GROUND_INFOSTRUC"]: #"RANDOM_INFOSTRUC", "RANDOM_BY_TYPE"]: # "GROUND_INFOSTRUC", "REAL_REAL", 

       filenames = [x for x in os.listdir(BASE_DIR) if language in x and MODEL_TYPE in x]
       existing = 0
       for name in filenames:
           with open(BASE_DIR+name, "r") as inFile:
               if paramsString in inFile.read():
                   existing += 1


       print(language, MODEL_TYPE, existing)
#       continue
       for _ in range({"REAL_REAL" : 5, "RANDOM_BY_TYPE" : 20, "RANDOM_INFOSTRUC" : 20, "GROUND" : 5, "GROUND_INFOSTRUC" : 5}[MODEL_TYPE] - existing):
          command = map(str,["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", version, "--language", language, "--model", MODEL_TYPE] + params2)
          print(command)
#          quit()
          result = subprocess.call(command)

