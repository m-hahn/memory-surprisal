from ud_languages import languages

import subprocess
import os
import random
random.shuffle(languages)

version = "yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py"
argumentNames = ["alpha", "gamma", "delta", "cutoff"]


for language in languages:

    filenames = [x for x in os.listdir("/u/scr/mhahn/deps/memory-need-ngrams/") if language in x and "GROUND" in x]
    if len(filenames) > 5:
      continue


    filenames = ([x for x in os.listdir("/u/scr/mhahn/deps/memory-need-ngrams/") if x.startswith("search-"+language+"_yWithMo") and len(open("/u/scr/mhahn/deps/memory-need-ngrams/"+x, "r").read().split("\n"))>=30])
    if len(filenames) == 0:
       continue
#    assert len(filenames) == 1
    with open("/u/scr/mhahn/deps/memory-need-ngrams/"+filenames[0], "r") as inFile:
        params = next(inFile).strip().split("\t")[2:]
        assert len(params) == 4
        params[-1] = 20
    params2 = []
    for i in range(len(params)):
      params2.append("--"+argumentNames[i])
      params2.append(params[i])
#    command = ["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "yHyperParamSearchGPUs_CorPost_Automated_OnlyWordForms_Slurm_Ngrams.py", language, "1", "2", "NONE", "RANDOM_BY_TYPE", "0.02", "30"]
    for MODEL_TYPE in ["GROUND"]: #"REAL_REAL", "RANDOM_BY_TYPE", "GROUND"]:
       for _ in range({"REAL_REAL" : 5, "RANDOM_BY_TYPE" : 20, "GROUND" : 5}[MODEL_TYPE]):
          command = map(str,["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", version, "--language", language, "--model", MODEL_TYPE] + params2)
          print(command)
#          quit()
          result = subprocess.call(command)

