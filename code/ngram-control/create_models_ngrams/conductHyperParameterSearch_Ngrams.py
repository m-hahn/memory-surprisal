
from ud_languages import languages

import subprocess
import os
import random
random.shuffle(languages)
for language in languages:
    filenames = [x for x in os.listdir("/u/scr/mhahn/deps/memory-need-ngrams/") if x.startswith("search-"+language+"_yWithMo") and len(open("/u/scr/mhahn/deps/memory-need-ngrams/"+x, "r").read().split("\n"))>=30]
    
    if len(filenames) > 0:
      continue
#    if language == "Czech":
 #      continue
    command = ["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "yHyperParamSearchGPUs_CorPost_Automated_OnlyWordForms_Slurm_Ngrams.py", language, "1", "2", "NONE", "RANDOM_BY_TYPE", "0.02", "30"]
    print(command)
    result = subprocess.call(command)

