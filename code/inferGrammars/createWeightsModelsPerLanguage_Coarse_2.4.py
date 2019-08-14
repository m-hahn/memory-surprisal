languages = set(["Wolof_2.4", "Welsh-Adap_2.4", "Lithuanian_2.4"])
languages = list(languages)
assert len(languages) == len(set(languages))

import os
import random


import subprocess

modelsDir = "/u/scr/mhahn/deps/manual_output_ground_coarse/"

while len(languages) > 0:
  files = os.listdir(modelsDir)
  language = random.choice(languages)
  relevant = [x for x in files if x.startswith(language+"_infer")]
  relevantModelExists = False
  for filename in relevant:
      with open(modelsDir+filename, "r") as inFile:
         header = next(inFile).strip().split("\t")
         line = next(inFile).strip().split("\t")
         counter = int(line[header.index("Counter")])
         print(counter)
         if counter > 1000000:
            relevantModelExists = True
            break

  if relevantModelExists:
     languages.remove(language)
     continue
  subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "inferWeightsCrossVariationalAllCorpora_NoPunct_NEWPYTORCH_Coarse_V.py", language, language])
#  break

