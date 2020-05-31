
PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"

import glob

import random

import subprocess
 


for aff in ["Prefixes", "Suffixes"]:
  models = glob.glob(f"../optimization/results/forWords_Sesotho_OptimizeOrder_FormsWordsGraphemes_{aff}_ByType*/*.tsv")
  random.shuffle(models)

 
  for model in models:
      subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Sesotho_EvaluateWeights_"+aff+"_ByType.py", "--model", model])
  for _ in range(20):
      subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Sesotho_EvaluateWeights_"+aff+"_ByType.py", "--model", "RANDOM"])

