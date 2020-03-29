
PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"

import glob

for aff in ["Prefixes", "Suffixes"]:
  models = glob.glob(PATH+"optimized_Sesotho_Acqdiv_forWords_Sesotho_OptimizeOrder_FormsWordsGraphemes_"+aff+"_ByType*.py_*.tsv")
  
  import random
  random.shuffle(models)
  
  import subprocess
  
  for model in models:
      model = model[model.rfind("_")+1:-4]
      subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Sesotho_EvaluateWeights_"+aff+"_ByType.py", "--model", model])
  for _ in range(10):
      subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Sesotho_EvaluateWeights_"+aff+"_ByType.py", "--model", "RANDOM"])

