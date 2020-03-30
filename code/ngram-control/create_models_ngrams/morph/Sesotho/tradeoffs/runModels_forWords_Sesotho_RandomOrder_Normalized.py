script = "forWords_Sesotho_RandomOrder_Normalized.py"

PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"

import glob
models_pfx = glob.glob(PATH+"optimized_Sesotho_Acqdiv_forWords_Sesotho_OptimizeOrder_Normalized_ByType_Prefixes.py_*.tsv")
models_sfx = glob.glob(PATH+"optimized_Sesotho_Acqdiv_forWords_Sesotho_OptimizeOrder_Normalized_ByType_Suffixes.py_*.tsv")

models_pfx.sort()
models_sfx.sort()

import subprocess

for i in range(min(len(models_pfx), len(models_sfx))):
    model_pfx = models_pfx[i]
    model_sfx = models_sfx[i]
    model_pfx = model_pfx[model_pfx.rfind("_")+1:-4]
    model_sfx = model_sfx[model_sfx.rfind("_")+1:-4]
    subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Sesotho_RandomOrder_Normalized.py", "--model_pfx", model_pfx, "--model_sfx", model_sfx])
model_pfx = "REAL"
model_sfx = "REAL"
subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Sesotho_RandomOrder_Normalized.py", "--model_pfx", model_pfx, "--model_sfx", model_sfx])
for _ in range(10):
  model_pfx = "RANDOM"
  model_sfx = "RANDOM"
  subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Sesotho_RandomOrder_FormsWordsGraphemes.py", "--model_pfx", model_pfx, "--model_sfx", model_sfx])
 
