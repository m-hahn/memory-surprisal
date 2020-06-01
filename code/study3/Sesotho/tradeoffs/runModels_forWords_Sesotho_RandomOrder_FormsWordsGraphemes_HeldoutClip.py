script = "forWords_Sesotho_RandomOrder_Normalized.py"

PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"

import glob
models_pfx = glob.glob("../optimization/results/forWords_Sesotho_OptimizeOrder_FormsWordsGraphemes_Prefixes_ByType_HeldoutClip/*.tsv")
models_sfx = glob.glob("../optimization/results/forWords_Sesotho_OptimizeOrder_FormsWordsGraphemes_Suffixes_ByType_HeldoutClip/*.tsv")

models_pfx.sort()
models_sfx.sort()
import random
random.shuffle(models_pfx)
random.shuffle(models_sfx)

import subprocess

for i in range(min(len(models_pfx), len(models_sfx))):
    model_pfx = models_pfx[i]
    model_sfx = models_sfx[i]
    subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Sesotho_RandomOrder_FormsWordsGraphemes_HeldoutClip.py", "--model_pfx", model_pfx, "--model_sfx", model_sfx])
model_sfx = "REAL"
subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Sesotho_RandomOrder_FormsWordsGraphemes_HeldoutClip.py", "--model_pfx", model_pfx, "--model_sfx", model_sfx])
for _ in range(40):
  model_pfx = "RANDOM"
  model_sfx = "RANDOM"
  subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Sesotho_RandomOrder_FormsWordsGraphemes_HeldoutClip.py", "--model_pfx", model_pfx, "--model_sfx", model_sfx])
 
