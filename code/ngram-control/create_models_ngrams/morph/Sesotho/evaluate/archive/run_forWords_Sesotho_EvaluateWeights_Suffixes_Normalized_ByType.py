
PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"

import glob
models_pfx = glob.glob(PATH+"optimized_Sesotho_Acqdiv_forWords_Sesotho_OptimizeOrder_Normalized_ByType_Suffixes.py_*.tsv")

import random
random.shuffle(models_pfx)

import subprocess

for model in models_pfx:
    model = model[model.rfind("_")+1:-4]
    subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", "--model", model])

