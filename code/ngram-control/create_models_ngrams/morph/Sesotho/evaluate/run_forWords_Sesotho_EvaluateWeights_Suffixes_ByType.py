script = "forWords_Sesotho_RandomOrder_Normalized.py"

PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"

import glob
models_sfx = glob.glob(PATH+"optimized_Sesotho_Acqdiv_forWords_Sesotho_OptimizeOrder_FormsWordsGraphemes_Suffixes_ByType.py_*.tsv")

import random
random.shuffle(models_sfx)

import subprocess

for model in models_sfx:
    model = model[model.rfind("_")+1:-4]
    subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", "--model", model])

