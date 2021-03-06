
PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"

import glob
models_sfx = glob.glob(PATH+"optimized_forWords_Japanese_OptimizeOrder_MorphemeGrammar*FullData*VarySets*.py_*.tsv")
#models_sfx += glob.glob(PATH+"optimized_forWords_Japanese_OptimizeOrder_MorphemeGrammar_Normalized_FullData.py_*.tsv")
#models_sfx += glob.glob(PATH+"optimized_forWords_Japanese_OptimizeOrder_MorphemeGrammar_FormsPhonemesFull_FullData.py_*.tsv")

import random
random.shuffle(models_sfx)

import subprocess

print(models_sfx)

for model in models_sfx:
    model = model[model.rfind("_")+1:-4]
    subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Celex_EvaluateWeights_MorphemeGrammar_FullData.py", "--model", model])


