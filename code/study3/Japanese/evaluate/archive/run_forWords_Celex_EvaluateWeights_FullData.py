
PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"

import glob
models_sfx = glob.glob(f"../optimization/results/forWords_Japanese_OptimizeOrder_MorphemeGrammar_FormsPhonemesFull_FullData_HeldoutClip/*.tsv")
models_sfx += glob.glob(f"../optimization/results/forWords_Japanese_OptimizeOrder_MorphemeGrammar_FormsPhonemesFull_FullData/*.tsv")

import random
random.shuffle(models_sfx)

import subprocess

print(models_sfx)

for model in models_sfx:
  #  model = model[model.rfind("_")+1:-4]
    subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Celex_EvaluateWeights_FullData.py", "--model", model])
for _ in range(20):
    model = "RANDOM"
    subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Celex_EvaluateWeights_FullData.py", "--model", model])


