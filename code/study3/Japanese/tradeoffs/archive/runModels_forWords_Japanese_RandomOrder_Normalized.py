PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"

import glob
models = glob.glob("../optimization/results/forWords*/*.tsv")
models.sort()

import subprocess

for i in range(len(models)):
    model = models[i]
    subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Japanese_RandomOrder_Normalized_FullData.py", "--model", model])
model = "REAL"
subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Japanese_RandomOrder_Normalized_FullData.py", "--model", model])
for _ in range(20):
  model = "RANDOM"
  subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Japanese_RandomOrder_Normalized_FullData.py", "--model", model])

