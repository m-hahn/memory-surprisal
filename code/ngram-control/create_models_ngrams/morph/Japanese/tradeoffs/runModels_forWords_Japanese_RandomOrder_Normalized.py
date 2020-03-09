PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"

import glob
models = glob.glob(PATH+"optimized_Japanese_2.4_optimized_forWords_Japanese_OptimizeOrder_Normalized_FullData_suru.py_*.tsv")

models.sort()

import subprocess

for i in range(len(models)):
    model = models[i]
    model = model[model.rfind("_")+1:-4]
    subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Japanese_RandomOrder_Normalized_FullData_suru.py", "--model", model])
model = "REAL"
subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Japanese_RandomOrder_Normalized_FullData_suru.py", "--model", model])
for _ in range(10):
  model = "RANDOM"
  subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "forWords_Japanese_RandomOrder_Normalized_FullData_suru.py", "--model", model])

