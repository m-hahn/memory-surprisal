PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"

import glob
models = glob.glob(PATH+"optimized_forWords_Japanese_OptimizeOrder*FullData*.py_*.tsv")

models.sort()

import subprocess

with open("optimizedModels.tsv", "w") as outFile:
  print("\t".join(["Script", "Model"]), file=outFile)
  for i in range(len(models)):
      model = models[i]
      script = model[model.rfind("forWords"):model.rfind("_")]
      model = model[model.rfind("_")+1:-4]
      print("\t".join([script, model]), file=outFile)
  
