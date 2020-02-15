import sys
import os

language = sys.argv[1]
skip = sys.argv[2] == "True" if len(sys.argv) > 2 else False
redo = sys.argv[3] == "True" if len(sys.argv) > 3 else False
assert not (redo and skip)
import subprocess

DIR = "/u/scr/mhahn/deps/locality_optimized_i1/"
files = os.listdir(DIR)
results = []
import os.path as path

import random

random.shuffle(files)

for name in files:
  if name.startswith(language) and "POS" not in name and "FuncHead" not in name:
    print(name)
    if redo or not path.isfile("/u/scr/mhahn/deps/locality_optimized_i1/ORDERING_EVAL/ORDER_"+name):
      if skip:
         continue
      subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "evaluateOrdering.py", "--language", language, "--model", DIR+name])
    surpFile = DIR+"/it_estimates/estimates-"+language+"_"+"yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py"+"_model_"+name+".txt"
    print(surpFile)
    if path.isfile(surpFile):
     with open(surpFile, "r") as inFile:
       surprisal = float(inFile.read().strip().split("\n")[-1].split(" ")[1])
    else:
       surprisal = 100
    with open("/u/scr/mhahn/deps/locality_optimized_i1/ORDERING_EVAL/ORDER_"+name, "r") as inFile:
      accuracy = float(next(inFile).strip())
    results.append((name, accuracy, surprisal, DIR+name))



import glob

dlm = glob.glob("/u/scr/mhahn/deps/locality_optimized_dlm/manual_output_funchead_coarse_depl_quasiF/"+language+"_optimizeDependencyLength_QuasiF.py_model_*.tsv")
neural = [] #glob.glob("/u/scr/mhahn/deps/manual_output_funchead_langmod_coarse_best/"+language+"_readDataDistCrossGPUFreeAllTwoEqual_NoClip_ByCoarseOnly_FixObj_OnlyLangmod_Replication_Best.py_model_*.tsv")
models = dlm + neural

random.shuffle(models)

for nameAndPath in models:
  name = nameAndPath.split("/")[-1]
  print(name)
  if not path.isfile("/u/scr/mhahn/deps/locality_optimized_i1/ORDERING_EVAL/ORDER_"+name):
    subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "evaluateOrdering.py", "--language", language, "--model", nameAndPath])
  surpFile = DIR+"/it_estimates/estimates-"+language+"_"+"yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py"+"_model_"+name[-100:]+".txt"
  print(surpFile)
  if not path.isfile(surpFile):
    surpFile = DIR+"/it_estimates/estimates-"+language+"_"+"yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py"+"_model_"+name+".txt"
  if path.isfile(surpFile):
   with open(surpFile, "r") as inFile:
     surprisal = float(inFile.read().strip().split("\n")[-1].split(" ")[1])
  else:
     surprisal = 100
  with open("/u/scr/mhahn/deps/locality_optimized_i1/ORDERING_EVAL/ORDER_"+name, "r") as inFile:
    accuracy = float(next(inFile).strip())
  results.append((name, accuracy, surprisal, nameAndPath))

print(results)

#quit()


from collections import defaultdict
valuesByScript = defaultdict(list)
surpsByScript = defaultdict(list)

results = sorted(results, key=lambda x:-x[2])
for r in results:
   print(r)
   if r[1] in [0, 1]:
     continue
   valuesByScript[r[0][len(language)+1:(r[0].index("_model_"))]].append(r[1])
   if r[2] < 100:
      surpsByScript[r[0][len(language)+1:(r[0].index("_model_"))]].append(r[2])
print("------------")
print("/u/scr/mhahn/deps/locality_optimized_i1/PER_LANGUAGE_MEANS/"+language+".txt")
with open("/u/scr/mhahn/deps/locality_optimized_i1/PER_LANGUAGE_MEANS/"+language+".txt", "w") as outFile:
  for s, t in valuesByScript.items():
     u = surpsByScript[s]
     print(s, sum(t)/len(t), sum(u)/(len(u)+0.00000001))
     print >> outFile, "\t".join([str(x) for x in (s, sum(t)/len(t), sum(u)/(len(u)+0.00000001))])
