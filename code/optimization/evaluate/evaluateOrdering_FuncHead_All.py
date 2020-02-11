import sys
import os

language = sys.argv[1]
import subprocess

DIR = "/u/scr/mhahn/deps/locality_optimized_i1/"
files = os.listdir(DIR)
results = []
import os.path as path

for name in files:
  if name.startswith(language) and "POS" not in name and "FuncHead" in name:
    print(name)
    if not path.isfile("/u/scr/mhahn/deps/locality_optimized_i1/ORDERING_EVAL/ORDER_"+name):
      subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "evaluateOrdering_FuncHead.py", "--language", language, "--model", DIR+name])
    surpFile = DIR+"/it_estimates/estimates-"+language+"_"+"yWithMorphologySequentialStreamDropoutDev_Ngrams_Log_FuncHead.py"+"_model_"+name+".txt"
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

dlm = glob.glob("/u/scr/mhahn/deps/manual_output_funchead_coarse_depl/"+language+"_readDataDistCrossLGPUDepLengthMomentumEntropyUnbiasedBaseline_OrderBugFixed_NoPunct_NEWPYTORCH_AllCorpPerLang_BoundIterations_FuncHead_CoarseOnly.py_model_*.tsv")
neural = glob.glob("/u/scr/mhahn/deps/manual_output_funchead_langmod_coarse_best/"+language+"_readDataDistCrossGPUFreeAllTwoEqual_NoClip_ByCoarseOnly_FixObj_OnlyLangmod_Replication_Best.py_model_*.tsv")
models = dlm + neural


for nameAndPath in models:
  name = nameAndPath.split("/")[-1]
  print(name)
  if not path.isfile("/u/scr/mhahn/deps/locality_optimized_i1/ORDERING_EVAL/ORDER_"+name):
    subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "evaluateOrdering_FuncHead.py", "--language", language, "--model", nameAndPath])
  surpFile = DIR+"/it_estimates/estimates-"+language+"_"+"yWithMorphologySequentialStreamDropoutDev_Ngrams_Log_FuncHead.py"+"_model_"+name[-100:]+".txt"
  print(surpFile)
  if not path.isfile(surpFile):
    surpFile = DIR+"/it_estimates/estimates-"+language+"_"+"yWithMorphologySequentialStreamDropoutDev_Ngrams_Log_FuncHead.py"+"_model_"+name+".txt"
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




results = sorted(results, key=lambda x:-x[2])
for r in results:
   print(r)

