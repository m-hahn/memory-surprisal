import os

PATH = "memory-need-pcfg/optimized"

path = "/u/scr/mhahn/deps/"+PATH+"/"
files = os.listdir(path)
import sys


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
args=parser.parse_args()
print(args)


def f(a):
   x=a   
   horizon = len(x)

   decay = [(x[(i-1 if i>0 else 0)]-x[i]) for i in range(len(x))]
   assert len(decay) == horizon

   memory = sum([i*(x[(i-1 if i>0 else 0)]-x[i]) for i in range(len(x))])

   residual = x[horizon-1]
   balanced = sum([i*(x[(i-1 if i>0 else 0)]-x[i]) for i in range(horizon)]) + horizon*x[horizon-1]

   mi = x[0] - x[horizon-1]

   unigramCE = x[0]
   return balanced, memory, residual, mi, decay, unigramCE

resultsPerType = {}



version = "yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py"

 


types = [" REAL_REAL ", "RANDOM_MODEL ", " RANDOM_BY_TYPE ", " GROUND "]

for fileName in files:
  if args.language not in fileName:
       continue
  if not fileName.startswith("estimates"):
     continue

  with open(path+fileName, "r") as inFile:
     result = inFile.read().strip().split("\n")
     print(len(result))
     params, _, surprisals = result
     params = dict([x.split("=") for x in params[10:-1].split(", ")])
     assert params["LEFT_CONTEXT"] == "5"
     assert params["MAX_BOUNDARY"] == "10"
     surprisals = [-float(x) for x in surprisals.split(" ")[int(params["LEFT_CONTEXT"])-1:]]
     print(params)
     print(surprisals)
     typeOfResult = params["model"][1:-1]
     if typeOfResult not in resultsPerType:
        resultsPerType[typeOfResult] = []
     balanced, memory, residual, mi, decay, unigramCE  = f(surprisals)
     duration = len(surprisals)
     uid = surprisals[0]

     q = fileName.index("_model_")
     q = fileName[q+7:]
     idOfModel = str(int(q[:q.index("_")]))

     resultsPerType[typeOfResult].append({"Residual" : residual, "ModelID" : idOfModel, "TotalMI" : mi, "Decay" : decay, "UnigramCE" : unigramCE})

print(resultsPerType)

outpath1 = "../../../results/raw/pcfg/"+args.language+"_pcfg_after_tuning.tsv"
outpath2 = "../../../results/raw/pcfg/"+args.language+"_pcfg_decay_after_tuning.tsv"

header = ["Type", "Residual", "ModelID", "TotalMI"]
headerDecay = ["Type", "Distance", "ConditionalMI", "TotalMI", "ModelID", "UnigramCE"]
with open(outpath1, "w") as outFile:
 with open(outpath2, "w") as outFileDecay:

  print >> outFile, "\t".join(header)
  print >> outFileDecay, "\t".join(headerDecay)
  
  for typeOfResult in types:
     typeOfResult = typeOfResult.strip()
#     print("#"+typeOfResult+"#")
 #    quit()
     if len(resultsPerType.get(typeOfResult, [])) == 0:
        continue
     print
     print typeOfResult
     for rand in resultsPerType.get(typeOfResult, []):
        print(rand) 
        rand["Type"] = typeOfResult
        outLine = [rand[x] for x in header]
        assert len(outLine) == len(header)
        print >> outFile, "\t".join(map(str,outLine))


        for i in range(1,len(rand["Decay"])):
           rand["Distance"] = i
           rand["ConditionalMI"] = max(0, rand["Decay"][i])
           print >> outFileDecay, "\t".join([str(rand[x]) for x in headerDecay]) 
print "../../../results/raw/pcfg/"+args.language+"_pcfg_after_tuning.tsv" 
print "../../../results/raw/pcfg/"+args.language+"_pcfg_decay_after_tuning.tsv"


