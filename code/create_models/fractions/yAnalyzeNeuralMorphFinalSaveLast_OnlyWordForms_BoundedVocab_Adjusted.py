import os

PATH = "memory-need-neural-wordforms-fractions"

path = "/u/scr/mhahn/deps/"+PATH+"/"
files = os.listdir(path)
import sys


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sortBy", type=int, default=0)
parser.add_argument("--horizon", type=int, default=20)
parser.add_argument("--language", dest="language", type=str, default="Czech")
parser.add_argument("--restrictToFinished", type=bool)
parser.add_argument("--onlyOptimized", type=bool, default=True)
parser.add_argument("--trainingSize", type=str)


args=parser.parse_args()
print(args)


def f(a):
   x = list(map(float,a.split(" ")))[:args.horizon]
   x = [min(x[:i+1]) for i in range(len(x))]
   decay = [(x[(i-1 if i>0 else 0)]-x[i]) for i in range(len(x))]
   assert len(decay) == args.horizon
   memory = sum([i*(x[(i-1 if i>0 else 0)]-x[i]) for i in range(len(x))])

   residual = x[args.horizon-1]
   balanced = sum([i*(x[(i-1 if i>0 else 0)]-x[i]) for i in range(args.horizon)]) + args.horizon*x[args.horizon-1]

   mi = x[0] - x[args.horizon-1]

   unigramCE = x[0]
   return balanced, memory, residual, mi, decay, unigramCE

resultsPerType = {}



types = [" REAL_REAL ", "RANDOM_BY_TYPE ", " GROUND ", " RANDOM_BY_TYPE_BRANCHING_ENT "]

# removed: " REAL ", 
# "RANDOM_MODEL_ST ", "RANDOM_BY_TYPE_ST ", "RANDOM_MODEL_CONS ", "RANDOM_BY_TYPE_CONS ", "RANDOM_MODEL_NONP ", "RANDOM_BY_TYPE_NONP ", " TOTALLY_RANDOM ", 
# , " GROUND_OVS ", " GROUND_SOV "
# "RANDOM_MODEL ", 

averageUnigramCE = [0.0,0]

for fileName in files:
  if "_"+args.trainingSize+"_" not in fileName:
       continue
  if not fileName.startswith("estimates"):
     continue
  if args.language not in fileName:
      continue
  with open(path+fileName, "r") as inFile:
     result = inFile.read().strip().split("\n")
     if len(result) < 4:
       continue

     typeOfResult = filter(lambda x:x in result[0], types)[0][:-1]
     if len(result) < 3:
         continue
     if typeOfResult not in resultsPerType:
        resultsPerType[typeOfResult] = []
     result[1] = list(map(lambda x:x if x=="" else float(x), result[1].replace("[","").replace("]","").split(" ")))
     if args.restrictToFinished:
       if len(result[1]) == 1 or result[1][-1] < result[1][-2]: # this had not been stopped by early stopping
          continue
     balanced, memory, residual, mi, decay, unigramCE  = f(result[2])
     duration = len(result[1])
     uid = float(result[4].split(" ")[1]) if "Exponent" in fileName else "NA"


     q = fileName.index("_model_")
     q = fileName[q+7:]
     idOfModel = str(int(q[:q.index("_")]))

     averageUnigramCE[0] += unigramCE
     averageUnigramCE[1] += 1


     resultsPerType[typeOfResult].append([balanced, memory, residual, result[0], duration, uid, idOfModel, mi, memory/mi, decay, unigramCE])




if averageUnigramCE[1] == 0:
    print("no results")
    print("/u/scr/mhahn/"+args.language+"_after_tuning_onlyWordForms_boundedVocab.tsv")
    print("/u/scr/mhahn/"+args.language+"_decay_after_tuning_onlyWordForms_boundedVocab.tsv")
    quit()

averageUnigramCE = averageUnigramCE[0] / averageUnigramCE[1]


outpath1 = "/u/scr/mhahn/"+args.language+"_after_tuning_onlyWordForms_boundedVocab_fraction"+args.trainingSize+".tsv"
outpath2 = "/u/scr/mhahn/"+args.language+"_decay_after_tuning_onlyWordForms_boundedVocab_fraction"+args.trainingSize+".tsv"

header = ["Model", "Language", "Code", "Drop1", "Emb", "Dim", "Layers", "lr", "Type", "Drop2", "Batch", "Length", "Balanced", "Memory", "Residual", "Duration", "NonUniformity", "ModelID", "MI"]
headerDecay = ["Model", "Language", "Code", "Type", "Distance", "ConditionalMI", "TotalMI", "ModelID", "UnigramCE"]
with open(outpath1, "w") as outFile:
 with open(outpath2, "w") as outFileDecay:

  print >> outFile, "\t".join(header)
  print >> outFileDecay, "\t".join(headerDecay)
  
  for typeOfResult in types:
     if len(resultsPerType.get(typeOfResult[:-1], [])) == 0:
        continue
     print
     print typeOfResult[:-1]
     print "\n".join(map(lambda x:str(x[:-1]),sorted(resultsPerType.get(typeOfResult[:-1], []), key=lambda x:x[args.sortBy])))
     for rand in resultsPerType.get(typeOfResult[:-1], []):
        parameters = rand[3].split(" ")
        if len(parameters) + 7 > len(header):
           parameters = parameters[:len(header)-7]
        assert len(parameters) + 7 == len(header), (len(parameters)+7, len(header))

        unigramCEHere = rand[10]
        rand[9][1] += (-unigramCEHere + averageUnigramCE)
        rand[1] += (-unigramCEHere + averageUnigramCE)
        rand[10] = averageUnigramCE

        print >> outFile, "\t".join(map(str,parameters + [rand[0], rand[1], rand[2], rand[4], rand[5], rand[6], rand[7]]))
        for i in range(1,args.horizon):
           print >> outFileDecay, "\t".join(map(str,[parameters[0], parameters[1], parameters[2], parameters[8], i, max(0, rand[9][i]), rand[7], rand[6], rand[10]]))
 
print outpath1
print outpath2
