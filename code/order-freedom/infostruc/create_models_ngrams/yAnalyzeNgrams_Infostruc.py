import os

PATH = "memory-need-ngrams_infostruc"

path = "/u/scr/mhahn/deps/"+PATH+"/"
files = os.listdir(path)
import sys


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
args=parser.parse_args()
print(args)


def f(a):
   
   x = list(map(float,a.split(" ")))
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


searchLanguage = "Czech"
filenames = [x for x in os.listdir("/u/scr/mhahn/deps/memory-need-ngrams/") if x.startswith("search-"+searchLanguage+"_yWithMo") and len(open("/u/scr/mhahn/deps/memory-need-ngrams/"+x, "r").read().split("\n"))>=30]
#assert len(filenames) == 1, filenames
if len(filenames) == 0:
   quit()
with open("/u/scr/mhahn/deps/memory-need-ngrams/"+filenames[0], "r") as inFile:
    params = next(inFile).strip().split("\t")[2:]
    assert len(params) == 4
    params[-1] = "20"
print("Parameters identified in the search: ", params)  
version = "yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py"
argumentNames = ["alpha", "gamma", "delta", "cutoff"]

 
params2 = []
for i in range(len(params)):
  params2.append("--"+argumentNames[i])
  params2.append(params[i])
correctParameters = " ".join(params2)


types = [" REAL_REAL ", "RANDOM_MODEL ", "RANDOM_BY_TYPE ", " GROUND ", " RANDOM_INFOSTRUC ", " GROUND_INFOSTRUC "]

for fileName in files:
  if args.language not in fileName:
       continue
  if not fileName.startswith("estimates"):
     continue

  with open(path+fileName, "r") as inFile:
     result = inFile.read().split("\n")
     if (correctParameters+" " not in result[0]+" "):
         continue
     print(fileName)
     typeOfResult = filter(lambda x:x in result[0], types)[0][:-1]
     if len(result) < 3:
         continue
     if typeOfResult not in resultsPerType:
        resultsPerType[typeOfResult] = []
     if "idForProcess" in result[0]:
        continue
     result[1] = list(map(lambda x:x if x=="" else float(x), result[1].replace("[","").replace("]","").split(" ")))
     balanced, memory, residual, mi, decay, unigramCE  = f(result[2])
     duration = len(result[1])
     uid = float(result[4].split(" ")[1]) if "Exponent" in fileName else "NA"

     q = fileName.index("_model_")
     q = fileName[q+7:]
     idOfModel = str(int(q[:q.index("_")]))

     resultsPerType[typeOfResult].append({"Parameters" : result[0], "Memory" : memory, "Residual" : residual, "ModelID" : idOfModel, "TotalMI" : mi, "Decay" : decay, "UnigramCE" : unigramCE})



outpath1 = "../raw/ngrams/"+args.language+"_ngrams_infostruc_after_tuning.tsv"
outpath2 = "../raw/ngrams/"+args.language+"_ngrams_infostruc_decay_after_tuning.tsv"

header = ["Type", "Memory", "Residual", "ModelID", "TotalMI"]
headerDecay = ["Type", "Distance", "ConditionalMI", "TotalMI", "ModelID", "UnigramCE"]
with open(outpath1, "w") as outFile:
 with open(outpath2, "w") as outFileDecay:

  print >> outFile, "\t".join(header)
  print >> outFileDecay, "\t".join(headerDecay)
  
  for typeOfResult in types:
     if len(resultsPerType.get(typeOfResult[:-1], [])) == 0:
        continue
     print
     print typeOfResult[:-1]
     for rand in resultsPerType.get(typeOfResult[:-1], []):
        parameters = rand["Parameters"].split(" ")
        print(rand) 
        rand["Type"] = parameters[4]
        outLine = [rand[x] for x in header]
        assert len(outLine) == len(header)
        print >> outFile, "\t".join(map(str,outLine))


        for i in range(1,len(rand["Decay"])):
           rand["Distance"] = i
           rand["ConditionalMI"] = max(0, rand["Decay"][i])
           print >> outFileDecay, "\t".join([str(rand[x]) for x in headerDecay]) 
print "../raw/ngrams/"+args.language+"_ngrams_infostruc_after_tuning.tsv" 
print "../raw/ngrams/"+args.language+"_ngrams_infostruc_decay_after_tuning.tsv"
