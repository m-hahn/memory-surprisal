import sys
import os

language = sys.argv[1]


PATH = "/u/scr/mhahn/deps/locality_optimized_i1/it_estimates/"

files = [x for x in os.listdir(PATH) if x.startswith("estimates-"+language+"_yWith") and "FuncHead" in x]

print(files)
results = []
for name in files:
   prefix = "estimates-"+language+"_yWithMorphologySequentialStreamDropoutDev_Ngrams_Log_FuncHead.py_model_"
   if name.startswith(prefix):
      model = name[len(prefix):-4]
      with open(PATH+name, "r") as inFile:
        data = inFile.read()
      results.append((model, [float(x) for x in data.split("\n")[2].split(" ")]))
results = sorted(results, key=lambda x:x[1][1], reverse=True)
with open("output/funchead_"+language+".tsv", "w") as outFile:
 print >> outFile, "\t".join(["Model", "BiSurp", "Surp"])
 for r in results:
   print(r)
   print >> outFile, ("\t".join([str(x) for x in [r[0], r[1][1], r[1][-1]]]))
# estimates-English_yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py_model_8027362_English_optimizePredictability_OnlyWords.py_model_7925234379.tsv.txt


