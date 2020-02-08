import sys
import os

language = sys.argv[1]


PATH = "/u/scr/mhahn/deps/locality_optimized_i1/it_estimates/"

files = [x for x in os.listdir(PATH) if x.startswith("estimates-"+language+"_yWith")]

print(files)
results = []
for name in files:
   prefix = "estimates-English_yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py_model_"
   if name.startswith(prefix):
      model = name[len(prefix):-4]
      with open(PATH+name, "r") as inFile:
        data = inFile.read()
      results.append((model, [float(x) for x in data.split("\n")[2].split(" ")]))
results = sorted(results, key=lambda x:x[1][-1], reverse=True)
for r in results:
   print(r)
# estimates-English_yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py_model_8027362_English_optimizePredictability_OnlyWords.py_model_7925234379.tsv.txt


