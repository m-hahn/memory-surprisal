import sys
import os

PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-accuracy/"

import glob
files = glob.glob(PATH+"accuracy_forWords_Celex_EvaluateWeights_FullData.py_*.txt")

# MAK at https://stackoverflow.com/questions/14063195/python-3-get-2nd-to-last-index-of-occurrence-in-string
def find_second_last(text, pattern):
   return text.rfind(pattern, 0, text.rfind(pattern))

print(files)

with open("results.tsv", "w") as outFile:
 print("\t".join([str(x) for x in ["Script", "Run", "Model", "Accuracy_Pairs", "Accuracy_Full"]]), file=outFile)
 for f in files:
  with open(f, "r") as inFile:
     accuracy_pairs, accuracy_full = inFile 
     script = f[f.index("forWords"):f.index(".py")+3]
     model = f[f.rfind("_")+1:-4]
     run = f[find_second_last(f, "_")+1:f.rfind("_")]
     if run == "ByType.py":
        continue
     print(script, model, accuracy_pairs.strip(), accuracy_full.strip())
     print("\t".join([str(x) for x in [script, run, model, accuracy_pairs.strip(), accuracy_full.strip()]]), file=outFile)



