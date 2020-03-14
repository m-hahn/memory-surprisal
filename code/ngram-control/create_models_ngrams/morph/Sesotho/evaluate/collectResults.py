import sys
import os

def flatten(x):
   r = []
   for y in x:
     for z in y:
       r.append(z)
   return r

PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-accuracy/"

import glob
files = glob.glob(PATH+"accuracy_forWords_Sesotho_EvaluateWeights_*.py_*.txt")

# MAK at https://stackoverflow.com/questions/14063195/python-3-get-2nd-to-last-index-of-occurrence-in-string
def find_second_last(text, pattern):
   return text.rfind(pattern, 0, text.rfind(pattern))

files.sort()

print(files)
with open("results.tsv", "w") as outFile:
 print("\t".join([str(x) for x in ["Script", "Run", "Model", "Accuracy_Pairs", "Accuracy_Full", "Accuracy_Pairs_Types", "Accuracy_Full_Types"]]), file=outFile)
 for f in files:
  with open(f, "r") as inFile:
     inFile = list(inFile)
     print(len(inFile))
     if len(inFile) < 5 or "." not in inFile[3]:
       continue
     accuracy_pairs = inFile[0]
     accuracy_full = inFile[1]
     accuracy_pairs_types = inFile[2]
     accuracy_full_types = inFile[3]
     errors = inFile[4:]
     script = f[f.index("forWords"):f.index(".py")+3]
     model = f[f.rfind("_")+1:-4]
     run = f[find_second_last(f, "_")+1:f.rfind("_")]
     print(f)
     if run == "ByType.py":
        continue
     print(script, model, accuracy_pairs.strip(), accuracy_full.strip())
     print("\t".join([str(x) for x in [script, run, model, accuracy_pairs.strip(), accuracy_full.strip()]]), file=outFile)



