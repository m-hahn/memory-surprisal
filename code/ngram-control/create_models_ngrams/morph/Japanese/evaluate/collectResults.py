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
files = glob.glob(PATH+"accuracy_forWords_Celex_EvaluateWeights_FullData.py_*.txt")


optimizedGrammars = glob.glob("/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/*.tsv")

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
     if run == "ByType.py":
        continue
     print(script, model, accuracy_pairs.strip(), accuracy_full.strip())
     print("\t".join([str(x) for x in [script, run, model, accuracy_pairs.strip(), accuracy_full.strip(), accuracy_pairs_types.strip(), accuracy_full_types.strip()]]), file=outFile)
     if model != "RANDOM":
       #print(errors[:10])
       optimizedRelevant = [x for x in optimizedGrammars if model in x]
       print(optimizedRelevant)
       assert len(optimizedRelevant) == 1
       with open(optimizedRelevant[0], "r") as inFileG:
         grammar = inFileG.read().strip().split("\n")
      #   print(grammar)
         print(grammar[0])
         grammar = dict([x.split(" ") for x in grammar[1:]])
      #   print(grammar)
         morphemes = [("suru", ['する'])]
         morphemes.append(("causative", ['せる']))
         morphemes.append(('passive/potential', ['れる', 'られる', '得る', 'える']))
         morphemes.append(('politeness', ['ます']))
         morphemes.append(('desiderative', ['たい']))
         morphemes.append(('negation', ['ない']))
         morphemes.append(('past', ['た']))
         morphemes.append(('future', ['う']))
         morphemes.append(('nonfinite', ['て']))
     #    print(morphemes)
         weights = flatten([[(x, y, int(grammar[y])) for y in z] for x, z in morphemes])
         weights.sort(key=lambda x:x[2])
         print(weights)
         print(accuracy_full)
         print(errors[:10])
