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
print(files)

optimizedGrammars = glob.glob("/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/*.tsv")

# MAK at https://stackoverflow.com/questions/14063195/python-3-get-2nd-to-last-index-of-occurrence-in-string
def find_second_last(text, pattern):
   return text.rfind(pattern, 0, text.rfind(pattern))

files.sort()

from collections import defaultdict

resultsByOptScript = defaultdict(list)

hasSeenModels = set()

lastScript = None
#print(files)
with open("results.tsv", "w") as outFile:
 print("\t".join([str(x) for x in ["Script", "Run", "Model", "Accuracy_Pairs", "Accuracy_Full", "Accuracy_Pairs_Types", "Accuracy_Full_Types"]]), file=outFile)
 for f in files:
  with open(f, "r") as inFile:
     inFile = list(inFile)
     if len(inFile) < 5:
       continue
     print(inFile[3])
     if "." not in inFile[3]:
       continue
     accuracy_pairs = inFile[0]
     accuracy_full = inFile[1]
     accuracy_pairs_types = inFile[2]
     accuracy_full_types = inFile[3]
     errors = inFile[4:]
     script = f[f.index("forWords"):f.index(".py")+3]
     if script != lastScript:
        #print("\n\n\n===============", script, "==================")
        lastScript = script
     model = f[f.rfind("_")+1:-4]
     if model in hasSeenModels:
        continue
     else:
        hasSeenModels.add(model)
        
     run = f[find_second_last(f, "_")+1:f.rfind("_")]
     if run == "ByType.py":
        continue
     print("\t".join([str(x) for x in [script, run, model, accuracy_pairs.strip(), accuracy_full.strip(), accuracy_pairs_types.strip(), accuracy_full_types.strip()]]), file=outFile)
     if model != "RANDOM":
       relevantModelFiles = glob.glob("/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/*"+model+"*")
       #print(relevantModelFiles)
       assert len(relevantModelFiles) == 1
       opt_script = relevantModelFiles[0]
       opt_script = opt_script[opt_script.index("forWords"):opt_script.index(".py")+3]
       #print(script, opt_script, model, accuracy_pairs.strip(), accuracy_full.strip())
       #print(errors[:10])
       optimizedRelevant = [x for x in optimizedGrammars if model in x]
       print(optimizedRelevant)
       assert len(optimizedRelevant) == 1
       with open(optimizedRelevant[0], "r") as inFileG:
         grammar = inFileG.read().strip().split("\n")
      #   print(grammar)
         #print(grammar[0])
         iterations, auc = grammar[0].split(" ")
         auc = float(auc)
         convergenceHistory = grammar[1]
         arguments = grammar[2]
         grammar = dict([x.split(" ") for x in grammar[3:]])
      #   print(grammar)
         print(f)
         print(sorted(list(grammar)))
         morphemes_prefixes = ['ng', 'om', 'sm', 'sr', 't^']
         morphemes = [(x,[x]) for x in morphemes_prefixes] 
     #    print(morphemes)
         print(morphemes)
         weights = flatten([[(x, y, int(grammar[y])) for y in z] for x, z in morphemes])
         weights.sort(key=lambda x:x[2])
         #print(weights)
         #print(accuracy_full)
         #print(errors[:10])
         #auc = float(arguments[arguments.index(" ")+1:arguments.find(" ", arguments.find(" ")+1)])
         resultsByOptScript[opt_script].append(((auc, script, opt_script, model, accuracy_pairs.strip(), accuracy_full.strip()), arguments, weights, accuracy_full, errors[:10]))

print("\n")
print("\n")
print("\n")
print("\n")
print("=======================")
print("\n")
print("\n")
print("\n")
print("\n")
for opt_script in sorted(list(resultsByOptScript)):
    print("\n")
    print("\n")
    print("=============  "+opt_script+"  ==================")
    for r in sorted(resultsByOptScript[opt_script], key=lambda x:x[0][0]):
        print("---")
        for s in r:
          print(s)
