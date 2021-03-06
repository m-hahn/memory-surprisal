import sys
import os

def flatten(x):
   r = []
   for y in x:
     for z in y:
       r.append(z)
   return r


import glob
files = glob.glob("results/accuracy_forWords_Celex_EvaluateWeights_MorphemeGrammar_FullData.py_*.txt")


optimizedGrammars = glob.glob("../optimization/results/*/*.tsv")

# MAK at https://stackoverflow.com/questions/14063195/python-3-get-2nd-to-last-index-of-occurrence-in-string
def find_second_last(text, pattern):
   return text.rfind(pattern, 0, text.rfind(pattern))

files.sort()

from collections import defaultdict

resultsByOptScript = defaultdict(list)

hasSeenModels = set()

#print(files)
with open("results.tsv", "w") as outFile:
 print("\t".join([str(x) for x in ["Script", "Run", "Model", "Accuracy_Pairs", "Accuracy_Full", "Accuracy_Pairs_Types", "Accuracy_Full_Types"]]), file=outFile)
 for f in files:
  with open(f, "r") as inFile:
     inFile = list(inFile)
     accuracy_pairs, accuracy_full, accuracy_pairs_types, accuracy_full_types, errors = inFile[0], inFile[1], inFile[2], inFile[3], inFile[4:]
     script = f[f.index("forWords"):f.index(".py")+3]
     model = f[f.rfind("_")+1:-4]
     if model in hasSeenModels and model != "RANDOM":
        assert False
        continue
     else:
        hasSeenModels.add(model)
        
     run = f[find_second_last(f, "_")+1:f.rfind("_")]
     print("\t".join([str(x) for x in [script, run, model, accuracy_pairs.strip(), accuracy_full.strip(), accuracy_pairs_types.strip(), accuracy_full_types.strip()]]), file=outFile)
     if model != "RANDOM":
       relevantModelFiles = glob.glob("../optimization/results/*/*"+model+"*")
       assert len(relevantModelFiles) == 1
       opt_script = relevantModelFiles[0]
       opt_script = opt_script[opt_script.index("forWords"):opt_script.index(".py")+3]
       optimizedRelevant = [x for x in optimizedGrammars if model in x]
       assert len(optimizedRelevant) == 1
       with open(optimizedRelevant[0], "r") as inFileG:
         grammar = inFileG.read().strip().split("\n")
         arguments = grammar[0]
         grammar = dict([x.split(" ") for x in grammar[1:]])
         morphemes = [("suru", ['する'])]
         if "MorphemeGrammar" not in f:
           assert False
           morphemes.append(("causative", ['せる']))
           morphemes.append(('passive/potential', ['れる', 'られる', '得る', 'える']))
         else:
           morphemes.append(("causative", ['CAUSATIVE']))
           morphemes.append(('passive/potential', ['PASSIVE_POTENTIAL']))
         morphemes.append(('politeness', ['ます']))
         morphemes.append(('desiderative', ['たい']))
         morphemes.append(('negation', ['ない']))
         morphemes.append(('past', ['た']))
         morphemes.append(('future', ['う']))
         morphemes.append(('nonfinite', ['て']))
         weights = flatten([[(x, y, int(grammar[y])) for y in z] for x, z in morphemes])
         weights.sort(key=lambda x:x[2])
         auc = float(arguments[arguments.index(" ")+1:arguments.find(" ", arguments.find(" ")+1)])
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
