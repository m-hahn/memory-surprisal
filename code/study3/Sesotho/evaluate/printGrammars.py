import sys
import os

def flatten(x):
   r = []
   for y in x:
     for z in y:
       r.append(z)
   return r


import glob
files = glob.glob("results/accuracy_forWords_Sesotho_EvaluateWeights_*.py_*.txt")
print(files)


optimizedGrammars = glob.glob("../optimization/results/*/*.tsv")

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
         iterations, auc = grammar[0].split(" ")
         auc = float(auc)
         convergenceHistory = grammar[1].split(" ")
         arguments = grammar[2]
         cutoffPerScript = {}
         grammar = dict([x.split(" ") for x in grammar[3:]])
         morphemes_prefixes = ['ng', 'om', 'sm', 'sr', 't^']
         morphemes_suffixes = ['ap', 'c', 'nt', 'rv', 'rc', 'p', 't^', 'm^', 'wh', 'rl']
         morphemes = [(x,[x]) for x in (morphemes_prefixes if "refix" in f else morphemes_suffixes)] 
         weights = flatten([[(x, y, int(grammar[y])) for y in z] for x, z in morphemes])
         weights.sort(key=lambda x:x[2])
         resultsByOptScript[opt_script].append(((auc, len(convergenceHistory)*50, script, opt_script, model, accuracy_pairs.strip(), accuracy_full.strip()), arguments, weights, accuracy_full, errors))

print("\n")
print("\n")
print("\n")
print("\n")
print("=======================")
print("\n")
print("\n")
print("\n")
print("\n")

names = {}
print(morphemes_prefixes)
print(morphemes_suffixes)
names = {'ng' : "Negation", 'om' : "Object", 'sm' : "Subject", 'sr' : "Subject (relative)", 't^' : "Tense/aspect", 'ap' : "Applicative", 'c' : "Causative", 'nt' : "Neuter", 'rv' : "Reversive", 'rc' : "Reciprocal", 'p' : "Passive", 'm^' : "Mood", 'wh' : "Interrogative", 'rl' : "Relative"}
for opt_script in sorted(list(resultsByOptScript)):
    print("\n")
    print("\n")
    print("=============  "+opt_script+"  ==================")
    morphemes = (morphemes_suffixes if "Suffix" in opt_script else morphemes_prefixes)
    with open(glob.glob("../extract/output/extracted_forWords_Sesotho_ExtractOrder_"+("Suf" if "Suffix" in opt_script else "Pre")+"fixes2_ByType.py_*.tsv")[0], "r") as inFile:
        real = [x.split("\t") for x in inFile.read().strip().split("\n")]
    real = [x[0] for x in real if x[0] in morphemes]
    print(real)
    optimized = [y[-3] for y in sorted(resultsByOptScript[opt_script], key=lambda x:x[0][0])[-1:]]
    print(optimized)
    for i in range(len(morphemes)):
        print(" & ".join([str(i+1), names[real[i]]] + [names[x[i][0]] for x in optimized]), "\\\\")
    optimized_errors = [y[-1] for y in sorted(resultsByOptScript[opt_script], key=lambda x:x[0][0])[-1:]]
    errors = defaultdict(int)
    print(optimized_errors[0])
    for error in optimized_errors[0]:
        left, right, freq = error.strip().split(" ")
        if left in names and right in names and left != right:
           key = (names.get(left, "other"), names.get(right, "other"))
        else:
          key = "(other)"
        errors[key] += int(freq)
    print("======================")
    errors = sorted(list(errors.items()), key=lambda x:x[1], reverse=True)
    count = 0
    for error, count in errors:
       if error == "(other)":
          continue
          print("\\multicolumn{2}{c}{(other)}", "&", count, "\\\\") #  + sum([x[1] for x in errors[5:]])
       else:
          count += 1
          print(error[0], "&", error[1], "&",count, "\\\\")
       if count == 4:
          break



