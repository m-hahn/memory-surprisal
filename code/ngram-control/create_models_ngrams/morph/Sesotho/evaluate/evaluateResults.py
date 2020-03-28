

with open("results.tsv", "r") as inFile:
    data = [x.split("\t") for x in inFile.read().strip().split("\n")]
    header = data[0]
    header = dict(zip(header, range(len(header))))
    data = data[1:]
#print(header)
#print(data)
header["Type"] = len(header)

scripts = sorted(list(set([x[header["Script"]] for x in data])))
print(scripts)

for line in data:
#    print(line[header["Model"]])
    line.append("RANDOM" if line[header["Model"]] == "RANDOM" else "OPTIM")

def mean(x):
    return sum(x)/len(x)

from math import sqrt

def sd(x):
    return sqrt(mean([y**2 for y in x]) - mean(x)**2)

results = {}
resultsTypes = {}
for script in scripts:
    data_ = [x for x in data if x[header["Script"]] == script]
   # print(data_)
  #  print(script)
    for typ in ["RANDOM", "OPTIM"]:
#        print(typ)
        data__ = [x for x in data_ if x[header["Type"]] == typ]
        #print(data__)
        accuracy_pairs = [float(x[header["Accuracy_Pairs"]]) for x in data__]
        accuracy_full = [float(x[header["Accuracy_Full"]]) for x in data__]
        if len(accuracy_full) == 0 or len(accuracy_pairs) == 0:
           print(script, typ)
           continue
        results[(script, typ)] = ("".join([str(x) for x in [round(mean(accuracy_pairs),3), " (SD ", round(sd(accuracy_pairs), 3), ") & ", round(mean(accuracy_full), 3), " (SD ", round(sd(accuracy_full), 3), ")"]]))
        accuracy_pairs_types = [float(x[header["Accuracy_Pairs_Types"]]) for x in data__]
        accuracy_full_types = [float(x[header["Accuracy_Full_Types"]]) for x in data__]
        resultsTypes[(script, typ)] = ("".join([str(x) for x in [round(mean(accuracy_pairs_types),3), " (SD ", round(sd(accuracy_pairs_types), 3), ") & ", round(mean(accuracy_full_types), 3), " (SD ", round(sd(accuracy_full_types), 3), ")"]]))

print(results)

print("Phonemes   &   Optimized  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_ByType.py", "OPTIM")]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", "OPTIM")]+" \\\\")
print("           &   Random  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_ByType.py", "RANDOM")]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", "RANDOM")]+" \\\\")
print("Morphemes  &   Optimized  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", "OPTIM")]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", "OPTIM")]+" \\\\")
print("           &   Random  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", "RANDOM")]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", "RANDOM")]+" \\\\")
print("")
print("")
print("")
print("Phonemes   &   Optimized  &  "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Prefixes_ByType.py", "OPTIM")]+" & "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", "OPTIM")]+" \\\\")
print("           &   Random  &  "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Prefixes_ByType.py", "RANDOM")]+" & "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", "RANDOM")]+" \\\\")
print("Morphemes  &   Optimized  &  "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", "OPTIM")]+" & "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", "OPTIM")]+" \\\\")
print("           &   Random  &  "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", "RANDOM")]+" & "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", "RANDOM")]+" \\\\")

