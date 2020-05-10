import collections

with open("../tradeoffs/optimizedModels.tsv", "r") as inFile:
    optimModels = [x.split("\t") for x in inFile.read().strip().split("\n")]
    header_optimModels = optimModels[0]
    header_optimModels = dict(zip(header_optimModels, range(len(header_optimModels))))
    optimModels = optimModels[1:]
optimModels = dict([x[::-1] for x in optimModels])

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

    line.append("RANDOM" if line[header["Model"]] == "RANDOM" else "OPTIM")

def mean(x):
    return sum(x)/len(x)

from math import sqrt

def sd(x):
    return sqrt(mean([y**2 for y in x]) - mean(x)**2+1e-10)

results = {}
resultsTypes = {}
for script in scripts:
    data_ = [x for x in data if x[header["Script"]] == script]
   # print(data_)
  #  print(script)
    for typ in ["RANDOM", "OPTIM"]:
#        print(typ)
        data__ = [x for x in data_ if x[header["Type"]] == typ]
        data___ = collections.defaultdict(list)
        optimScripts = []
        for line in data__:
            modelHere = (line[header["Model"]])
            if modelHere in optimModels:
               optimScript = optimModels[modelHere]
            elif modelHere == "RANDOM":
               optimScript = "RANDOM"
            else:
                print("UNKNOWN MODEL", line)
                continue
            data___[optimScript].append(line)

        #print(data__)
        for optimScript in data___:
          data_2 = data___[optimScript]
          accuracy_pairs = [float(x[header["Accuracy_Pairs"]]) for x in data_2]
          accuracy_full = [float(x[header["Accuracy_Full"]]) for x in data_2]
          if len(accuracy_full) == 0 or len(accuracy_pairs) == 0:
             print(script, typ)
             continue
          print("datapoints", len(data_2))
          results[(script, optimScript)] = ("".join([str(x) for x in [round(mean(accuracy_pairs),3), " (SD ", round(sd(accuracy_pairs), 3), ") & ", round(mean(accuracy_full), 3), " (SD ", round(sd(accuracy_full), 3), ")"]]))
          accuracy_pairs_optimScriptes = [float(x[header["Accuracy_Pairs_Types"]]) for x in data_2]
          accuracy_full_optimScriptes = [float(x[header["Accuracy_Full_Types"]]) for x in data_2]
          resultsTypes[(script, optimScript)] = ("".join([str(x) for x in [round(mean(accuracy_pairs_optimScriptes),3), " (SD ", round(sd(accuracy_pairs_optimScriptes), 3), ") & ", round(mean(accuracy_full_optimScriptes), 3), " (SD ", round(sd(accuracy_full_optimScriptes), 3), ")"]]))

print([x for x in list(results) if x[1] != "RANDOM"])

optimPhon = "forWords_Japanese_OptimizeOrder_MorphemeGrammar_FormsPhonemesFull_FullData.py"
optimMorph = "forWords_Japanese_OptimizeOrder_MorphemeGrammar_Normalized_FullData.py"
optimPhon_HeldoutClip = "forWords_Japanese_OptimizeOrder_MorphemeGrammar_FormsPhonemesFull_FullData_HeldoutClip.py"
optimMorph_HeldoutClip = "forWords_Japanese_OptimizeOrder_MorphemeGrammar_Normalized_FullData_HeldoutClip.py"

evalScript = "forWords_Celex_EvaluateWeights_MorphemeGrammar_FullData.py"


#Optimized for Phoneme Prediction   &   0.976 (SD 0.007) & 0.971 (SD 0.011) \\
#Optimized for Morpheme Prediction  &   0.873 (SD 0.154) & 0.85 (SD 0.184) \\
#Random Baseline     &  0.519 (SD 0.177) & 0.559 (SD 0.202) \\


print("Optimized for Phoneme Prediction   &   "+results[(evalScript, optimPhon)]+" \\\\")
print("Optimized for Morpheme Prediction  &   "+results[(evalScript, optimMorph)]+" \\\\")
print("Random Baseline    &  "+results[(evalScript, "RANDOM")]+" \\\\")
print("")
print("")
print("")
print("")
print("Optimized for Phoneme Prediction   &   "+resultsTypes[(evalScript, optimPhon)]+" \\\\")
print("Optimized for Morpheme Prediction  &   "+resultsTypes[(evalScript, optimMorph)]+" \\\\")
print("Random Baseline    &  "+resultsTypes[(evalScript, "RANDOM")]+" \\\\")
print("")
print("")
print("")
print("")
print("Optimized for Phoneme Prediction   &   "+results[(evalScript, optimPhon_HeldoutClip)]+" \\\\")
print("Optimized for Morpheme Prediction  &   "+results[(evalScript, optimMorph_HeldoutClip)]+" \\\\")
print("Random Baseline    &  "+results[(evalScript, "RANDOM")]+" \\\\")
print("")
print("")
print("")
print("")
print("Optimized for Phoneme Prediction   &   "+resultsTypes[(evalScript, optimPhon_HeldoutClip)]+" \\\\")
print("Optimized for Morpheme Prediction  &   "+resultsTypes[(evalScript, optimMorph_HeldoutClip)]+" \\\\")
print("Random Baseline    &  "+resultsTypes[(evalScript, "RANDOM")]+" \\\\")

