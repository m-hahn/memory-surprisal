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

print([x[1] for x in list(results) if x[1] != "RANDOM"])
#quit()

optimPhonPref = "forWords_Sesotho_OptimizeOrder_FormsWordsGraphemes_Prefixes_ByType.py"
optimPhonSuff = "forWords_Sesotho_OptimizeOrder_FormsWordsGraphemes_Suffixes_ByType.py"
optimMorphPref = "forWords_Sesotho_OptimizeOrder_Normalized_ByType_Prefixes.py"
optimMorphSuff = "forWords_Sesotho_OptimizeOrder_Normalized_ByType_Suffixes.py"
optimPhonPref_HeldoutClip = "forWords_Sesotho_OptimizeOrder_FormsWordsGraphemes_Prefixes_ByType_HeldoutClip.py"
optimPhonSuff_HeldoutClip = "forWords_Sesotho_OptimizeOrder_FormsWordsGraphemes_Suffixes_ByType_HeldoutClip.py"
optimMorphPref_HeldoutClip = "forWords_Sesotho_OptimizeOrder_Normalized_ByType_Prefixes_HeldoutClip.py"
optimMorphSuff_HeldoutClip = "forWords_Sesotho_OptimizeOrder_Normalized_ByType_Suffixes_HeldoutClip.py"


 
print("")      
print("")      
print("")      
print("")      
print("")      

     



print("For main paper:")
print("Sesotho &   Optimized  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", optimMorphPref)]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", optimMorphSuff)]+" \\\\")
print("           &   Baseline  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", "RANDOM")]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", "RANDOM")]+" \\\\")

print("")
print("")
print("")
print("")
print("")
print("For SI:")
print("Tok. & Naive & Phon. &   Opt.  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_ByType.py", optimPhonPref)]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", optimPhonSuff)]+" \\\\")
print("     &       &       &   Rand. &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_ByType.py", "RANDOM")]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", "RANDOM")]+" \\\\")
print("     &       & Morph.&   Opt.  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", optimMorphPref)]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", optimMorphSuff)]+" \\\\")
print("     &       &       &   Rand.  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", "RANDOM")]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", "RANDOM")]+" \\\\ \\hline")
print("     & Heldout & Phon.   &   Opt.  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_ByType.py", optimPhonPref_HeldoutClip)]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", optimPhonSuff_HeldoutClip)]+" \\\\")
print("     &         &         &   Rand.  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_ByType.py", "RANDOM")]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", "RANDOM")]+" \\\\")
print("     &          & Morph. &   Opt.  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", optimMorphPref_HeldoutClip)]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", optimMorphSuff_HeldoutClip)]+" \\\\")
print("     &          &        &   Rand.  &  "+results[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", "RANDOM")]+" & "+results[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", "RANDOM")]+" \\\\ \\hline \\hline")
print("Typ. & Naive   & Phon. &   Opt.&  "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Prefixes_ByType.py", optimPhonPref)]+" & "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", optimPhonSuff)]+" \\\\")
print("     &       &       &  Rand.  &  "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Prefixes_ByType.py", "RANDOM")]+" & "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", "RANDOM")]+" \\\\")
print("     &       & Morph.&   Opt.  &  "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", optimMorphPref)]+" & "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", optimMorphSuff)]+" \\\\")
print("     &       &       &   Rand.  &  "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", "RANDOM")]+" & "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", "RANDOM")]+" \\\\ \\hline")
print("     &Heldout  & Phon.   &   Opt.  &  "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Prefixes_ByType.py", optimPhonPref_HeldoutClip)]+" & "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", optimPhonSuff_HeldoutClip)]+" \\\\")
print("     &         &         &   Rand.  &  "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Prefixes_ByType.py", "RANDOM")]+" & "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Suffixes_ByType.py", "RANDOM")]+" \\\\")
print("     &         &Morph.  &   Opt.  &  "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", optimMorphPref_HeldoutClip)]+" & "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", optimMorphSuff_HeldoutClip)]+" \\\\")
print("     &         &           &   Rand.  &  "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Prefixes_Normalized_ByType.py", "RANDOM")]+" & "+resultsTypes[("forWords_Sesotho_EvaluateWeights_Suffixes_Normalized_ByType.py", "RANDOM")]+" \\\\")

