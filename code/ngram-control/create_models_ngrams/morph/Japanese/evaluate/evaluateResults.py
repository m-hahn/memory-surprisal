

with open("results.tsv", "r") as inFile:
    data = [x.split("\t") for x in inFile.read().strip().split("\n")]
    header = data[0]
    header = dict(zip(header, range(len(header))))
    data = data[1:]
#print(header)
#print(data)


import os
files = os.listdir("/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/")

for line in data:
    line.append("RANDOM" if line[header["Model"]] == "RANDOM" else "OPTIM")
    relevant = [x for x in files if "_"+line[header["Model"]]+".tsv" in x]
    print(line[header["Model"]])
    if line[header["Model"]] == "2980412":
     print(relevant)

    if len(relevant) == 0:
       assert line[-1] == "RANDOM"
       line.append("both")
    else:
       print(relevant)
#    print(line)

       if "FormsPhonemesFull" in relevant[0] and "suru" not in relevant[0]:
           line.append("phonemes")
       elif ("Normalized" in relevant[0] and "suru" not in relevant[0]) or "MorphemeGrammar_FullData.py" in relevant[0]:
           line.append("morphemes")
       else:
           line.append("neither")
       if "MorphemeGrammar" in relevant[0]:
           print("MORPHEME")
           line[-1] += "_m"

def mean(x):
    return sum(x)/len(x)

from math import sqrt

def sd(x):
    return sqrt(mean([y**2 for y in x]) - mean(x)**2)

scripts = sorted(list(set([x[-1] for x in data])))


results = {}
for script in scripts:
    data_ = [x for x in data if x[-1] == script]
    #print(data_)
  #  print(script)
    for typ in ["RANDOM", "OPTIM"]:
#        print(typ)
        data__ = [x for x in data_ if x[-2] == typ]
        #print(data__)
        accuracy_pairs = [float(x[header["Accuracy_Pairs"]]) for x in data__]
 #       print(accuracy_pairs)
        accuracy_full = [float(x[header["Accuracy_Full"]]) for x in data__]
  #      print(accuracy_full)
        if len(data__) == 0:
           continue
        results[(script, typ)] = ("".join([str(x) for x in [round(mean(accuracy_pairs),3), " (SD ", round(sd(accuracy_pairs), 3), ") & ", round(mean(accuracy_full), 3), " (SD ", round(sd(accuracy_full), 3), ")"]]))

print(results)

print("Phonemes   &   "+results[("phonemes", "OPTIM")]+" \\\\")
print("Morphemes  &   "+results[("morphemes", "OPTIM")]+" \\\\")
print("Random     &  "+results[("both", "RANDOM")]+" \\\\")
print("Phonemes (m)   &   "+results[("phonemes_m", "OPTIM")]+" \\\\")
print("Morphemes (m)  &   "+results[("morphemes_m", "OPTIM")]+" \\\\")
print("Random     &  "+results[("both_m", "RANDOM")]+" \\\\")

