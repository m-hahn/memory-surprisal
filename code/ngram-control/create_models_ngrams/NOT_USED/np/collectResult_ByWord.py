import os

PATH = "/u/scr/mhahn/deps/memory-need-ngrams-np/"

files = os.listdir(PATH)

def getMIs(surprisals):
   mis = []
   for i in range(len(surprisals)-1):
     mis.append(surprisals[i] - surprisals[i+1])
   return mis

def getCumSums(mis):
  cumsumsIt, cumsumsTIt = [0], [0]
  for i in range(len(surprisals)-1):
    cumsumsIt.append(mis[i] + cumsumsIt[-1])
    cumsumsTIt.append((i+1) * mis[i] + cumsumsTIt[-1])
  return cumsumsIt, cumsumsTIt

with open("resultsByWord.tsv", "w") as outFile:
  print("\t".join(["Language", "Estimator", "Model", "Distance", "Surprisal", "Script", "Part", "It", "SumIt", "SumTIt"]), file=outFile)
  for name in sorted(files):
    script = name[name.index("yWith"):name.index("_model")]
    plugin = ("Plugin" in name)
    name_ = name[10:].split("_")
    language = name_[0]
    model = name_[-1][:-4]
    with open(PATH+name, "r") as inFile:
       next(inFile)
       next(inFile)
       next(inFile)
       for line in inFile:
         surprisals = line.strip().split(" ")
         if len(surprisals) <= 1:
            continue
         print(language, plugin, model, surprisals, script)
         word = surprisals[0]
         surprisals = [float(x) for x in surprisals[1:]]
         mis = getMIs(surprisals)
         cumsumsIt, cumsumsTIt = getCumSums(mis)
         for i in range(len(surprisals)-1):
            print("\t".join([str(x) for x in [language, "Plugin" if plugin else "Heldout", model, i, surprisals[i], script, word, mis[i], cumsumsIt[i], cumsumsTIt[i]]]), file=outFile)
