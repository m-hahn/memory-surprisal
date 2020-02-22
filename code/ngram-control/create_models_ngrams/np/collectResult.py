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

with open("results.tsv", "w") as outFile:
  print("\t".join(["Language", "Estimator", "Model", "Distance", "Surprisal", "Script", "Part", "It", "SumIt", "SumTIt"]), file=outFile)
  for name in sorted(files):
    with open(PATH+name, "r") as inFile:
       next(inFile)
       next(inFile)
       surprisals = [float(x) for x in next(inFile).strip().split(" ")]
    mis = getMIs(surprisals)
    cumsumsIt, cumsumsTIt = getCumSums(mis)
    script = name[name.index("yWith"):name.index("_model")]
    plugin = ("Plugin" in name)
    name = name[10:].split("_")
    language = name[0]
    model = name[-1][:-4]
    print(language, plugin, model, surprisals, script)
    for i in range(len(surprisals)-1):
       print("\t".join([str(x) for x in [language, "Plugin" if plugin else "Heldout", model, i, surprisals[i], script, mis[i], cumsumsIt[i], cumsumsTIt[i]]]), file=outFile)
