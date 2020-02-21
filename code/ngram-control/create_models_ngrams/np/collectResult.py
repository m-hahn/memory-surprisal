import os

PATH = "/u/scr/mhahn/deps/memory-need-ngrams-np/"

files = os.listdir(PATH)

with open("results.tsv", "w") as outFile:
  print("\t".join(["Language", "Estimator", "Model", "Distance", "Surprisal"]), file=outFile)
  for name in sorted(files):
    with open(PATH+name, "r") as inFile:
       next(inFile)
       next(inFile)
       surprisals = [float(x) for x in next(inFile).strip().split(" ")]
    plugin = ("Plugin.py" in name)
    name = name[10:].split("_")
    language = name[0]
    model = name[-1][:-4]
    print(language, plugin, model, surprisals)
    for i in range(len(surprisals)):
       print("\t".join([str(x) for x in [language, "Plugin" if plugin else "Heldout", model, i, surprisals[i]]]), file=outFile)
