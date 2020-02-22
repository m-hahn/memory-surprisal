import os

PATH = "/u/scr/mhahn/deps/memory-need-ngrams-np/"

files = os.listdir(PATH)

with open("results.tsv", "w") as outFile:
  print("\t".join(["Language", "Estimator", "Model", "Distance", "Surprisal", "Script"]), file=outFile)
  for name in sorted(files):
    with open(PATH+name, "r") as inFile:
       next(inFile)
       next(inFile)
       surprisals = [float(x) for x in next(inFile).strip().split(" ")]
    script = name[name.index("yWith"):name.index("_model")]
    plugin = ("Plugin" in name)
    name = name[10:].split("_")
    language = name[0]
    model = name[-1][:-4]
    print(language, plugin, model, surprisals, script)
    for i in range(len(surprisals)):
       print("\t".join([str(x) for x in [language, "Plugin" if plugin else "Heldout", model, i, surprisals[i], script]]), file=outFile)
