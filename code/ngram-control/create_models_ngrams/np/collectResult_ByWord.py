import os

PATH = "/u/scr/mhahn/deps/memory-need-ngrams-np/"

files = os.listdir(PATH)

with open("resultsByWord.tsv", "w") as outFile:
  print("\t".join(["Language", "Estimator", "Model", "Distance", "Surprisal", "Script", "Part"]), file=outFile)
  for name in sorted(files):
    script = name[name.index("yWith"):name.index("_model")]
    plugin = ("Plugin" in name)
    name_ = name[10:].split("_")
    language = name[0]
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
         surprisals = surprisals[1:]
         for i in range(len(surprisals)):
            print("\t".join([str(x) for x in [language, "Plugin" if plugin else "Heldout", model, i, surprisals[i], script, word]]), file=outFile)
