with open("../../ud_languages.txt", "r") as inFile:
  languages = inFile.read().strip().split("\n")
with open("output/perLanguageMeans_FuncHead.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "Script", "Accuracy", "Surprisal"])
  for language in languages:
    try:
     with open("/u/scr/mhahn/deps/locality_optimized_i1/PER_LANGUAGE_MEANS/"+language+"_"+"FuncHead.txt", "r") as inFile:
      for line in inFile:
         name, accuracy, surprisal = line.strip().split("\t")
         print >> outFile, ("\t".join([language, name, accuracy, surprisal]))
    except IOError:
      _ = 0
  
