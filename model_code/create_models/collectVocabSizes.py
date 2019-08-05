with open("../../ud_languages.txt", "r") as inFile:
  languages = inFile.read().strip().split("\n")

import subprocess
with open("vocab_sizes.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "OOV_Perc", "VocabSize", "MostFrequentOOVFreq", "HapaxAmongOOV"])
  for language in languages:
    print >> outFile, (subprocess.check_output(["../python27", "vocab_size.py", language])).strip()


