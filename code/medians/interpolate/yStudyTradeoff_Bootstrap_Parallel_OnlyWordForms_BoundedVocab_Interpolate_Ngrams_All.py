with open("../../ud_languages.txt", "r") as inFile:
  languages = inFile.read().strip().split("\n")

import subprocess

# ./python27 yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_HistogramsByMem_All.py > ../results/tradeoff/listener-curve-histogram_byMem.tsv

with open("../../../results/tradeoff/ngrams/listener-curve-interpolated.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "Type", "FileName", "Position", "Memory", "Surprisal"])
  for language in languages:
     print(language)
     print >> outFile, subprocess.check_output(["../python27", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_Interpolate_Ngrams.py", language]).strip()


