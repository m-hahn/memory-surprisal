#from ud_languages import languages

with open("../../ud_languages.txt", "r") as inFile:
  languages = inFile.read().strip().split("\n")

import subprocess

# ./python27 yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_HistogramsByMem_All.py > ../results/tradeoff/listener-curve-histogram_byMem.tsv

with open("../../../results/tradeoff/ngrams/listener-curve-ci-median.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "Type", "Position", "Memory", "MedianEmpirical", "MedianLower", "MedianUpper", "Level"])
  for language in languages:
     print(language)
     print >> outFile, subprocess.check_output(["../python27", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_BinomialTest_Single_MedianCI_Ngrams.py", language]).strip()


