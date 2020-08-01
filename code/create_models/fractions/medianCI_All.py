import sys
trainingSize = sys.argv[1]
import subprocess

# ./python27 yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_HistogramsByMem_All.py > ../results/tradeoff/listener-curve-histogram_byMem.tsv
language = "Czech"
with open("results/tradeoff/listener-curve-ci-median_"+trainingSize+".tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "Type", "Position", "Memory", "MedianEmpirical", "MedianLower", "MedianUpper", "Level"])
  if True:
     print(language)
     print >> outFile, subprocess.check_output(["./python27", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_BinomialTest_Single_MedianCI.py", language, trainingSize]).strip()


