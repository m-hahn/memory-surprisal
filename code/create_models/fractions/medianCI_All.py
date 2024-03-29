import sys
import subprocess

# ./python27 yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_HistogramsByMem_All.py > ../results/tradeoff/listener-curve-histogram_byMem.tsv
for language in ["Czech"]: #, "English", "Spanish", "Russian"]:
 for trainingSize in ["500", "1000", "2000", "5000", "10000"]:
  with open("results/tradeoff/"+language+"_listener-curve-ci-median_"+trainingSize+".tsv", "w") as outFile:
   print >> outFile, "\t".join(["Language", "Type", "Position", "Memory", "MedianEmpirical", "MedianLower", "MedianUpper", "Level"])
   if True:
     print(language)
     print >> outFile, subprocess.check_output(["./python27", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_BinomialTest_Single_MedianCI.py", language, trainingSize]).strip()


