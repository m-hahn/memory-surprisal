from ud_languages import languages

import subprocess

# ./python27 yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_HistogramsByMem_All.py > ../results/tradeoff/listener-curve-histogram_byMem.tsv

with open("../results/tradeoff/listener-curve-binomial-confidence-bound-quantile-05.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "Type", "Position", "LowerConfidenceBound", "Level", "Memory"])
  for language in languages:
     print(language)
     print >> outFile, subprocess.check_output(["./python27", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_BinomialTest_Single_UnimodalBoundOnQuantile_BothDirections_05.py", language]).strip()


