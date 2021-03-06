from ud_languages import languages

import subprocess

# ./python27 yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_HistogramsByMem_All.py > ../results/tradeoff/listener-curve-histogram_byMem.tsv

with open("../../../results/tradeoff/listener-curve-binomial-test-equalOrBigger.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "Type", "Position", "Memory", "BetterEmpirical", "pValue"])
  for language in languages:
     print(language)
     print >> outFile, subprocess.check_output(["./python27", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_BinomialTest_Single_EqualOrBigger.py", "--language", language, "--base_directory", "word-level", "--suffix", "_decay_after_tuning_onlyWordForms_boundedVocab"]).strip()

