
import subprocess

# ./python27 yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_HistogramsByMem_All.py > ../results/tradeoff/listener-curve-histogram_byMem.tsv
languages = ["Czech-PDT"]

with open("../../../results/tradeoff/listener-curve-interpolated_infostruc.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "Type", "FileName", "Position", "Memory", "Surprisal"])
  for language in languages:
     print(language)
     print >> outFile, subprocess.check_output(["python2", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_Interpolate.py", language]).strip()


