from ud_languages import languages

import subprocess


with open("../../../../results/tradeoff/listener-curve-auc.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Language", "Type", "ModelID", "AUC"])
  for language in languages:
     print(language)
     print >> outFile, subprocess.check_output(["./python27", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_AUC.py", language]).strip()


