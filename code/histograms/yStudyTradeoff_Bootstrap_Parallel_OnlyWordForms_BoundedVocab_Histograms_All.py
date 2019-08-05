from ud_languages import languages

import subprocess

# ./python27 yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_Histograms_All.py > ../results/tradeoff/listener-curve-histogram_byMI.tsv

print "\t".join(["Language", "Type", "MI", "Memory"])
for language in languages:
   subprocess.call(["./python27", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_Histograms.py", language])
