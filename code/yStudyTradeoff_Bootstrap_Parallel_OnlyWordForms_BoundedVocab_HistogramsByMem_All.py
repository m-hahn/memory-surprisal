from ud_languages import languages

import subprocess

# ./python27 yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_HistogramsByMem_All.py > ../results/tradeoff/listener-curve-histogram_byMem.tsv

print "\t".join(["Language", "Type", "Memory", "MI"])
for language in languages:
   subprocess.call(["./python27", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_HistogramsByMem.py", language])
