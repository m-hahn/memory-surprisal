from ud_languages import languages

import subprocess

# ./python27 yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_ConditionalMeans_All.py > ../results/tradeoff/listener-curve-means.tsv

print "\t".join(["Language", "Position", "Type", "Memory", "MI_Mean", "MI_SD", "MI_SE"])
for language in languages:
   subprocess.call(["./python27", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_ConditionalMeans.py", language])
