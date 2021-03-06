import sys

from ud_languages import languages


# ./python27 tradeoffStats_OnlyWordForms_BoundedVocab_FORALL.py REAL_REAL > ../results/tradeoff/stats-onlyWordForms-boundedVocab_REAL.tsv
# ./python27 tradeoffStats_OnlyWordForms_BoundedVocab_FORALL.py GROUND > ../results/tradeoff/stats-onlyWordForms-boundedVocab_GROUND.tsv

real_type = sys.argv[1] 

import subprocess
with open("../results/tradeoff/stats-onlyWordForms-boundedVocab_"+("REAL" if real_type == "REAL_REAL" else "GROUND")+".tsv", "w") as outFile:
   print >> outFile, ("\t".join(["Language", real_type, "RANDOM_BY_TYPE"]))
   for language in languages:
       print >> outFile, (subprocess.check_output(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "tradeoffStats_OnlyWordForms_BoundedVocab.py", language, real_type]).strip())

