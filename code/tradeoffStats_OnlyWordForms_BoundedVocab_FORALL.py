import sys

#python ~/scr/CODE/deps/yStudyTradeoffPerLang_FORALL.py RANDOM_BY_TYPE > ~/scr/tradeoff-perLanguage-RANDOM_BY_TYPE.tsv
#python ~/scr/CODE/deps/yStudyTradeoffPerLang_FORALL.py RANDOM_MODEL > ~/scr/tradeoff-perLanguage-RANDOM_MODEL.tsv
from ud_languages import languages


   

import subprocess
print("\t".join(["Language", "REAL_REAL", "RANDOM_BY_TYPE"]))
for language in languages:
    print(subprocess.check_output(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "tradeoffStats_OnlyWordForms_BoundedVocab.py", language]).strip())

