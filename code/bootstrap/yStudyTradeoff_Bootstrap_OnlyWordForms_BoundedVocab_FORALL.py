
# ./python27 yStudyTradeoff_Bootstrap_OnlyWordForms_BoundedVocab_FORALL.py REAL_REAL > tradeoff/listener-curve-onlyWordForms-boundedVocab_REAL.tsv
# ./python27 yStudyTradeoff_Bootstrap_OnlyWordForms_BoundedVocab_FORALL.py GROUND > tradeoff/listener-curve-onlyWordForms-boundedVocab_GROUND.tsv

import sys
realType = sys.argv[1]

with open("../../ud_languages.txt", "r") as inFile:
   languages = inFile.read().strip().split("\n")

languages = set(languages)

import subprocess
outPath = "../../results/tradeoff/listener-curve-onlyWordForms-boundedVocab_"+("REAL" if realType == "REAL_REAL" else "GROUND")+".tsv"
print(outPath)
with open(outPath, "w") as outFile:
   print >> outFile, "\t".join(["language", "result1Mean", "result2Mean", "result1Low", "result1High", "result2Low", "result2High", "result3Mean", "result3Low", "result3High"])
   for language in languages:
       try:
          result = subprocess.check_output(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_Adjusted_Difference.py", language, "RANDOM_BY_TYPE", realType]).strip().split("\n")
          print >> outFile, result[0]    
       except subprocess.CalledProcessError:
          _ = 0
   
