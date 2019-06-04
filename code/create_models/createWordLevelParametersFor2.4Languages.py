languages = []
#languages += ["Arabic", "Catalan", "Czech", "Dutch", "Finnish", "French", "German", "Hindi", "Norwegian", "Spanish"]
#languages.remove("Czech") # for now

# With 5000 as the threshold:
languages += ["Wolof_2.4", "Welsh-Adap_2.4", "Lithuanian_2.4"]

import random
import os
import subprocess
import sys

searchPath = "/u/scr/mhahn/deps/memory-need-neural-wordforms/"

while len(languages) > 0:
   language = random.choice(languages)
   searches = [x for x in os.listdir(searchPath) if x.startswith("search-"+language+"_yWith") and "_OnlyWordForms_BoundedVocab_V.py" in x]
   bestPriorModel = "NONE"
   if len(searches) > 0:
      bestLinesSoFar = 0
      for search in searches:
         with open(searchPath+search, "r") as inFile:
            lines = sum(1 for _ in inFile)
            if lines > bestLinesSoFar:
                bestLinesSoFar = lines
                bestPriorModel = searchPath+search
      if bestLinesSoFar >= 75:
        print("Skip "+language)
        languages.remove(language)
        continue
#   haveFoundGoodCoverage = 
#   for filename in searches:
#      with open(searchPath+"/"+filename, "r") as inFile:
#         inFile = inFile.read().split("\n")
#         if len(inFile) > 
   command = ["./python27", "yHyperParamSearchGPUs_CorPost_Automated_OnlyWordForms_Slurm_BoundedVocab_V.py", language, "1", "2", bestPriorModel, "RANDOM_BY_TYPE", "0.02", "100"]
   print(command)
   p = subprocess.Popen(command,stdout=sys.stdout, stderr=sys.stderr)
   p.wait()



