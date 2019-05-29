languages = []
#languages += ["Arabic", "Catalan", "Czech", "Dutch", "Finnish", "French", "German", "Hindi", "Norwegian", "Spanish"]
#languages.remove("Czech") # for now

# With 5000 as the threshold:
languages += ["Basque", "Bulgarian", "Croatian", "Estonian", "Hebrew", "Japanese", "Polish", "Romanian", "Slovak", "Slovenian", "Swedish"]
languages += ["Afrikaans", "Chinese", "Danish", "Greek", "Hungarian",  "North_Sami", "Persian", "Serbian", "Tamil", "Turkish", "Ukrainian", "Vietnamese"]
languages += ["Amharic-Adap", "Armenian-Adap",  "Breton-Adap",  "Buryat-Adap", "Cantonese-Adap","Faroese-Adap", "Kazakh-Adap", "Kurmanji-Adap", "Naija-Adap","Thai-Adap", "Uyghur-Adap"]

languages += ["Bambara-Adap", "Erzya-Adap", "Maltese", "Latvian"]

# ALSO manually running a Korean BoundedVocab version
languages += ["Indonesian", "Urdu", "Portuguese", "Korean", "English", "German", "Italian"]

import random
import os
import subprocess
import sys

searchPath = "/u/scr/mhahn/deps/memory-need-neural-wordforms/"

while len(languages) > 0:
   language = random.choice(languages)
   searches = [x for x in os.listdir(searchPath) if x.startswith("search-"+language+"_yWith") and "_OnlyWordForms_BoundedVocab.py" in x]
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
   command = ["./python27", "yHyperParamSearchGPUs_CorPost_Automated_OnlyWordForms_Slurm_BoundedVocab.py", language, "1", "2", bestPriorModel, "RANDOM_BY_TYPE", "0.02", "100"]
   print(command)
   p = subprocess.Popen(command,stdout=sys.stdout, stderr=sys.stderr)
   p.wait()



