#languages = []
#languages += ["Arabic", "Catalan", "Czech", "Dutch", "Finnish", "French", "German", "Hindi", "Norwegian", "Spanish"]
#languages += ["Basque", "Bulgarian", "Croatian", "Estonian", "Hebrew", "Japanese", "Polish", "Romanian", "Slovak", "Slovenian", "Swedish"]
#languages += ["Afrikaans", "Chinese", "Danish", "Greek", "Hungarian",  "North_Sami", "Persian", "Serbian", "Tamil", "Turkish", "Ukrainian", "Vietnamese"]
#languages += ["Amharic-Adap", "Armenian-Adap",  "Breton-Adap",  "Buryat-Adap", "Cantonese-Adap","Faroese-Adap", "Kazakh-Adap", "Kurmanji-Adap", "Naija-Adap","Thai-Adap", "Uyghur-Adap"]
#languages += ["Bambara-Adap", "Erzya-Adap", "Maltese", "Latvian"]
#
#languages += ["Indonesian", "Urdu", "Portuguese", "English", "Italian"]


	
import random
import os
import subprocess
import sys


searchPath = "/u/scr/mhahn/deps/memory-need-neural-wordforms-fractions/"

for language in ["Czech"]:
 for fraction in [500, 1000, 2000, 5000, 10000, 20000]:
   scriptName = "yCreateRandomModels_"+language+"_OnlyWordForms_BoundedVocab_Fraction_"+str(fraction)+".py"

   files = [x for x in os.listdir("/u/scr/mhahn/CODE/deps/") if x == scriptName]
   if len(files) == 0:
     searches = [x for x in os.listdir(searchPath) if x.startswith("search-"+language+"_yWith") and "_OnlyWordForms_BoundedVocab_Fraction.py_model_"+str(fraction)+"_" in x]
     bestPriorModel = "NONE"
     if len(searches) > 0:
        bestLinesSoFar = 0
        for search in searches:
           print(search)
           with open(searchPath+search, "r") as inFile:
              lines = sum(1 for _ in inFile)
              if lines > bestLinesSoFar:
                  bestLinesSoFar = lines
                  bestPriorModel = searchPath+search
     else:
       print("No search", language, fraction)
       continue
     print(bestPriorModel)
     with open(bestPriorModel, "r") as inFile:
         inFile = [x.split("\t") for x in inFile.read().strip().split("\n")]
         bestChoice = inFile[0]
         for x in inFile[:5]:
             x[1] = [float(y) for y in x[1][1:-1].split(", ")]
             if len(x[1]) > 1:
                bestChoice = x
                break
#         print(language, bestChoice)     
         dropout1 = float(bestChoice[2])
         emb_dim = int(bestChoice[3])
         lstm_dim = int(bestChoice[4])
         layers = int(bestChoice[5])
         learning_rate = float(bestChoice[6])
         dropout2 = float(bestChoice[7])
         batch_size = int(bestChoice[8])
         replaceWordProbability = float(bestChoice[9])
#         print(dropout1, replaceWordProbability)
 #        print(scriptName)
         with open(scriptName, "w") as outFile:
             print >> outFile, "# "+bestPriorModel
             print >> outFile, ""
             print >> outFile, ""
             print >> outFile, "import subprocess"
             print >> outFile, "import random"
             print >> outFile, ""
             print >> outFile, "from math import exp"
             print >> outFile, "import sys"
             print >> outFile, ""
             print >> outFile, "prescribedID = None"
             print >> outFile, "dropout1 = "+str(dropout1)
             print >> outFile, "emb_dim = "+str(emb_dim)
             print >> outFile, "lstm_dim = "+str(lstm_dim)
             print >> outFile, "layers = "+str(layers)
             print >> outFile, ""
             print >> outFile, ""
             print >> outFile, "learning_rate = "+str(learning_rate)
             print >> outFile, "dropout2 = "+str(dropout2)
             print >> outFile, "batch_size = "+str(batch_size)
             print >> outFile, "sequence_length = 20"
             print >> outFile, "input_noising = "+str(replaceWordProbability)
             print >> outFile, ""
             print >> outFile, "fraction = "+str(fraction)
             print >> outFile, "language = '"+language+"'"
             print >> outFile, ""
             print >> outFile, 'for model in ["GROUND", "RANDOM_BY_TYPE", "REAL_REAL"]:'
             print >> outFile, ' for _ in range(10 if model == "RANDOM_BY_TYPE" else 5):'
             print >> outFile, "   command = ['./python27', 'yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_BoundedVocab_Fraction.py', language, language, dropout1, emb_dim, lstm_dim, layers, learning_rate, model, dropout2, batch_size,input_noising,  sequence_length, 'None', 'GPU0', 'False', fraction]"
             print >> outFile, "   command = map(str,command)"
             print >> outFile, "   subprocess.call(command)"
             print >> outFile, "quit()"
#         quit()
