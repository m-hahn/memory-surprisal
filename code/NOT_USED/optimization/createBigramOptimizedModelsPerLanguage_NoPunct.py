import os
import subprocess
import sys
languages = set()
keys = {}



path = "/u/scr/corpora/Universal_Dependencies_2.1/ud-treebanks-v2.1/"
files = os.listdir(path)
for name in files:
   if name.startswith("UD_"):
       language = name[name.index("_")+1:]
       if "-" in language:
         language = language[:language.index("-")]
       if language in languages:
         continue
       files2 = os.listdir(path+"/"+name)
       for name2 in files2:
           if name2.endswith("-ud-train.conllu"):
               key = name2.split("-")[0]
               if "_" in key:
                  key = key[:key.index("_")]
               languages.add(language)
               keys[language] = key
               print
               print ". ~/createCPUEnv.sh"
               print " ".join(['python', 'readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py', language, keys[language], "0.001", "0.0001", "0.9", "SGD"])
languages = sorted(list(languages))
print languages

#startAfter = sys.argv[2]
#endBefore = sys.argv[3]
#for language in languages:
#   if language <= startAfter:
#      continue
#   if language >= endBefore:
#      break
#   subprocess.call(['python', 'inferWeightsCrossVariationalAllCorpora.py', language, keys[language])




##
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Korean ko 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Basque eu 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py German de 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Urdu ur 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Hindi hi 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Slovak sk 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Vietnamese vi 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py North_Sami sme 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Serbian sr 0.001 0.0001 0.9 SGD






#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Hebrew he 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Romanian ro 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Finnish fi 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Danish da 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Old_Church_Slavonic cu 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Galician gl 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Swedish sv 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Marathi mr 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Greek el 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Latin la 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Polish pl 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Spanish es 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Kazakh kk 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Arabic ar 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Japanese ja 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Slovenian sl 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Ancient_Greek grc 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Latvian lv 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Swedish_Sign_Language swl 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Coptic cop 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Turkish tr 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Ukrainian uk 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Hungarian hu 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Russian ru 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Italian it 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Chinese zh 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Dutch nl 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Bulgarian bg 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Irish ga 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Norwegian no 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Indonesian id 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Tamil ta 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py French fr 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Belarusian be 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Lithuanian lt 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Afrikaans af 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Persian fa 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Portuguese pt 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Croatian hr 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py English en 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Estonian et 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Gothic got 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Telugu te 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Czech cs 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Catalan ca 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Korean ko 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Basque eu 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py German de 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Urdu ur 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Hindi hi 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Slovak sk 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Vietnamese vi 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py North_Sami sme 0.001 0.0001 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Serbian sr 0.001 0.0001 0.9 SGD








#
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Coptic cop 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Turkish tr 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Ukrainian uk 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Hungarian hu 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Russian ru 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Italian it 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Chinese zh 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Dutch nl 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Bulgarian bg 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Irish ga 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Norwegian no 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Indonesian id 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Tamil ta 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py French fr 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Belarusian be 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Lithuanian lt 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Afrikaans af 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Persian fa 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Portuguese pt 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Croatian hr 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py English en 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Estonian et 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Gothic got 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Telugu te 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Czech cs 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Catalan ca 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Korean ko 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Basque eu 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py German de 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Urdu ur 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Hindi hi 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Slovak sk 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Vietnamese vi 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py North_Sami sme 0.001 0.1 0.9 SGD
#
#. ~/createCPUEnv.sh
#python readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py Serbian sr 0.001 0.1 0.9 SGD

