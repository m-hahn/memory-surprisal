
# ./python27 yStudyTradeoff_Bootstrap_OnlyWordForms_BoundedVocab_FORALL.py REAL_REAL > tradeoff/listener-curve-onlyWordForms-boundedVocab_REAL.tsv
# ./python27 yStudyTradeoff_Bootstrap_OnlyWordForms_BoundedVocab_FORALL.py GROUND > tradeoff/listener-curve-onlyWordForms-boundedVocab_GROUND.tsv

import sys
realType = sys.argv[1]

languages = []
languages += ["Arabic", "Catalan", "Czech", "Dutch", "Finnish", "French", "German", "Hindi", "Norwegian", "Spanish"]
languages += ["Basque", "Bulgarian", "Croatian", "Estonian", "Hebrew", "Japanese", "Polish", "Romanian", "Slovak", "Slovenian", "Swedish"]
languages += ["Afrikaans", "Chinese", "Danish", "Greek", "Hungarian",  "North_Sami", "Persian", "Serbian", "Turkish", "Ukrainian", "Vietnamese"]
languages += ["Amharic-Adap", "Armenian-Adap",  "Breton-Adap",  "Buryat-Adap", "Cantonese-Adap","Faroese-Adap", "Kazakh-Adap", "Kurmanji-Adap", "Naija-Adap","Thai-Adap", "Uyghur-Adap"]
languages += ["Bambara-Adap", "Erzya-Adap", "Maltese", "Latvian"]

languages += ["Indonesian", "Urdu" , "Portuguese", "English", "Italian", "Russian"]
languages = set(languages)

import subprocess
print "\t".join(["language", "result1Mean", "result2Mean", "result1Low", "result1High", "result2Low", "result2High", "result3Mean", "result3Low", "result3High"])
for language in languages:
    try:
       result = subprocess.check_output(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "/u/scr/mhahn/CODE/deps/yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_Adjusted_Difference.py", language, "RANDOM_BY_TYPE", realType]).strip().split("\n")
       print result[0]    
    except subprocess.CalledProcessError:
       _ = 0

