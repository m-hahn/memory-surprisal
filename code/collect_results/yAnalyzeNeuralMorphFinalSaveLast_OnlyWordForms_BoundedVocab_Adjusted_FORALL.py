
import ud_languages


import subprocess
for language in ud_languages.languages:
    print(language)
    subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "yAnalyzeNeuralMorphFinalSaveLast_OnlyWordForms_BoundedVocab_Adjusted.py", "--sortBy", "2", "--horizon", "20", "--language", language, "--restrictToFinished", "True", "--onlyOptimized", "True"])

