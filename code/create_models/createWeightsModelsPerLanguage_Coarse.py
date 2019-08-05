languages = set(["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"])

assert len(languages) == 51

for language in ['Afrikaans', 'Amharic-Adap', 'Arabic', 'Armenian-Adap', 'Basque', 'Belarusian', 'Breton-Adap', 'Bulgarian', 'Buryat-Adap', 'Cantonese-Adap', 'Catalan', 'Chinese', 'Croatian', 'Czech', 'Danish', 'Dutch', 'Estonian', 'Faroese-Adap', 'Finnish', 'French', 'German', 'Greek', 'Hebrew', 'Hindi', 'Hungarian', 'Indonesian', 'Irish', 'Japanese', 'Kazakh-Adap', 'Kurmanji-Adap', 'Latvian', 'Lithuanian', 'Marathi', 'Naija-Adap', 'North_Sami', 'Norwegian', 'Persian', 'Polish', 'Romanian', 'Serbian', 'Slovak', 'Slovenian', 'Spanish', 'Swedish', 'Tamil', 'Thai-Adap', 'Turkish', 'Ukrainian', 'Uyghur-Adap', 'Vietnamese', 'Polish-LFG']:
   languages.add(language)


languages = list(languages)

assert len(languages) == len(set(languages))

import os
import random


import subprocess

modelsDir = "/u/scr/mhahn/deps/manual_output_ground_coarse/"

while len(languages) > 0:
  files = os.listdir(modelsDir)
  language = random.choice(languages)
  relevant = [x for x in files if x.startswith(language+"_infer")]
  relevantModelExists = False
  for filename in relevant:
      with open(modelsDir+filename, "r") as inFile:
         header = next(inFile).strip().split("\t")
         line = next(inFile).strip().split("\t")
         counter = int(line[header.index("Counter")])
         print(counter)
         if counter > 1000000:
            relevantModelExists = True
            break

  if relevantModelExists:
     languages.remove(language)
     continue
  subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "inferWeightsCrossVariationalAllCorpora_NoPunct_NEWPYTORCH_Coarse.py", language, language])
#  break

