from ud_languages import languages

import os
import random


import subprocess

modelsDir = "/u/scr/mhahn/deps/manual_output_ground_coarse/"
modelsDirOut = "/u/scr/mhahn/deps/manual_output_ground_coarse_final/"

files = os.listdir(modelsDir)
import shutil

for language in languages:
  relevant = [x for x in files if x.startswith(language+"_infer")]
  relevantModelExists = False
  farthestName, farthestCounter = None, 0
  for filename in relevant:
      with open(modelsDir+filename, "r") as inFile:
         header = next(inFile).strip().split("\t")
         line = next(inFile).strip().split("\t")
         counter = int(line[header.index("Counter")])
         print(counter)
         if counter > farthestCounter:
           farthestName = filename
           farthestCounter = counter

  print(farthestName, farthestCounter)
  shutil.copyfile(modelsDir+farthestName, modelsDirOut+farthestName)
