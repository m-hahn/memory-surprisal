import sys
import os

with open("../../ud_languages.txt", "r") as inFile:
  languages = inFile.read().strip().split("\n")

import random
import subprocess

random.shuffle(languages)

for language in languages:
  if language == "Czech":
     continue
  subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "evaluateOrdering_All.py", language])

