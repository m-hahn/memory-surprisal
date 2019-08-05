import os

with open("../ud_languages.txt", "r") as inFile:
   languages = inFile.read().strip().split("\n")

import subprocess

for language in languages:
   subprocess.call(["./python27", "yAnalyzeNgrams.py", "1", language])

