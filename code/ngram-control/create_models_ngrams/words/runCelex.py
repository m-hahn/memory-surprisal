languages = ["english", "german", "dutch"]
import subprocess
for language in languages:
   for model in ["REAL_REAL"] + ["RANDOM" for _ in range(10)]:
     subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "forWords_Celex.py", "--language", language, "--model", model])

