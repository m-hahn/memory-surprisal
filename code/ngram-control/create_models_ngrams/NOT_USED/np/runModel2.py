import subprocess
import sys

def permutations(x):
   a = []
   for y in x:
     for z in x:
       if z == y:
         continue
       for w in x:
         if w in [y,z]:
           continue
         yield "".join([y,z,w])
language = sys.argv[1]
for plugin in [False, True]:
   subprocess.run(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "yWithMorphologySequentialStreamDropoutDev_Ngrams_Log_Restrict_OnlyNPs"+("_Plugin" if plugin else "")+"_2.py", "--language", language, "--model", "GROUND"])
   for order in permutations(["A", "N", "D"]):
      print(order)
      subprocess.run(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "yWithMorphologySequentialStreamDropoutDev_Ngrams_Log_Restrict_OnlyNPs"+("_Plugin" if plugin else "")+"_2.py", "--language", language, "--model", "GROUND_"+order])
