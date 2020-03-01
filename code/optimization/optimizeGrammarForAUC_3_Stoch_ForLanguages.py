from ud_languages import languages
from random import choice
import glob
import subprocess
script = "optimizeGrammarForAUC_3_Stoch.py"
for _ in range(10000):
   language = choice(languages)+"_2.4"
   if len(glob.glob("/u/scr/mhahn/deps/memory-need-ngrams-auc-optimized/optimized_"+language+"_"+script+"_*.tsv")) >= 5:
     continue
   subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", script, "--language", language])


