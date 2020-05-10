with open("hyperparameters.tsv") as inFile:
   data = [x.split("\t") for x in inFile.read().strip().split("\n")]
header = data[0]
data = data[1:]

import random
import glob
import subprocess
random.shuffle(data)

for row in data:
   language = row[0]
#   if language in ["Estonian"]: # skip those with _ words for now. also, some problem with Estonian
 #     continue
   for model in ["GROUND", "RANDOM_BY_TYPE"]:
      count = {"GROUND" : 1, "RANDOM_BY_TYPE" : 5}[model]
      while True:
        models = glob.glob("/u/scr/mhahn/deps/memory-need-pcfg/optimized/estimates-"+language+"_2.4_cky_gpu_Stat10_FewNTs_Debug_UD3_GPU_Lexical_Rel_NoSmooth9.py_model_*_"+model+".txt")
        if len(models) >= count:
            break
        arguments = []
        for i in range(2, len(header)):
           arguments.append("--"+header[i])
           arguments.append(row[i])
        command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "cky_gpu_Stat10_FewNTs_Debug_UD3_GPU_Lexical_Rel_NoSmooth9.py", "--language", language+"_2.4", "--model", model, "--saveOptimized", "True"] + arguments
        print(command)
        subprocess.call(command)

