import sys
import subprocess
import os
import random

language = sys.argv[1]

models = []
#DLM
import glob

dlm = glob.glob("/u/scr/mhahn/deps/locality_optimized_dlm/manual_output_funchead_coarse_depl/"+language+"_optimizeDependencyLength.py_model_*.tsv")
i1  = glob.glob("/u/scr/mhahn/deps/locality_optimized_i1/"+language+"_optimizeGrammarForI1_*.py_model_*.tsv")
neural = glob.glob("/u/scr/mhahn/deps/locality_optimized_neural/manual_output_funchead_langmod_coarse_best_ud/"+language+"_optimizePredictability_OnlyWords.py_model_*.tsv")
ground = glob.glob("~/scr/CODE/memory-surprisal/results/manual_output_ground_coarse/"+language+"_inferWeightsCrossVariationalAllCorpora_NoPunct_NEWPYTORCH_Coarse.py_model_*.tsv")
models = dlm + i1 + neural + ground
print(models)
#quit()



version = "yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py"
argumentNames = ["alpha", "gamma", "delta", "cutoff"]


BASE_DIR = "/u/scr/mhahn/deps/memory-need-ngrams/"

for model in models:


    filenames = ([x for x in os.listdir(BASE_DIR) if x.startswith("search-"+language+"_yWithMo") and len(open(BASE_DIR+x, "r").read().split("\n"))>=30])
    if len(filenames) == 0:
       assert False
#    assert len(filenames) == 1
    with open(BASE_DIR+filenames[0], "r") as inFile:
        params = next(inFile).strip().split("\t")[2:]
        assert len(params) == 4
        params[-1] = 10
    params2 = []
    for i in range(len(params)):
      params2.append("--"+argumentNames[i])
      params2.append(params[i])
    paramsString = " ".join([str(x) for x in params2])
#    command = ["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "yHyperParamSearchGPUs_CorPost_Automated_OnlyWordForms_Slurm_Ngrams.py", language, "1", "2", "NONE", "RANDOM_BY_TYPE", "0.02", "30"]
    command = map(str,["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", version, "--language", language, "--model", model] + params2)
    print(command)
#    quit()
    result = subprocess.call(command)

