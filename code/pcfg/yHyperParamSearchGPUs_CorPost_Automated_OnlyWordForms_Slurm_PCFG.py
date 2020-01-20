
# based on https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py


import subprocess
import random
from math import exp


import random

myID = random.randint(0,1000000000)


import sys

gpus = 1
numberOfJobs = 1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--language', type=str)
parser.add_argument('--MODEL_TYPE', type=str, default="RANDOM_BY_TYPE")
parser.add_argument('--noiseVariance', type=float, default=0.0025) 
parser.add_argument('--priorKnowledge', type=str, default=None) # variable
parser.add_argument('--limit', type=int, default=15) # ?
args = parser.parse_args()

if args.priorKnowledge is not None:
   assert args.MODEL_TYPE in args.priorKnowledge
assert args.MODEL_TYPE == "RANDOM_BY_TYPE", "are you sure?"




import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return expected_improvement

n_iters = 10
sample_loss = None


bounds = []
bounds.append(['VOCAB_FOR_RELATION_THRESHOLD', int, 10, 30, 50, 100, 300, 500, 800, 1000, 2000]) # variable
bounds.append(['OTHER_WORDS_SMOOTHING', float, 0.0001])
bounds.append(['MERGE_ACROSS_RELATIONS_THRESHOLD', int, 10, 20, 50, 100, 400, 800, 1000]) 
bounds.append(['REPLACE_WORD_WITH_PLACEHOLDER', float, 0.0, 0.2, 0.4]) 




values = [x[2:] for x in bounds]
names = [x[0] for x in bounds]

import random

def sample():
   while True:
     result = [random.choice(values[i]) for i in range(len(bounds))]
#     if result[names.index("lstm_dim")] == 1024 and result[names.index("layers")] == 3:
#        continue
#     if result[names.index("batch_size")] < 16:
 #       continue
  #   if result[names.index("learning_rate")] > 1.0:
   #     continue
    # if result[names.index("dropout1")] > 0.05:
     #   continue

#     if result[names.index("dropout2")] >= 1.0:
 #       continue

     return result

def represent(x):
   result = [float(values[i].index(x[i]))/len(values[i]) for i in range(len(x))]
   return result
  

n_pre_samples=5
gp_params=None
random_search=False
alpha=args.noiseVariance # 0.0025
epsilon=1e-7

xp_raw = []
y_list = []

if args.priorKnowledge is not None:
  with open(args.priorKnowledge, "r") as inFile:
    for line in inFile:
      line = line.strip().split("\t")
      line[1] = map(float, line[1][1:-1].split(","))
      for y in line[1]:
         if str(y).lower().startswith("n"):
            y = 100
         y_list.append(y)
         xp_raw.append(map(lambda x:float(x) if "." in x else int(x),line[2:]))
print xp_raw

# 4.699497452695695	[4.66408287849234]	0.35	200	128	1	0.005	0.3	18



kernel = gp.kernels.Matern()
model = gp.GaussianProcessRegressor(kernel=kernel,
                                    alpha=alpha,
                                    n_restarts_optimizer=10,
                                    normalize_y=True)

theirGPUs = []
perGPU = ([0]*gpus)
runningProcesses = []
theirIDs = []
theirXPs = []
positionsInXPs = []


IDsForXPs = []

argumentNames = [x[0] for x in bounds]


def extractArguments(x):
   result = []
   for arg in argumentNames:
      result.append("--"+arg)
      result.append(x[names.index(arg)])
   return result

import os
import subprocess

version = "cky_gpu_Stat10_FewNTs_Debug_UD3_GPU_Lexical_Rel_NoSmooth9.py"


parametersWithLargeNTVocab = []

def getResult(i):
   if runningProcesses[i].poll() is not None:
     try:
      with open("/u/scr/mhahn/deps/memory-need-pcfg/estimates-"+args.language+"_"+version+"_model_"+str(theirIDs[i])+"_"+args.MODEL_TYPE+".txt", "r") as inFile:
         next(inFile)
         loss = float(next(inFile))
         return loss
     except IOError:
       print("IO Error", theirIDs[i])
       parametersWithLargeNTVocab.append(theirXPs[i])
       return 10
   else:
      return None

import time

posteriorMeans = []

#for n in range(n_iters):
while True:
    assert len(runningProcesses) == len(theirIDs)
    assert len(runningProcesses) == len(positionsInXPs)
    assert len(runningProcesses) == len(theirXPs)
    assert len(runningProcesses) == len(theirGPUs)
#    print "PROCESSES"
#    print runningProcesses
#    print theirIDs

    canReplace = None
    if len(runningProcesses) >= numberOfJobs: # wait until some process terminates
       for i in range(len(runningProcesses)):
          loss = getResult(i)
          if loss is not None:
              canReplace = i
              y_list[positionsInXPs[i]] = loss
              break
       if canReplace is None:
         print "Sleeping"
         print "/u/scr/mhahn/deps/memory-need-pcfg/search-"+args.language+"_"+version+"_model_"+str(myID)+"_"+args.MODEL_TYPE+".txt"
         time.sleep(5)
         print "Checking again"
         continue
       del runningProcesses[canReplace]
       del theirIDs[canReplace]
       del positionsInXPs[canReplace]
       del theirXPs[canReplace]
       perGPU[theirGPUs[canReplace]] -= 1
       assert perGPU[theirGPUs[canReplace]] >= 0
       del theirGPUs[canReplace]
       print "OBTAINED RESULT"

    if len(posteriorMeans) > 50 and random.random() > 0.95:
       print "Sampling old point, to see whether it really looks good"
#       print posteriorMeans
       nextPoint = random.choice(posteriorMeans[:10])[2]
 #      print nextPoint
  #     quit()
    else:        
#       if len(runningProcesses) < numberOfJobs:
       if random.random() > 0.9 or len(xp_raw) - numberOfJobs < 20: # choose randomly until we have 20 datapoints to base our posterior on
          print "Choose randomly"
          nextPoint = sample()
       else:
          samples = [sample() for _ in range(1000)]
          acquisition = [expected_improvement(np.array(represent(x)), model, 100, False, len(bounds)) for x in samples] 
          best = np.argmax(np.array(acquisition))
          nextPoint = samples[best]

    print "NEW POINT"
    print nextPoint

    mu, sigma = model.predict(np.array(represent(nextPoint)).reshape(-1, len(bounds)), return_std=True)
    print mu
    
    # create an ID for this process, start it
    idForProcess = random.randint(0,1000000000)


    
    my_env = os.environ.copy()

#    quit()
#    subprocess.call(command)
    FNULL = open(os.devnull, "w")
#    p = None
    gpu = np.argmin(perGPU)
    print "GPU "+str(gpu)+" out of "+str(gpus)
    perGPU[gpu] += 1

    command = map(str,["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", version, "--language", args.language, "--model", args.MODEL_TYPE] + extractArguments(nextPoint) + ["--myID", idForProcess])
    print " ".join(command)

    #my_env["CUDA_VISIBLE_DEVICES"] = str(gpus[gpu])
    p = subprocess.Popen(command, stdout=FNULL, env=my_env) # stderr=FNULL, 
    runningProcesses.append(p)
    theirIDs.append(idForProcess)
    theirXPs.append(nextPoint)
    IDsForXPs.append(idForProcess)
    theirGPUs.append(gpu)
    print "ALLOCATED GPUs"
    print theirGPUs
#    sampledResult = 
#    x_to_predict = x.reshape(-1, n_params)
#
    mu, sigma = model.predict(np.array(represent(nextPoint)).reshape(-1, len(bounds)), return_std=True)
    sampledResult = np.random.normal(loc=mu, scale=sigma)


    # Update lists
    positionsInXPs.append(len(xp_raw))
    xp_raw.append(nextPoint)
    y_list.append(sampledResult)

    
    xp_raw_filtered = []
    y_list_filtered = []

    for i in range(len(xp_raw)):
        if i in positionsInXPs:
           continue
        xp_raw_filtered.append(xp_raw[i])
        y_list_filtered.append(y_list[i])
    
    xp_filtered = np.array(map(represent, xp_raw_filtered)).reshape(len(xp_raw_filtered), len(bounds))
    yp_filtered = np.array(y_list_filtered)



    print "USING"
    print xp_raw_filtered
    print xp_filtered
    print IDsForXPs
    print yp_filtered
    if len(xp_raw_filtered) > 0:
       model.fit(xp_filtered, yp_filtered)
     
       # find setting with best posteriori mean
       posteriorMeans = {}
       for i in range(len(xp_raw)):
           if i in positionsInXPs:
              continue
           if str(xp_raw[i]) not in posteriorMeans:
             posteriorMu, posteriorSigma = model.predict(np.array(represent(xp_raw[i])).reshape(-1, len(bounds)), return_std=True)
             # sort by upper 95 \% confidence bound
             posteriorMeans[str(xp_raw[i])] = (posteriorMu[0], [y_list[i]], xp_raw[i], posteriorMu[0]-2*posteriorSigma[0], posteriorMu[0]+2*posteriorSigma[0])
           else:
             posteriorMeans[str(xp_raw[i])][1].append(y_list[i])
       posteriorMeans = [posteriorMeans[x] for x in posteriorMeans]
       posteriorMeans = sorted(posteriorMeans, key=lambda x:x[4]) # sort by upper confidence bound
       print "Best Parameter Settings"
       print posteriorMeans
       print "/u/scr/mhahn/deps/memory-need-pcfg/search-"+args.language+"_"+version+"_model_"+str(myID)+"_"+args.MODEL_TYPE+".txt"
       with open("/u/scr/mhahn/deps/memory-need-pcfg/search-"+args.language+"_"+version+"_model_"+str(myID)+"_"+args.MODEL_TYPE+".txt", "w") as outFile:
          print >> outFile, "\n".join(map(lambda x:"\t".join(map(str,[x[0], x[1]] + x[2])), posteriorMeans))
#       posteriorMeans = sorted(posteriorMeans, key=lambda x:x[3]) # sort by lower confidence bound
       posteriorMeans = sorted(posteriorMeans, key=lambda x:x[0]) # sort by expectation
 
    if len(posteriorMeans) >= args.limit:
        print "/u/scr/mhahn/deps/memory-need-pcfg/search-"+args.language+"_"+version+"_model_"+str(myID)+"_"+args.MODEL_TYPE+".txt"
        break

    xp = np.array(map(represent, xp_raw)).reshape(len(xp_raw), len(bounds))
    yp = np.array(y_list)


    model.fit(xp, yp)


quit()











