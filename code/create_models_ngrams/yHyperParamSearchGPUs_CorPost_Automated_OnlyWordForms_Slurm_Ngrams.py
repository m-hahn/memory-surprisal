
# based on https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py


import subprocess
import random
from math import exp


import random

myID = random.randint(0,1000000000)


import sys

language = sys.argv[1]
gpus = int(sys.argv[2]) #map(int,sys.argv[2].split(","))
assert gpus == 1

numberOfJobs = int(sys.argv[3])

priorKnowledge = sys.argv[4] if len(sys.argv)>4 else None
if priorKnowledge == "NONE":
   priorKnowledge = None

MODEL_TYPE = sys.argv[5] if len(sys.argv)>5 else  "RANDOM_BY_TYPE"
assert MODEL_TYPE == "RANDOM_BY_TYPE", "are you sure?"

noiseVariance = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0025

if priorKnowledge is not None:
   assert MODEL_TYPE in priorKnowledge

limit = int(sys.argv[7]) if len(sys.argv) > 7 else 100

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
bounds.append(["alpha", float, 0.95, 1.0]) # + [x/20.0 for x in range(15, 21)])
bounds.append(["gamma", int, 1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30]) # , 200, 300
bounds.append(["delta", int, 0.1, 0.2, 0.5, 1.0 , 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]) #, 1024]) #, 1024]) # 64, 128, 
bounds.append(["cutoff", int, 2,3,4,5,6,7,8,9,10]) #,7,8,9,10]) #, 1024]) #, 1024]) # 64, 128, 


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
alpha=noiseVariance # 0.0025
epsilon=1e-7

xp_raw = []
y_list = []

if priorKnowledge is not None:
  with open(priorKnowledge, "r") as inFile:
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

argumentNames = ["alpha", "gamma", "delta", "cutoff"]

#bounds.append(["alpha", float] + [x/20.0 for x in range(21)])
#bounds.append(["gamma", int, 1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30]) # , 200, 300
#bounds.append(["delta", int, 0.1, 0.2, 0.5, 1.0 , 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]) #, 1024]) #, 1024]) # 64, 128, 
#bounds.append(["cutoff", int, 2,3,4,5,6,7,8,9,10]) #, 1024]) #, 1024]) # 64, 128, 


def extractArguments(x):
   result = []
   for arg in argumentNames:
      result.append("--"+arg)
      result.append(x[names.index(arg)])
   return result

import os
import subprocess

version = "yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py"

def getResult(i):
   if runningProcesses[i].poll() is not None:
      with open("/u/scr/mhahn/deps/memory-need-ngrams/estimates-"+language+"_"+version+"_model_"+str(theirIDs[i])+"_"+MODEL_TYPE+".txt", "r") as inFile:
         next(inFile)
         loss = float(next(inFile))
         return loss
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
         print "/u/scr/mhahn/deps/memory-need-ngrams/search-"+language+"_"+version+"_model_"+str(myID)+"_"+MODEL_TYPE+".txt"
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

    command = map(str,["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", version, "--language", language, "--model", MODEL_TYPE] + extractArguments(nextPoint) + ["--idForProcess", idForProcess])
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
       print "/u/scr/mhahn/deps/memory-need-ngrams/search-"+language+"_"+version+"_model_"+str(myID)+"_"+MODEL_TYPE+".txt"
       with open("/u/scr/mhahn/deps/memory-need-ngrams/search-"+language+"_"+version+"_model_"+str(myID)+"_"+MODEL_TYPE+".txt", "w") as outFile:
          print >> outFile, "\n".join(map(lambda x:"\t".join(map(str,[x[0], x[1]] + x[2])), posteriorMeans))
#       posteriorMeans = sorted(posteriorMeans, key=lambda x:x[3]) # sort by lower confidence bound
       posteriorMeans = sorted(posteriorMeans, key=lambda x:x[0]) # sort by expectation
 
    if len(posteriorMeans) >= limit:
        print "/u/scr/mhahn/deps/memory-need-ngrams/search-"+language+"_"+version+"_model_"+str(myID)+"_"+MODEL_TYPE+".txt"
        break

    xp = np.array(map(represent, xp_raw)).reshape(len(xp_raw), len(bounds))
    yp = np.array(y_list)


    model.fit(xp, yp)


quit()











