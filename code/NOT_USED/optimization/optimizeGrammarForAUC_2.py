# optimizeGrammarForAUC_2.py
# Too slow, should replace with the original plugin-estimator suffix array

import random
import sys

objectiveName = "LM"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="Japanese_2.4")
parser.add_argument("--model", dest="model", type=str)
parser.add_argument("--alpha", dest="alpha", type=float, default=0.0)
parser.add_argument("--gamma", dest="gamma", type=int, default=1)
parser.add_argument("--delta", dest="delta", type=float, default=1.0)
parser.add_argument("--cutoff", dest="cutoff", type=int, default=3)
parser.add_argument("--idForProcess", dest="idForProcess", type=int, default=random.randint(0,10000000))
import random



args=parser.parse_args()
print(args)


assert args.alpha >= 0
assert args.alpha <= 1
assert args.delta >= 0
assert args.gamma >= 1





myID = args.idForProcess


TARGET_DIR = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"



posUni = set() 

posFine = set() 






from math import log, exp
from random import random, shuffle, randint, Random, choice

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator_V import CorpusIterator_V

originalDistanceWeights = {}

def makeCoarse(x):
   if ":" in x:
      return x[:x.index(":")]
   return x

def initializeOrderTable():
   orderTable = {}
   keys = set()
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in CorpusIterator_V(args.language,partition).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          line["coarse_dep"] = makeCoarse(line["dep"])
          depsVocab.add(line["coarse_dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          if line["coarse_dep"] == "root":
             continue
          posHere = line["posUni"]
          posHead = sentence[line["head"]-1]["posUni"]
          dep = line["coarse_dep"]
          direction = "HD" if line["head"] < line["index"] else "DH"
          key = dep
          keyWithDir = (dep, direction)
          orderTable[keyWithDir] = orderTable.get(keyWithDir, 0) + 1
          keys.add(key)
          distanceCounts[key] = distanceCounts.get(key,0.0) + 1.0
          distanceSum[key] = distanceSum.get(key,0.0) + abs(line["index"] - line["head"])
   #print orderTable
   dhLogits = {}
   for key in keys:
      hd = orderTable.get((key, "HD"), 0) + 1.0
      dh = orderTable.get((key, "DH"), 0) + 1.0
      dhLogit = log(dh) - log(hd)
      dhLogits[key] = dhLogit
      originalDistanceWeights[key] = (distanceSum[key] / distanceCounts[key])
   return dhLogits, vocab, keys, depsVocab



dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()

posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

posFine = list(posFine)
itos_pos_ptb = posFine
stoi_pos_ptb = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

print(itos_deps)




# "linearization_logprobability"
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   # Loop Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum #+ sum(line.get("children_decisions_logprobs",[]))


   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   result.append(line)
   line["relevant_logprob_sum"] = allGradients
   if "children_HD" in line:
      for child in line["children_HD"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   return allGradients

import numpy.random
import numpy as np



def orderSentence(sentence, weights, coordinate, newWeight, printThings, training):
   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0

   for line in sentence:
       line["coarse_dep"] = makeCoarse(line["dep"])
   for j, line in enumerate(sentence):
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      if line["coarse_dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         continue
      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhSampled = (weights[line["coarse_dep"]] < weights["HEAD"])
      direction = "DH" if dhSampled else "HD"
      line["direction"] = direction
      if printThings: 
         print("\t".join(list(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction   ]  ))))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])

   for line in sentence:
      if "children_DH" in line:
        line["children_DH"] = sorted(line["children_DH"], key=lambda x:weights[sentence[x-1]["coarse_dep"]])
      if "children_HD" in line:
        line["children_HD"] = sorted(line["children_HD"], key=lambda x:weights[sentence[x-1]["coarse_dep"]])
    

   
   linearized = []
   logprob_sum = recursivelyLinearize(sentence, root, linearized, None)
   if printThings or len(linearized) == 0:
     print(" ".join(map(lambda x:x["word"], sentence)))
     print(" ".join(map(lambda x:x["word"], linearized)))


   # store new dependency links
   moved = [None] * len(sentence)
   for i, x in enumerate(linearized):
      moved[x["index"]-1] = i
   for i,x in enumerate(linearized):
      if x["head"] == 0: # root
         x["reordered_head"] = 0
      else:
         x["reordered_head"] = 1+moved[x["head"]-1]
   return linearized



print(itos_deps)
weights = itos_deps[::] + ["HEAD"]
shuffle(weights)
weights = dict(zip(weights[::], range(len(weights))))
weights = dict(list(zip(list(weights), [2*x for x in range(len(weights))])))
print(weights)



morphKeyValuePairs = set()

vocab_lemmas = {}

corpusTrain = CorpusIterator_V(args.language,"train", storeMorph=True).iterator(rejectShortSentences = False)
pairs = set()
counter = 0
data = list(corpusTrain)
print(counter)
#print(data)
print(len(data))

#quit()
import torch.nn as nn
import torch
from torch.autograd import Variable

torch.manual_seed(myID)

import os
import numpy.random



def calculateTradeoffForWeights(weights, relevantAffix):
    dev = []
    for _ in range(args.cutoff+2):
      dev.append("PAD")
    dev.append("SOS")
    for sentence in data:
       depOccurs = False
       for line in sentence:
          if line["coarse_dep"] == relevantAffix:
             depOccurs = True
             break
       if not depOccurs:
          continue
       ordered = orderSentence(sentence, weights, None, None, False, False)
       for ch in ordered:
         dev.append(ch["word"])
       dev.append("EOS")
       for _ in range(args.cutoff+2):
         dev.append("PAD")
       dev.append("SOS")
    
    itos = list(set(dev))
    
    
    dev = dev[::-1]
    #dev = list(createStreamContinuous(corpusDev))[::-1]
    
    
    #corpusTrain = CorpusIterator(args.language,"dev", storeMorph=True).iterator(rejectShortSentences = False)
    #train = list(createStreamContinuous(corpusTrain))[::-1]
    train = dev
    
    idev = range(len(dev))
    itrain = range(len(train))
    
    idev = sorted(idev, key=lambda i:dev[i:i+args.cutoff+2])
    itrain = idev #sorted(itrain, key=lambda i:train[i:i+args.cutoff+2])
    
#    print(idev)
    
    idevInv = [x[1] for x in sorted(zip(idev, range(len(idev))), key=lambda x:x[0])]
    itrainInv = idevInv # [x[1] for x in sorted(zip(itrain, range(len(itrain))), key=lambda x:x[0])]
    
    assert idev[idevInv[5]] == 5
    assert itrain[itrainInv[5]] == 5
    
    
    
    def getStartEnd(k):
       start = [0 for _ in dev]
       end = [len(train)-1 for _ in dev]
       if k == 0:
          return start, end
       # Start is the FIRST train place that is >=
       # End is the FIRST train place that is >
       l = 0
       l2 = 0
       for j in range(len(dev)):
         prefix = tuple(dev[idev[j]:idev[j]+k])
         while l2 < len(train):
            prefix2 = tuple(train[itrain[l2]:itrain[l2]+k])
            if prefix <= prefix2:
                 start[j] = l2
                 break
            l2 += 1
         if l2 == len(train):
            start[j] = l2
         while l < len(train):
            prefix2 = tuple(train[itrain[l]:itrain[l]+k])
            if prefix < prefix2:
                 end[j] = l
                 break
            l += 1
         if l == len(train):
            end[j] = l
         start2, end2 = start[j], end[j]
         assert start2 <= end2
         if False and start2 > 0 and end2 < len(train):
           assert prefix > tuple(train[itrain[start2-1]:itrain[start2-1]+k])
           assert prefix <= tuple(train[itrain[start2]:itrain[start2]+k])
           assert prefix >= tuple(train[itrain[end2-1]:itrain[end2-1]+k])
           assert prefix < tuple(train[itrain[end2]:itrain[end2]+k])
       return start, end
    
    
    lastProbability = [None for _ in idev]
    newProbability = [None for _ in idev]
    
    devSurprisalTable = []
    startKLast, endKLast = None, None
    for k in range(0,args.cutoff):
#       print(k)
       if startKLast == None:
          startK, endK = getStartEnd(k) # Possible speed optimization: There is some redundant computation here, could be reused from the previous iteration. But the algorithm is very fast already.
       else:
          startK, endK = startKLast, endKLast
       startK2, endK2 = getStartEnd(k+1)
       startKLast, endKLast = startK2, endK2

       cachedFollowingCounts = {}
       for j in range(len(idev)):
    #      print(dev[idev[j]])
          if dev[idev[j]] in ["PAD", "SOS"]:
             continue
          start2, end2 = startK2[j], endK2[j]
          devPref = tuple(dev[idev[j]:idev[j]+k+1])
          if False and start2 > 0 and end2 < len(train):
            assert devPref > tuple(train[itrain[start2-1]:itrain[start2-1]+k+1]), (devPref, tuple(train[itrain[start2-1]:itrain[start2-1]+k+1]))
            assert devPref <= tuple(train[itrain[start2]:itrain[start2]+k+1]), (devPref, tuple(train[itrain[start2]:itrain[start2]+k+1]))
            assert devPref >= tuple(train[itrain[end2-1]:itrain[end2-1]+k+1])
            assert devPref < tuple(train[itrain[end2]:itrain[end2]+k+1])
    
          assert start2 <= end2
    
          countNgram = end2-start2
          if k >= 1:
             if idev[j]+1 < len(idevInv):
               prefixIndex = idevInv[idev[j]+1]
               assert dev[idev[prefixIndex]] == dev[idev[j]+1]
       
               prefixStart, prefixEnd = startK[prefixIndex], endK[prefixIndex]
               countPrefix = prefixEnd-prefixStart
               if countPrefix < args.gamma: # there is nothing to interpolate with, just back off
                  assert k > 0
                  newProbability[j] = lastProbability[j]
               else:
                  #assert countPrefix >= countNgram, (countPrefix, countNgram)
       
#                  following = set()
                  if (prefixStart, prefixEnd) in cachedFollowingCounts:
                      followingCount = cachedFollowingCounts[(prefixStart, prefixEnd)]
                  else:
                    followingCount = 0
                    for l in range(prefixStart, prefixEnd):
                      if k < itrain[l]+1:
#                         following.add(train[itrain[l]-1])
                         followingCount += 1
                         assert devPref[1:] == tuple(train[itrain[l]-1:itrain[l]+k])[1:], (k, itrain[l], l, devPref , tuple(train[itrain[l]-1:itrain[l]+k]))
 #                   followingCount = len(following)
                    cachedFollowingCounts[(prefixStart, prefixEnd)] = followingCount
                  if followingCount == 0:
                      newProbability[j] = lastProbability[j]
                  else:
                      assert countNgram > 0
                      probability = log(max(countNgram - args.alpha, 0.0) + args.alpha * followingCount * exp(lastProbability[j])) -  log(countPrefix)
                      newProbability[j] = probability
             else:
                newProbability[j] = lastProbability[j]
          elif k == 0:
                  probability = log(countNgram + args.delta) - log(len(train) + args.delta * len(itos))
                  newProbability[j] = probability
       lastProbability = newProbability 
       newProbability = [None for _ in idev]
       assert all([x is None or x <=0 for x in lastProbability])
       try:
           lastProbabilityFiltered = [x for x in lastProbability if x is not None]
           surprisal = - sum([x for x in lastProbabilityFiltered])/len(lastProbabilityFiltered)
       except ValueError:
    #       print >> sys.stderr, "PROBLEM"
     #      print >> sys.stderr, lastProbability
           surprisal = 1000
       devSurprisalTable.append(surprisal)
     #  print("Surprisal", surprisal, len(itos))
    #print(devSurprisalTable)
    mis = [devSurprisalTable[i] - devSurprisalTable[i+1] for i in range(len(devSurprisalTable)-1)]
    tmis = [mis[x]*(x+1) for x in range(len(mis))]
    #print(mis)
    #print(tmis)
    auc = 0
    memory = 0
    mi = 0
    for i in range(len(mis)):
       mi += mis[i]
       memory += tmis[i]
       auc += mi * tmis[i]
    #print("MaxMemory", memory)
    assert 10>memory, memory
    auc += mi * (10-memory)
    #print("AUC", auc)
    return auc
    #assert False
    
    #outpath = TARGET_DIR+"/estimates-"+args.language+"_"+__file__+"_model_"+str(myID)+"_"+args.model+".txt"
    #print(outpath)
    #with open(outpath, "w") as outFile:
    #         print >> outFile, str(args)
    #         print >> outFile, devSurprisalTable[-1]
    #         print >> outFile, " ".join(map(str,devSurprisalTable))
    #
    #
   


for iteration in range(1000):
  coordinate=choice(itos_deps+["HEAD"])
  mostCorrect, mostCorrectValue = 0, None
  for newValue in [-1] + [2*x+1 for x in range(len(weights))] + [weights[coordinate]]:
     if random() < 0.8 and newValue != weights[coordinate]:
        continue
     print(newValue, mostCorrect, coordinate)
     weights_ = {x : y if x != coordinate else newValue for x, y in weights.items()}
     correctCount = calculateTradeoffForWeights(weights_, coordinate)
     if correctCount > mostCorrect:
        mostCorrectValue = newValue
        mostCorrect = correctCount
  print(iteration, mostCorrect)
  weights[coordinate] = mostCorrectValue
  itos_ = sorted(itos+["HEAD"], key=lambda x:weights[x])
  weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))
  print(weights)
  for x in itos_:
     print("\t".join([str(y) for y in [x, weights[x]]]))
#  if (iteration + 1) % 50 == 0:
#     with open(TARGET_DIR+"/optimized_"+__file__+"_"+str(myID)+".tsv", "w") as outFile:
#        print(iteration, mostCorrect, file=outFile)
#        for key in itos_:
#           print(key, weights[key], file=outFile)



