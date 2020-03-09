# optimizeGrammarForAUC_3.py
# Utilizing /juicier/scr120/scr/mhahn/CODE/deps/ConditionalMIDecay/computeConditionalMIDecay_Deterministic_Words_Generalized.py

# Seems to work well, optimizes AUC.

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
parser.add_argument("--cutoff", dest="cutoff", type=int, default=5)
parser.add_argument("--idForProcess", dest="idForProcess", type=int, default=random.randint(0,10000000))
import random



args=parser.parse_args()
print(args)


assert args.alpha >= 0
assert args.alpha <= 1
assert args.delta >= 0
assert args.gamma >= 1





myID = args.idForProcess


TARGET_DIR = "/u/scr/mhahn/deps/memory-need-ngrams-auc-optimized/"



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

words = list(vocab)
itos_words = ["PAD", "SOS", "EOS"] + words
stoi_words = dict(zip(itos_words, range(len(itos_words))))


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

if True:
  dhWeights = {}
  distanceWeights = {}
  groundPath = "/u/scr/mhahn/deps/manual_output_ground_coarse/"
  import os
  files = [x for x in os.listdir(groundPath) if x.startswith(args.language[:args.language.rfind("_")]+"_infer")]
  print(files)
  assert len(files) > 0
  with open(groundPath+files[0], "r") as inFile:
     headerGrammar = next(inFile).strip().split("\t")
     print(headerGrammar)
     dhByDependency = {}
     distByDependency = {}
     for line in inFile:
         line = line.strip().split("\t")
         assert int(line[headerGrammar.index("Counter")]) >= 1000000
#         if line[headerGrammar.index("Language")] == language:
#           print(line)
         dependency = line[headerGrammar.index("Dependency")]
         dhHere = float(line[headerGrammar.index("DH_Mean_NoPunct")])
         distHere = float(line[headerGrammar.index("Distance_Mean_NoPunct")])
  #       if dhHere < 0:
   #        distHere = -distHere
         print(dependency, dhHere, distHere)
         dhByDependency[dependency] = dhHere
         distByDependency[dependency] = distHere
  for key in range(len(itos_deps)):
     if itos_deps[key].split(":")[0] not in dhByDependency:
        dhByDependency[itos_deps[key].split(":")[0]] = 0.0
        print("ERROR", itos_deps[key])
        continue
     dhWeights[key] = dhByDependency[itos_deps[key].split(":")[0]]
     distanceWeights[key] = distByDependency[itos_deps[key].split(":")[0]]
  originalCounter = "NA"



# "linearization_logprobability"
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   # Loop Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum #+ sum(line.get("children_decisions_logprobs",[]))


   # there are the gradients of its children
   for child in line["children_DH"]:
      allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   result.append(line)
   line["relevant_logprob_sum"] = allGradients
   for child in line["children_HD"]:
      allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   return allGradients

import numpy.random
import numpy as np



def orderSentence(sentence, weights, coordinate, newWeight, printThings, training):
   root = None
   for line in sentence:
     line["children_DH"] = []
     line["children_HD"] = []

   for j, line in enumerate(sentence):
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      if line["coarse_dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         continue
      dhSampled = (weights[line["coarse_dep"]] < weights["HEAD"])
      direction = "DH" if dhSampled else "HD"
      line["direction"] = direction
      if printThings: 
         print("\t".join(list(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], line["head"], dhSampled, direction   ]  ))))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction].append(line["index"])

   for line in sentence:
     line["children_DH"].sort(key=lambda x:weights[sentence[x-1]["coarse_dep"]])
     line["children_HD"].sort(key=lambda x:weights[sentence[x-1]["coarse_dep"]])
    

   
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
print(dhByDependency)
preHeadWeights = [x for x in itos_deps if dhByDependency[x] > 0]
postHeadWeights = [x for x in itos_deps if dhByDependency[x] <= 0]
shuffle(preHeadWeights)
shuffle(postHeadWeights)

weights = preHeadWeights + ["HEAD"] + postHeadWeights

weights = dict(zip(weights[::], range(len(weights))))
weights = dict(list(zip(list(weights), [2*x for x in range(len(weights))])))
print(weights)



morphKeyValuePairs = set()

vocab_lemmas = {}

corpusTrain = CorpusIterator_V(args.language,"train", storeMorph=True).iterator(rejectShortSentences = False)
pairs = set()
counter = 0
data = list(corpusTrain)
for sentence in data:
   for line in sentence:
       line["coarse_dep"] = makeCoarse(line["dep"])

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


contextLength = args.cutoff
lengthForPrediction = args.cutoff

def calculateTradeoffForWeights(weights, relevantAffix):
    wordsWithoutSentinels = 0
    dev = []
    for _ in range(args.cutoff+2):
      dev.append(stoi_words["PAD"])
    dev.append(stoi_words["SOS"])
    wordsWithoutSentinels += 1

    numberOfRelevantSentences = 0
    for sentence in data:
       #depOccurs = False
       #for line in sentence:
       #   if line["coarse_dep"] == relevantAffix or relevantAffix == "HEAD" or relevantAffix == None:
       #      depOccurs = True
       #      break
       #if not depOccurs:
       #   continue
       numberOfRelevantSentences += 1
       ordered = orderSentence(sentence, weights, None, None, False, False)
       dev.append(stoi_words["SOS"])
       wordsWithoutSentinels += 1
       for ch in ordered:
         dev.append(stoi_words[ch["word"]])
         wordsWithoutSentinels += 1
       dev.append(stoi_words["EOS"])
       wordsWithoutSentinels += 1
       for _ in range(args.cutoff+2):
         dev.append(stoi_words["PAD"])
#    print(len([x for x in dev if x != 0]), wordsWithoutSentinels)
   
    array = dev    
    #print array
    
    print("Sorting "+str(numberOfRelevantSentences)+" sentences.")
    if numberOfRelevantSentences == 0:
        return 1.0
    indices = range(len(array))
    indices = sorted(indices, key=lambda x:array[x:x+args.cutoff])
    #print indices
    print("Now calculating information")
    
    # bigram surprisal
    startOfCurrentPrefix = None
    
    
    endPerStart = [None for _ in range(len(indices))]
    endPerStart[0] = len(endPerStart)-1
    
    lastCrossEntropy = 10000
    memory = 0
    
    devSurprisalTable = []
    
    for contextLength in range(0, args.cutoff-1):
          crossEntropy = 0
          totalSum = 0
          i = 0
          lengthsOfSuffixes = 0
          countTowardsSurprisal = 0
          while i < len(indices):
       #        if indices[i]+contextLength >= len(array):
       #          startOfCurrentSuffix += 1
       #          i += 1
               while endPerStart[i] is None:
                   i += 1
               endOfCurrentPrefix = endPerStart[i] # here, end means the last index (not the one where a new thing starts)
               endPerStart[i] = None
               assert endOfCurrentPrefix is not None, (i, len(indices))
               # now we know the range where the current prefix occurs
               countOfCurrentPrefix = ((endOfCurrentPrefix-i+1)) # if contextLength >= 1 else wordsWithoutSentinels)
               assert countOfCurrentPrefix >= 1
               startOfCurrentSuffix = i
       #        if indices[startOfCurrentSuffix]+contextLength >= len(array):
       #          startOfCurrentSuffix += 1
       #          i += 1
               j = i
               firstNonSentinelSuffixForThisPrefix = i # by default, will be modified in time in case sentinels show up
               probSumForThisPrefix = 0
               while j <= endOfCurrentPrefix:
                    # is j the last one?
       
                    assert j == endOfCurrentPrefix or j+1 < len(indices), (i,j)
       
                    assert j < len(indices)
       
                    # when there is nothing to predict
                    if indices[j]+contextLength >= len(array):
                        j+=1
                        startOfCurrentSuffix+=1
                        continue 
       #
                    # don't want to be predicting the final "0" tokens
       #             if array[indices[j]+contextLength] == 0:
       #                 j+=1
       #                 startOfCurrentSuffixExceptSentinels+=1
       #                 continue
       
                    assert indices[startOfCurrentSuffix]+contextLength < len(array), (i,j)
                    assert j >= i
                    assert endOfCurrentPrefix >= j
                    if j == endOfCurrentPrefix or indices[j+1]+contextLength >= len(array) or array[indices[j+1]+contextLength] != array[indices[startOfCurrentSuffix]+contextLength]:
                      endOfCurrentSuffix = j # here, end means the last index (not the one where a new thing starts)
       #               print (i, j)
                      lengthOfCurrentSuffix =  endOfCurrentSuffix - startOfCurrentSuffix + 1
                      lengthsOfSuffixes += lengthOfCurrentSuffix
       
                      if array[indices[startOfCurrentSuffix]+contextLength] != 0: # don't incur loss for predicting sentinel
                         countOfCurrentPrefixWithoutSentinelSuffix = endOfCurrentPrefix - firstNonSentinelSuffixForThisPrefix + 1 # here important that sentinel comes first when sorting (is 0)
                         assert countOfCurrentPrefixWithoutSentinelSuffix <= countOfCurrentPrefix, ["endOfCurrentPrefix", endOfCurrentPrefix, "firstNonSentinelSuffixForThisPrefix", firstNonSentinelSuffixForThisPrefix, "i", i]
                         conditionalProbability = float(lengthOfCurrentSuffix) / countOfCurrentPrefixWithoutSentinelSuffix
                         probSumForThisPrefix += conditionalProbability
                         surprisal = -log(conditionalProbability)
                         probabilityThatThisSurprisalIsIncurred = float(lengthOfCurrentSuffix) / wordsWithoutSentinels
                         crossEntropy += probabilityThatThisSurprisalIsIncurred * surprisal
                         totalSum += probabilityThatThisSurprisalIsIncurred
                         countTowardsSurprisal += lengthOfCurrentSuffix
                      else:
                         firstNonSentinelSuffixForThisPrefix = j+1
                      endPerStart[startOfCurrentSuffix] = endOfCurrentSuffix
                      startOfCurrentSuffix = j+1
                    if j == endOfCurrentPrefix:
                      break
                    if indices[j+1]+contextLength >= len(array):
                       startOfCurrentSuffix = j+2
                       j+=2
                    else:
                       j+=1
               i = endOfCurrentPrefix+1
               assert lengthsOfSuffixes >= i-contextLength
               assert min(abs(probSumForThisPrefix - 0.0), abs(probSumForThisPrefix - 1.0)) < 0.00001, probSumForThisPrefix
          assert i-lengthsOfSuffixes == contextLength
          #assert lastCrossEntropy >= crossEntropy
       
          print("==================================================countTowardsSurprisal", countTowardsSurprisal)
          memory += min(lengthForPrediction, contextLength) * (lastCrossEntropy-crossEntropy)
          print("CONTEXT LENGTH "+str(contextLength)+"   "+str( crossEntropy)+"  "+str((lastCrossEntropy-crossEntropy))+"   "+str(memory))
          assert abs(totalSum - 1.0) < 0.00001, totalSum
       
     #   print >> outFile, "\t".join(map(str,["FileName","ModelName","Counter", "Model", "Distance", "Entropy", "ConditionalMI", "Memory"]))
      
          #print("\t".join(map(str,[myID, __file__, 100000, model, originalCounter, contextLength, crossEntropy, (lastCrossEntropy-crossEntropy) if contextLength > 0 else crossEntropy, memory, lengthForPrediction, contextLength ])))
          lastCrossEntropy = crossEntropy
          devSurprisalTable.append(crossEntropy)
    #  lengthForPrediction = int(sys.argv[4]) #20
    #contextLength = int(sys.argv[5]) #20
    print(devSurprisalTable)
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
    assert 20>memory, memory
    auc += mi * (20-memory)
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
   

print(weights)
#quit()
HEADWeight = weights["HEAD"]

fullAUCs = []

for iteration in range(20000):
  #assert weights["amod"] < weights["HEAD"]
  #assert weights["obj"] > weights["HEAD"]
  assert weights["HEAD"] == HEADWeight
  coordinate=choice(itos_deps)
  mostCorrect, mostCorrectValue = 0, None
  for newValue in [-1] + [2*x+1 for x in range(len(weights))] + [weights[coordinate]]:
     if (weights[coordinate] < weights["HEAD"]) != (newValue < weights["HEAD"]):
        continue
     if random() < 0.0 and newValue != weights[coordinate]:
        continue
     print(newValue, mostCorrect, coordinate)
     weights_ = {x : y if x != coordinate else newValue for x, y in weights.items()}
     correctCount = calculateTradeoffForWeights(weights_, coordinate)
     if correctCount > mostCorrect:
        mostCorrectValue = newValue
        mostCorrect = correctCount
  assert (weights[coordinate] < weights["HEAD"]) == (mostCorrectValue < weights["HEAD"])
  print("===================")
  print("Iteration", iteration, "Best AUC", mostCorrect)
  weights[coordinate] = mostCorrectValue
  itos_ = sorted(itos_deps+["HEAD"], key=lambda x:weights[x])
  weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))
  print(weights)
  for x in itos_:
     print("\t".join([str(y) for y in [x, weights[x]]]))
  if (iteration + 1) % 50 == 0:
     fullAUCs.append(calculateTradeoffForWeights(weights_, None))
     with open(TARGET_DIR+"/optimized_"+args.language+"_"+__file__+"_"+str(myID)+".tsv", "w") as outFile:
        print(iteration, mostCorrect, file=outFile)
        print(" ".join([str(x) for x in fullAUCs]), file=outFile)
        print(str(args), file=outFile)
        for key in itos_:
           print(key, weights[key], file=outFile)



