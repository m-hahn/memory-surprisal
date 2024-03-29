
import random
import sys

objectiveName = "LM"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--model", dest="model", type=str)
parser.add_argument("--alpha", dest="alpha", type=float, default=0)
parser.add_argument("--gamma", dest="gamma", type=int, default=1)
parser.add_argument("--delta", dest="delta", type=float, default=1.0)
parser.add_argument("--cutoff", dest="cutoff", type=int, default=7)
parser.add_argument("--idForProcess", dest="idForProcess", type=int, default=random.randint(0,10000000))
import random



args=parser.parse_args()
print(args)


assert args.model.startswith("GROUND")

assert args.alpha >= 0
assert args.alpha <= 1
assert args.delta >= 0
assert args.gamma >= 1





myID = args.idForProcess


TARGET_DIR = "/u/scr/mhahn/deps/memory-need-ngrams-np/"



posUni = set() 

posFine = set() 





import math
from math import log, exp
from random import random, shuffle, randint

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator_V import CorpusIterator_V as CorpusIterator

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}

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
   for partition in ["train"]:
     for sentence in CorpusIterator(args.language,partition, storeMorph=True).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          vocab_lemmas[line["lemma"]] = vocab_lemmas.get(line["lemma"], 0) + 1

          depsVocab.add(line["dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          for morph in line["morph"]:
              morphKeyValuePairs.add(morph)
          if line["dep"] == "root":
             continue

          posHere = line["posUni"]
          posHead = sentence[line["head"]-1]["posUni"]
          dep = line["dep"]
          direction = "HD" if line["head"] < line["index"] else "DH"
          key = (posHead, dep, posHere)
          keyWithDir = (posHead, dep, posHere, direction)
          orderTable[keyWithDir] = orderTable.get(keyWithDir, 0) + 1
          keys.add(key)
          distanceCounts[key] = distanceCounts.get(key,0.0) + 1.0
          distanceSum[key] = distanceSum.get(key,0.0) + abs(line["index"] - line["head"])
   dhLogits = {}
   for key in keys:
      hd = orderTable.get((key[0], key[1], key[2], "HD"), 0) + 1.0
      dh = orderTable.get((key[0], key[1], key[2], "DH"), 0) + 1.0
      dhLogit = log(dh) - log(hd)
      dhLogits[key] = dhLogit
      originalDistanceWeights[key] = (distanceSum[key] / distanceCounts[key])
   return dhLogits, vocab, keys, depsVocab

import torch.nn as nn
import torch
from torch.autograd import Variable


def recursivelyLinearize(sentence, position, result):
   line = sentence[position-1]

   if "children_DH" in line:
      for child in line["children_DH"]:
         recursivelyLinearize(sentence, child, result)
   result.append(line)
   if "children_HD" in line:
      for child in line["children_HD"]:
         recursivelyLinearize(sentence, child, result)

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       if args.model == "REAL":
          return remainingChildren
       logits = [(x, distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]]) for x in remainingChildren]
       logits = sorted(logits, key=lambda x:x[1], reverse=(not reverseSoftmax))
       childrenLinearized = map(lambda x:x[0], logits)
       return childrenLinearized           

model_ = (args.model+"_").split("_")


def orderSentence(sentence, dhLogits, printThings):

   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   if args.model == "REAL_REAL":
       # Collect tokens to be removed (i.e., punctuation)
      eliminated = []
   for line in sentence:
      line["coarse_dep"] = makeCoarse(line["dep"])
      if line["dep"] == "root":
          root = line["index"]
          continue
      # Exclude Punctuation
      if line["dep"].startswith("punct"):
         if args.model == "REAL_REAL":
            eliminated.append(line)
         continue
      # Determine ordering relative to head
      key = (sentence[line["head"]-1]["posUni"], line["dep"], line["posUni"])
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      if True or args.model == "REAL":
         dhSampled = (line["head"] > line["index"])
      else:
         dhSampled = (dhLogit > 0) 
     
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print("\t".join(map(str,["ORD", line["word"], line["posUni"], line["index"], ("->".join(list(key)) + "         ")[:22], line["head"], dhLogit, dhSampled, direction])))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])

   nounPhrases = []
   for line in sentence:
      if line["posUni"] == "NOUN":
         childrenLeft = [sentence[i-1] for i in line.get("children_DH", [])]
         childrenRight = [sentence[i-1] for i in line.get("children_HD", [])]
         leftDependencies = [x["dep"] for x in childrenLeft]
         if len(leftDependencies) == 0:
            continue
         if set(leftDependencies).issubset(set(["case", "det", "nummod", "amod"])):
            leftLengths = [len(x.get("children_DH", []) + x.get("children_HD", [])) for x in childrenLeft]
            if max(leftLengths+[0]) == 0:
              if len(leftDependencies) == 1:
               continue
              dependents = [sentence[i-1] for i in line.get("children_DH", [])]
              if model_[1] != "":
                  positions = {{"A" : "amod", "N" : "nummod", "D" : "det"}[x] : model_[1].index(x) for x in "AND"}
                  positions["case"] = -1
#                  print(positions)
                  dependents = sorted(dependents, key=lambda x:positions[x["coarse_dep"]])
#                  quit()
#              if args.model == "GROUND_AND":
#                dependents = sorted(dependents, key=lambda x:{"case" : -1, "amod" : 0, "nummod" : 1, "det" : 2}[x["coarse_dep"]])
#              elif args.model == "GROUND_NDA":
#                dependents = sorted(dependents, key=lambda x:{"case" : -1, "amod" : 2, "nummod" : 0, "det" : 1}[x["coarse_dep"]])
#              elif args.model == "GROUND_ADN":
#                dependents = sorted(dependents, key=lambda x:{"case" : -1, "amod" : 0, "nummod" : 2, "det" : 1}[x["coarse_dep"]])
#              elif args.model == "GROUND_DAN":
#                dependents = sorted(dependents, key=lambda x:{"case" : -1, "amod" : 1, "nummod" : 2, "det" : 0}[x["coarse_dep"]])
#              elif args.model != "GROUND":
#                assert False
              nounPhrases.append(dependents + [line])
              if random() > 0.98:
                 print([x["word"] for x in nounPhrases[-1]])
   return nounPhrases


dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()



posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   
itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)


import os

if args.model == "REAL" or args.model == "REAL_REAL":
  originalCounter = "NA"
elif args.model == "RANDOM_BY_TYPE":
  dhByType = {}
  distByType = {}
  for dep in itos_pure_deps:
    dhByType[dep.split(":")[0]] = random() - 0.5
    distByType[dep.split(":")[0]] = random()
  for key in range(len(itos_deps)):
     dhWeights[key] = dhByType[itos_deps[key][1].split(":")[0]]
     distanceWeights[key] = distByType[itos_deps[key][1].split(":")[0]]
  originalCounter = "NA"
elif args.model.startswith("GROUND"):
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
         dependency = line[headerGrammar.index("Dependency")]
         dhHere = float(line[headerGrammar.index("DH_Mean_NoPunct")])
         distHere = float(line[headerGrammar.index("Distance_Mean_NoPunct")])
         print(dependency, dhHere, distHere)
         dhByDependency[dependency] = dhHere
         distByDependency[dependency] = distHere
  for key in range(len(itos_deps)):
     if itos_deps[key][1].split(":")[0] not in dhByDependency:
        continue
     dhWeights[key] = dhByDependency[itos_deps[key][1].split(":")[0]]
     distanceWeights[key] = distByDependency[itos_deps[key][1].split(":")[0]]
  originalCounter = "NA"
else:
  assert False, args.model

#
#print zip(itos_deps,distanceWeights)
#
#print dhWeights[stoi_deps[("NOUN", "amod", "ADJ")]]
#print dhWeights[stoi_deps[("NOUN", "nummod", "NUM")]]
#print dhWeights[stoi_deps[("NOUN", "det", "DET")]]


AMOD = distanceWeights[stoi_deps[("NOUN", "amod", "ADJ")]]
NUMMOD = distanceWeights[stoi_deps[("NOUN", "nummod", "NUM")]]
DET = distanceWeights[stoi_deps[("NOUN", "det", "DET")]]


if args.model== "GROUND_AND":
   for x in range(len(itos_deps)):
      if itos_deps[x][1] == "amod":
          distanceWeights[x] = DET
      if itos_deps[x][1] == "det":
          distanceWeights[x] = AMOD
elif args.model== "GROUND_DAN":
   for x in range(len(itos_deps)):
      if itos_deps[x][1] == "amod":
          distanceWeights[x] = NUMMOD
      if itos_deps[x][1] == "nummod":
          distanceWeights[x] = AMOD
elif args.model== "GROUND_ADN":
   for x in range(len(itos_deps)):
      if itos_deps[x][1] == "amod":
          distanceWeights[x] = DET
      if itos_deps[x][1] == "det":
          distanceWeights[x] = NUMMOD
      if itos_deps[x][1] == "det":
         distanceWeights[x] = AMOD
  
#print distanceWeights[stoi_deps[("NOUN", "amod", "ADJ")]]
#print distanceWeights[stoi_deps[("NOUN", "nummod", "NUM")]]
#print distanceWeights[stoi_deps[("NOUN", "det", "DET")]]
#   

words = list(vocab.items())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = list(map(lambda x:x[0], words))
stoi = dict(list(zip(itos, range(len(itos)))))

if len(itos) > 6:
   assert stoi[itos[5]] == 5



vocab_size = len(itos)









import torch.cuda
import torch.nn.functional



crossEntropy = 10.0
counter = 0
lastDevLoss = None
failedDevRuns = 0
devLosses = [] 


MAX_DIST = 5


def createStreamContinuous(corpus):
    global crossEntropy
    global devLosses

    input_indices = [2] # Start of Segment
    wordStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       if sentCount % 10 == 0:
         print(["DEV SENTENCES", sentCount])

 #      dependencies = set([x["dep"] for x in sentence])
#       if "amod" not in dependencies and "det" not in dependencies:
 #         continue
       nounPhrases = orderSentence(sentence, dhLogits, sentCount % 500 == 0)
      
       timeSinceRelevant = MAX_DIST
       for np in nounPhrases:
         #print(np)
         for line in np+["EOS"]:
          if line == "EOS":
            yield ("EOS", MAX_DIST, "EOS", "EOS")
          else:
            if line["dep"] in ["amod", "det"]:
                timeSinceRelevant = 0
            else:
                timeSinceRelevant += 1
            yield (line["word"], min(MAX_DIST,timeSinceRelevant), line["posUni"], line["coarse_dep"])
         for _ in range(args.cutoff+2):
            yield ("PAD", MAX_DIST, "PAD", "PAD")
         yield ("SOS", MAX_DIST, "SOS", "PAD")



corpusDev = CorpusIterator(args.language,"train", storeMorph=True).iterator(rejectShortSentences = False)
dev = list(createStreamContinuous(corpusDev))[::-1]


#corpusTrain = CorpusIterator(args.language,"train", storeMorph=True).iterator(rejectShortSentences = False)
#train = list(createStreamContinuous(corpusTrain))[::-1]
train = dev

idev = range(len(dev))
itrain = range(len(train))

devW = [x[0] for x in dev]
trainW = [x[0] for x in train]


idev = sorted(idev, key=lambda i:devW[i:i+20])
itrain = sorted(itrain, key=lambda i:trainW[i:i+20])

print(idev)

idevInv = [x[1] for x in sorted(zip(idev, range(len(idev))), key=lambda x:x[0])]
itrainInv = [x[1] for x in sorted(zip(itrain, range(len(itrain))), key=lambda x:x[0])]

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
     prefix = tuple(devW[idev[j]:idev[j]+k])
     while l2 < len(train):
        prefix2 = tuple(trainW[itrain[l2]:itrain[l2]+k])
        if prefix <= prefix2:
             start[j] = l2
             break
        l2 += 1
     if l2 == len(train):
        start[j] = l2
     while l < len(train):
        prefix2 = tuple(trainW[itrain[l]:itrain[l]+k])
        if prefix < prefix2:
             end[j] = l
             break
        l += 1
     if l == len(train):
        end[j] = l
     start2, end2 = start[j], end[j]
     assert start2 <= end2
     if start2 > 0 and end2 < len(train):
       assert prefix > tuple(trainW[itrain[start2-1]:itrain[start2-1]+k]), (prefix, tuple(trainW[itrain[start2-1]:itrain[start2-1]+k]))
       assert prefix <= tuple(trainW[itrain[start2]:itrain[start2]+k])
       assert prefix >= tuple(trainW[itrain[end2-1]:itrain[end2-1]+k])
       assert prefix < tuple(trainW[itrain[end2]:itrain[end2]+k])
   return start, end


lastProbability = [None for _ in idev]
newProbability = [None for _ in idev]

devSurprisalTable = []
devSurprisalTables = {"amod" : ([]), "det" : ([]), "nummod" : ([]), "case" :([]), "NOUN" : ([]), "EOS" : ([])}
for k in range(0,args.cutoff):
   print(k)
   startK, endK = getStartEnd(k) # Possible speed optimization: There is some redundant computation here, could be reused from the previous iteration. But the algorithm is very fast already.
   startK2, endK2 = getStartEnd(k+1)
   cachedFollowingCounts = {}
   surprisalByPOS = {"amod" : ([0,0]), "det" : ([0,0]), "nummod" : ([0,0]), "case" :([0,0]), "NOUN" : ([0,0]), "EOS" : ([0,0])}
   for j in range(len(idev)):
#      print(dev[j], dev[j][0] == "PAD")
#      print(devW[idev[j]])
      if devW[idev[j]] in ["PAD", "SOS"]:
         continue
#      print(devW[idev[j]])
      start2, end2 = startK2[j], endK2[j]
      devPref = tuple(devW[idev[j]:idev[j]+k+1])
      if start2 > 0 and end2 < len(train):
        assert devPref > tuple(trainW[itrain[start2-1]:itrain[start2-1]+k+1]), (devPref, tuple(trainW[itrain[start2-1]:itrain[start2-1]+k+1]))
        assert devPref <= tuple(trainW[itrain[start2]:itrain[start2]+k+1]), (devPref, tuple(trainW[itrain[start2]:itrain[start2]+k+1]))
        assert devPref >= tuple(trainW[itrain[end2-1]:itrain[end2-1]+k+1])
        assert devPref < tuple(trainW[itrain[end2]:itrain[end2]+k+1])

      assert start2 <= end2

      countNgram = end2-start2
      if k >= 1:
         if idev[j]+1 < len(idevInv):
           prefixIndex = idevInv[idev[j]+1]
           assert devW[idev[prefixIndex]] == devW[idev[j]+1]
   
           prefixStart, prefixEnd = startK[prefixIndex], endK[prefixIndex]
           countPrefix = prefixEnd-prefixStart
           if countPrefix < args.gamma: # there is nothing to interpolate with, just back off
              assert k > 0
              newProbability[j] = lastProbability[j]
           else:
              assert countPrefix >= countNgram, (countPrefix, countNgram)
   
              following = set()
              if (prefixStart, prefixEnd) in cachedFollowingCounts:
                  followingCount = cachedFollowingCounts[(prefixStart, prefixEnd)]
              else:
                for l in range(prefixStart, prefixEnd):
                  if k < itrain[l]+1:
                     following.add(trainW[itrain[l]-1])
                     assert devPref[1:] == tuple(trainW[itrain[l]-1:itrain[l]+k])[1:], (k, itrain[l], l, devPref , tuple(trainW[itrain[l]-1:itrain[l]+k]))
                followingCount = len(following)
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
   #   print(newProbability[j], dev[idev[j]][2])
      if dev[idev[j]][3] not in surprisalByPOS and dev[idev[j]][2] != "NOUN":
         print(dev[idev[j]][3], dev[idev[j]][2])
         relation = None
      elif dev[idev[j]][3] not in surprisalByPOS:
         relation = "NOUN"
      else:
         relation = dev[idev[j]][3]
      if relation is not None:
         surprisalByPOS[relation][0] += newProbability[j]
         surprisalByPOS[relation][1] += 1
   #           print(k, probability, devW[idev[j]], countNgram)
#              assert abs(probability) > 1e-5, devW[idev[j]]
   for pos in surprisalByPOS:
     devSurprisalTables[pos].append(surprisalByPOS[pos][0] / surprisalByPOS[pos][1])
   print(devSurprisalTables)
   lastProbability = newProbability 
   newProbability = [None for _ in idev]
   assert all([x is None or x <=0 for x in lastProbability])
   try:
    #   print(lastProbability[:100])
       lastProbabilityFiltered = [x for x in lastProbability if x is not None]
     #  print(lastProbabilityFiltered[:100])
       surprisal = - sum([x for x in lastProbabilityFiltered])/len(lastProbabilityFiltered)
   except ValueError:
       print("PROBLEM", file=sys.stderr)
       print(lastProbability, file=sys.stderr)
       surprisal = 1000
   devSurprisalTable.append(surprisal)
   print("Surprisal", surprisal, len(itos))


#assert False

outpath = TARGET_DIR+"/estimates-"+args.language+"_"+__file__+"_model_"+str(myID)+"_"+args.model+".txt"
print(outpath)
with open(outpath, "w") as outFile:
         print(str(args), file=outFile)
         print(devSurprisalTable[-1], file=outFile)
         print(" ".join(map(str,devSurprisalTable)), file=outFile)
         for key in sorted(list(devSurprisalTables)):
            print(" ".join([key] + list(map(str,devSurprisalTables[key]))), file=outFile)


