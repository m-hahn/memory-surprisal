
import random
import sys

objectiveName = "LM"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--model", dest="model", type=str)
parser.add_argument("--alpha", dest="alpha", type=float, default=1.0)
parser.add_argument("--gamma", dest="gamma", type=int, default=1)
parser.add_argument("--delta", dest="delta", type=float, default=1.0)
parser.add_argument("--cutoff", dest="cutoff", type=int, default=10)
parser.add_argument("--idForProcess", dest="idForProcess", type=int, default=random.randint(0,10000000))
import random



args=parser.parse_args()
print(args)


assert args.alpha >= 0
assert args.alpha <= 1
assert args.delta >= 0
assert args.gamma >= 1


assert "REAL" not in args.model


myID = args.idForProcess


TARGET_DIR = "/u/scr/mhahn/deps/locality_optimized_i1/it_estimates/"




posUni = set() 

posFine = set() 






from math import log, exp
from random import random, shuffle, randint

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator import CorpusIterator

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}

def initializeOrderTable():
   orderTable = {}
   keys = set()
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in CorpusIterator(args.language,partition).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          line["fine_dep"] = line["dep"]
          depsVocab.add(line["fine_dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          if line["fine_dep"] == "root":
             continue
          posHere = line["posUni"]
          posHead = sentence[line["head"]-1]["posUni"]
          dep = line["fine_dep"]
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


# "linearization_logprobability"
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

mistaken = 0
correct = 0

def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       logits = [(x, distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]]) for x in remainingChildren]
       logits = sorted(logits, key=lambda x:x[1], reverse=(not reverseSoftmax))
       if len(logits)> 1:
#         print(reverseSoftmax, logits)
         global mistaken 
         global correct
         for i in range(len(logits)):
           for j in range(i):
              if (logits[i][0] > logits[j][0]): # != reverseSoftmax:
                correct += 1
              else:
                mistaken += 1
       childrenLinearized = map(lambda x:x[0], logits)
       return childrenLinearized           



def orderSentence(sentence, dhLogits, printThings):

   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   for line in sentence:
      line["fine_dep"] = line["dep"]
      if line["fine_dep"] == "root":
          root = line["index"]
          continue
      # Exclude Punctuation
      if line["fine_dep"].startswith("punct"):
         continue
      # Determine ordering relative to head
      key = (sentence[line["head"]-1]["posUni"], line["fine_dep"], line["posUni"])
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      if True or args.model == "REAL":
         dhSampled = (line["head"] > line["index"])
      else:
         assert False
         dhSampled = (dhLogit > 0) 
     
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], (".".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]])+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])

   if args.model != "REAL_REAL":
      for line in sentence:
         if "children_DH" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
            line["children_DH"] = childrenLinearized
         if "children_HD" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
            line["children_HD"] = childrenLinearized

   
   linearized = []
   recursivelyLinearize(sentence, root, linearized)
   if args.model == "REAL_REAL":
      linearized = filter(lambda x:"removed" not in x, sentence)
   if printThings or len(linearized) == 0:
     print " ".join(map(lambda x:x["word"], sentence))
     print " ".join(map(lambda x:x["word"], linearized))
   return linearized, logits


dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()



posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = sorted(vocab_deps, key=lambda x:x[1])
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)


import os

if args.model == "REAL" or args.model == "REAL_REAL":
  originalCounter = "NA"
elif args.model == "RANDOM_BY_TYPE":
  dhByType = {}
  distByType = {}
  for key in range(len(itos_deps)):
     dhWeights[key] = random() - 0.5
     distanceWeights[key] = random()
  originalCounter = "NA"
elif args.model == "GROUND":
  assert False
  groundPath = "/u/scr/mhahn/deps/manual_output_ground_coarse/"
  import os
  files = [x for x in os.listdir(groundPath) if x.startswith(args.language+"_infer")]
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
     dhWeights[key] = dhByDependency[itos_deps[key]]
     distanceWeights[key] = distByDependency[itos_deps[key]]
  originalCounter = "NA"
else:
#  with open("/u/scr/mhahn/deps/locality_optimized_i1/Chinese_optimizeGrammarForI1_3.py_model_675523898.tsv", "r") as inFile:
  assert "POS" in args.model
  with open("/u/scr/mhahn/deps/locality_optimized_i1/"+args.model, "r") as inFile:

     headerGrammar = next(inFile).strip().split("\t")
     print(headerGrammar)
     dhByDependency = {}
     distByDependency = {}
     for line in inFile:
         line = line.strip().split("\t")




         dependency = line[headerGrammar.index("CoarseDependency")]
         head = line[headerGrammar.index("HeadPOS")]
         dependent = line[headerGrammar.index("DependentPOS")]
         dhHere = float(line[headerGrammar.index("DH_Weight")])
         distHere = float(line[headerGrammar.index("DistanceWeight")])
         print(dependency, dhHere, distHere)
         key = (head, dependency, dependent)
         dhByDependency[key] = dhHere
         distByDependency[key] = distHere
  for key in range(len(itos_deps)):
     dhWeights[key] = dhByDependency[itos_deps[key]]
     distanceWeights[key] = distByDependency[itos_deps[key]]
  originalCounter = "NA"



words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))

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




def createStreamContinuous(corpus):
    global crossEntropy
    global devLosses

    input_indices = [2] # Start of Segment
    wordStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       if sentCount % 10 == 0:
         print ["DEV SENTENCES", sentCount]

       ordered, _ = orderSentence(sentence, dhLogits, sentCount % 500 == 0)

       for line in ordered+["EOS"]:
          if line == "EOS":
            yield "EOS"
          else:
            yield line["word"]



corpusDev = CorpusIterator(args.language,"dev", storeMorph=True).iterator(rejectShortSentences = False)
dev = list(createStreamContinuous(corpusDev))[::-1]


corpusTrain = CorpusIterator(args.language,"train", storeMorph=True).iterator(rejectShortSentences = False)
train = list(createStreamContinuous(corpusTrain))[::-1]


idev = range(len(dev))
itrain = range(len(train))

idev = sorted(idev, key=lambda i:dev[i:i+20])
itrain = sorted(itrain, key=lambda i:train[i:i+20])

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
     if start2 > 0 and end2 < len(train):
       assert prefix > tuple(train[itrain[start2-1]:itrain[start2-1]+k])
       assert prefix <= tuple(train[itrain[start2]:itrain[start2]+k])
       assert prefix >= tuple(train[itrain[end2-1]:itrain[end2-1]+k])
       assert prefix < tuple(train[itrain[end2]:itrain[end2]+k])
   return start, end


lastProbability = [None for _ in idev]
newProbability = [None for _ in idev]


print(correct/(mistaken+correct+0.0))

assert False

devSurprisalTable = []
for k in range(0,args.cutoff):
   print(k)
   startK, endK = getStartEnd(k) # Possible speed optimization: There is some redundant computation here, could be reused from the previous iteration. But the algorithm is very fast already.
   startK2, endK2 = getStartEnd(k+1)
   cachedFollowingCounts = {}
   for j in range(len(idev)):
      start2, end2 = startK2[j], endK2[j]
      devPref = tuple(dev[idev[j]:idev[j]+k+1])
      if start2 > 0 and end2 < len(train):
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
              assert countPrefix >= countNgram, (countPrefix, countNgram)
   
              following = set()
              if (prefixStart, prefixEnd) in cachedFollowingCounts:
                  followingCount = cachedFollowingCounts[(prefixStart, prefixEnd)]
              else:
                for l in range(prefixStart, prefixEnd):
                  if k < itrain[l]+1:
                     following.add(train[itrain[l]-1])
                     assert devPref[1:] == tuple(train[itrain[l]-1:itrain[l]+k])[1:], (k, itrain[l], l, devPref , tuple(train[itrain[l]-1:itrain[l]+k]))
                followingCount = len(following)
                cachedFollowingCounts[(prefixStart, prefixEnd)] = followingCount
              if followingCount == 0:
                  newProbability[j] = lastProbability[j]
              else:
          
                  probability = log(max(countNgram - args.alpha, 0.0) + args.alpha * followingCount * exp(lastProbability[j])) -  log(countPrefix)
                  newProbability[j] = probability
         else:
            newProbability[j] = lastProbability[j]
      elif k == 0:
              probability = log(countNgram + args.delta) - log(len(train) + args.delta * len(itos))
              newProbability[j] = probability
   lastProbability = newProbability 
   newProbability = [None for _ in idev]
   assert all([x <=0 for x in lastProbability])
   try:
       surprisal = - sum([x for x in lastProbability])/len(lastProbability)
   except ValueError:
       print >> sys.stderr, "PROBLEM"
       print >> sys.stderr, lastProbability
       surprisal = 1000
   devSurprisalTable.append(surprisal)
   print("Surprisal", surprisal, len(itos))

assert False

outpath = TARGET_DIR+"/estimates-"+args.language+"_"+__file__+"_model_"+args.model.split("/")[-1]+".txt"
print(outpath)
with open(outpath, "w") as outFile:
         print >> outFile, " ".join(sys.argv)
         print >> outFile, devSurprisalTable[-1]
         print >> outFile, " ".join(map(str,devSurprisalTable))



