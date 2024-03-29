# based on yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py

import random
import sys

objectiveName = "LM"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--model", dest="model", type=str)
parser.add_argument("--alpha", dest="alpha", type=float, default=0.0)
parser.add_argument("--gamma", dest="gamma", type=int, default=1)
parser.add_argument("--delta", dest="delta", type=float, default=1.0)
parser.add_argument("--cutoff", dest="cutoff", type=int, default=15)
parser.add_argument("--idForProcess", dest="idForProcess", type=int, default=random.randint(0,10000000))
import random



args=parser.parse_args()
print(args)


assert args.alpha >= 0
assert args.alpha <= 1
assert args.delta >= 0
assert args.gamma >= 1





myID = args.idForProcess


TARGET_DIR = "/u/scr/mhahn/deps/memory-need-ngrams-phonotactics/"



posUni = set() 

posFine = set() 






from math import log, exp
from random import random, shuffle, randint, Random

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



def orderSentence(sentence, dhLogits, printThings):

   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   if args.model == "REAL_REAL":
       # Collect tokens to be removed (i.e., punctuation)
      eliminated = []
   for line in sentence:
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
      if args.model == "REAL":
         dhSampled = (line["head"] > line["index"])
      else:
         dhSampled = (dhLogit > 0) 
     
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], ("->".join(list(key)) + "         ")[:22], line["head"], dhLogit, dhSampled, direction]))

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
   if args.model == "REAL_REAL":
       while len(eliminated) > 0:
          line = eliminated[0]
          del eliminated[0]
          if "removed" in line:
             continue
          line["removed"] = True
          if "children_DH" in line:
            assert 0 not in line["children_DH"]
            eliminated = eliminated + [sentence[x-1] for x in line["children_DH"]]
          if "children_HD" in line:
            assert 0 not in line["children_HD"]
            eliminated = eliminated + [sentence[x-1] for x in line["children_HD"]]

   
   linearized = []
   recursivelyLinearize(sentence, root, linearized)
   if args.model == "REAL_REAL":
      linearized = filter(lambda x:"removed" not in x, sentence)
   if printThings or len(linearized) == 0:
     print " ".join(map(lambda x:x["word"], sentence))
     print " ".join(map(lambda x:x["word"], linearized))
   return linearized, logits


#dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()



import torch.cuda
import torch.nn.functional



crossEntropy = 10.0
counter = 0
lastDevLoss = None
failedDevRuns = 0
devLosses = [] 


#
#
#def createStreamContinuous(corpus):
#    global crossEntropy
#    global devLosses
#
#    input_indices = [2] # Start of Segment
#    wordStartIndices = []
#    sentCount = 0
#    words = []
#    for sentence in corpus:
#       sentCount += 1
#       if sentCount % 10 == 0:
#         print ["DEV SENTENCES", sentCount]
#
#       for line in sentence:
#          words.append(line["word"])
#    shuffle(words)
#    for WORD in words: 
#      if args.model == "REAL_REAL":
#         WORD2 = WORD
#      elif args.model == "EVEN_ODD":
#        WORDA = WORD[::2]
#        WORDB = WORD[1::2]
#        WORD2 = WORDA+WORDB
#        assert len(WORD2) == len(WORD)
#      elif args.model == "SORTED": # not invertible
#        WORD2 = "".join(sorted(list(WORD)))
#      for x in WORD2:
#         yield x
#      for _ in range(args.cutoff+2):
#         yield "EOW"
#






prefix = args.language[0]

words = []

with open("/u/scr/corpora/ldc/1996/LDC96L14/"+args.language+"/"+prefix+"ml/"+prefix+"ml.cd", "r") as inFile:
  for line in inFile:
     line = line.strip().split("\\")
     if line[3] != "M":
        continue
     WORD = line[1]
     if args.model == "REAL_REAL":
        WORD2 = WORD
     elif args.model == "RANDOM":
       WORD2 = list(WORD)
       Random(myID).shuffle(WORD2)
       WORD2 = "".join(WORD2)
     else:
       assert False
     print(WORD2)
     words.append(WORD2)
print(len(words))
#quit()

Random(5).shuffle(words)
#words=words[:5000]

dev = []
for word in words:
   for ch in word:
     dev.append(ch)
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

devSurprisalTable = []
for k in range(0,args.cutoff):
   print(k)
   startK, endK = getStartEnd(k) # Possible speed optimization: There is some redundant computation here, could be reused from the previous iteration. But the algorithm is very fast already.
   startK2, endK2 = getStartEnd(k+1)
   cachedFollowingCounts = {}
   for j in range(len(idev)):
#      print(dev[idev[j]])
      if dev[idev[j]] in ["PAD", "SOS"]:
         continue
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
       print >> sys.stderr, "PROBLEM"
       print >> sys.stderr, lastProbability
       surprisal = 1000
   devSurprisalTable.append(surprisal)
   print("Surprisal", surprisal, len(itos))


#assert False

outpath = TARGET_DIR+"/estimates-"+args.language+"_"+__file__+"_model_"+str(myID)+"_"+args.model+".txt"
print(outpath)
with open(outpath, "w") as outFile:
         print >> outFile, str(args)
         print >> outFile, devSurprisalTable[-1]
         print >> outFile, " ".join(map(str,devSurprisalTable))



