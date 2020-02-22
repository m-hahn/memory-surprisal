# based on yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py

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


TARGET_DIR = "/u/scr/mhahn/deps/memory-need-ngrams-morphology/"



posUni = set() 

posFine = set() 






from math import log, exp
from random import random, shuffle, randint, Random, choice

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator_V import CorpusIterator_V

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}

corpusTrain = CorpusIterator_V(args.language,"train", storeMorph=True).iterator(rejectShortSentences = False)
pairs = set()
counter = 0
data = []
for sentence in corpusTrain:
#    print(len(sentence))
    verb = []
    for line in sentence[::-1]:
#       print(line)
       if line["posUni"] == "PUNCT":
          continue
       verb.append(line)
       if line["posUni"] == "VERB":
          verb = verb[::-1]
#          print(verb)
#          print([x["dep"] for x in verb])
#          print([x["posUni"] for x in verb])
#          print([x["word"] for x in verb])
#          print([x["lemma"] for x in verb])
#          print([x["head"] for x in verb])
          for i in range(1,len(verb)):
            for j in range(1,i):
              pairs.add((verb[i]["lemma"], verb[j]["lemma"]))
              if (verb[j]["lemma"], verb[i]["lemma"]) in pairs:
                 print("======", (verb[i]["lemma"], verb[j]["lemma"]), [x["dep"] for x in verb], "".join([x["word"] for x in verb]))
          if len(verb) > 1:
            data.append([x["lemma"] for x in verb])
          counter += 1
          break
       if line["posUni"] not in ["AUX", "SCONJ"]:
          break
       if line["dep"] not in ["aux"]:
          break
print(counter)
print(data)
print(len(data))

#quit()
import torch.nn as nn
import torch
from torch.autograd import Variable


import numpy.random



import torch.cuda
import torch.nn.functional


words = []


affixFrequencies = {}
for verbWithAff in data:
  for affix in verbWithAff[1:]:
    affixFrequencies[affix] = affixFrequencies.get(affix, 0)+1

itos = set()
for verbWithAff in data:
  for affix in verbWithAff[1:]:
    itos.add(affix)
itos = sorted(list(itos))
stoi = dict(list(zip(itos, range(len(itos)))))


print(itos)
print(stoi)

itos_ = itos[::]
shuffle(itos_)
weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))

def getCorrectOrderCount(weights, coordinate, newValue):
   correct = 0
   incorrect = 0
   for verb in data:
      for i in range(1, len(verb)):
         for j in range(1, i):
             if verb[i] == coordinate:
                 weightI = newValue
             else:
                weightI = weights[verb[i]]

             if verb[j] == coordinate:
                 weightJ = newValue
             else:
                weightJ = weights[verb[j]]
             if weightI > weightJ:
               correct+=1
             else:
               incorrect+=1
   return correct/(correct+incorrect)

for iteration in range(200):
  coordinate = choice(itos)
  mostCorrect, mostCorrectValue = 0, None
  for newValue in [-1] + [2*x+1 for x in range(len(itos))]:
     correctCount = getCorrectOrderCount(weights, coordinate, newValue)
#     print(coordinate, newValue, iteration, correctCount)
     if correctCount > mostCorrect:
        mostCorrectValue = newValue
        mostCorrect = correctCount
  print(iteration, mostCorrect)
  weights[coordinate] = mostCorrectValue
  itos_ = sorted(itos, key=lambda x:weights[x])
  weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))
  print(weights)
  for x in itos_:
     print("\t".join([str(y) for y in [x, weights[x], affixFrequencies[x]]]))
with open("output/extracted_"+str(myID)+".tsv", "w") as outFile:
  for x in itos_:
  #   if affixFrequencies[x] < 10:
   #    continue
     print("\t".join([str(y) for y in [x, weights[x], affixFrequencies[x]]]), file=outFile)

quit()

for verbWithAff in data:
  verb = verbWithAff[0]
  affixes = verbWithAff[1:]
  if args.model == "REAL_REAL":
     WORD2 = verbWithAff
  elif args.model == "RANDOM":
    Random(myID).shuffle(affixes)
    WORD2 = [verb] + affixes
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
#       print >> sys.stderr, "PROBLEM"
 #      print >> sys.stderr, lastProbability
       surprisal = 1000
   devSurprisalTable.append(surprisal)
   print("Surprisal", surprisal, len(itos))


#assert False

#outpath = TARGET_DIR+"/estimates-"+args.language+"_"+__file__+"_model_"+str(myID)+"_"+args.model+".txt"
#print(outpath)
#with open(outpath, "w") as outFile:
#         print >> outFile, str(args)
#         print >> outFile, devSurprisalTable[-1]
#         print >> outFile, " ".join(map(str,devSurprisalTable))
#
#

