# based on yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py

import random
import sys
import romkan

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


def getRepresentation(lemma):
   if lemma == "させる" or lemma == "せる":
     return "CAUSATIVE"
   elif lemma == "れる" or lemma == "られる" or lemma == "える" or lemma == "得る" or lemma == "ける":
     return "PASSIVE_POTENTIAL"
   else:
     return lemma





from math import log, exp
from random import random, shuffle, randint, Random, choice

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator_V import CorpusIterator_V

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}


def processVerb(verb):
    if len(verb) > 0:
      if "VERB" in [x["posUni"] for x in verb[1:]]:
        print([x["word"] for x in verb])
      data.append(verb)

corpusTrain = CorpusIterator_V(args.language,"train", storeMorph=True).iterator(rejectShortSentences = False)
pairs = set()
counter = 0
data = []
for sentence in corpusTrain:
#    print(len(sentence))
    verb = []
    for line in sentence:
       if line["posUni"] == "PUNCT":
          processVerb(verb)
          verb = []
          continue
       elif line["posUni"] == "VERB":
          processVerb(verb)
          verb = []
          verb.append(line)
       elif line["posUni"] == "AUX" and len(verb) > 0:
          verb.append(line)
       elif line["posUni"] == "SCONJ" and line["word"] == 'て':
          verb.append(line)
          processVerb(verb)
          verb = []
       else:
          processVerb(verb)
          verb = []
print(len(data))
#quit()
print(counter)
#print(data)
print(len(data))

#quit()
import torch.nn as nn
import torch
from torch.autograd import Variable


import numpy.random



import torch.cuda
import torch.nn.functional


words = []

affixFrequency = {}
for verbWithAff in data:
  for affix in verbWithAff[1:]:
    affixLemma = getRepresentation(affix["lemma"])
    affixFrequency[affixLemma] = affixFrequency.get(affixLemma, 0)+1


itos = set()
for verbWithAff in data:
  for affix in verbWithAff[1:]:
    affixLemma = getRepresentation(affix["lemma"])
    itos.add(affixLemma)
itos = sorted(list(itos))
stoi = dict(list(zip(itos, range(len(itos)))))


print(itos)
print(stoi)

itos_ = itos[::]
shuffle(itos_)
weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))

raw2Tagged = dict()

with open("../data/extractedVerbs.txt", "r") as inFileRaw:
  with open("../data/extractedVerbs_tagged.txt", "r") as inFileTagged:
    try:
     for index in range(1000000):
       raw = next(inFileRaw).strip()
       tagged = next(inFileTagged).strip()
       raw2Tagged[raw] = tagged
    except StopIteration:
       _ = 0
for line in data:
 # print(" ".join([x["word"] for x in line]))
  raw = "".join([x["word"] for x in line])
  if False and max([ord(y) for y in raw]) < 12531: # no need to transform
    for x in line:
      x["hiragana"] = x["word"]
    continue
  tagged = [x.split("/") for x in raw2Tagged[raw].split(" ")]
  if raw == "".join([x[2] for x in tagged]):
     for x in line:
        x["hiragana"] = x["word"]
     continue
  iTagged = 0
  jTagged = 0
  iLine = 0
  jLine = 0
  line[iLine]["hiragana"] = ""

  for i in range(len(raw)):
#    print(iTagged, jTagged, tagged)    
    jTagged += 1
    if jTagged == len(tagged[iTagged][0]):
      line[iLine]["hiragana"] += tagged[iTagged][2]
      iTagged += 1
      jTagged = 0 
    jLine += 1
    if jLine == len(line[iLine]["word"]):
      iLine += 1
      if iLine < len(line):
        line[iLine]["hiragana"] = ""
      jLine = 0 
     
#  print(line)
  assert "".join([x["hiragana"] for x in line]) == "".join([x[2] for x in tagged])
#  print(line)
#  print(tagged)
#  print([x["word"] for x in line])
#  print("RAW", raw)
#  print(max([ord(y) for y in raw]))
  hasFoundProblem = []
  for index, x in enumerate(line):
#     print(x["word"], "==", x["hiragana"])
     if x["word"] != x["hiragana"] and max([ord(y) for y in x["word"]]) < 200 and x["word"] not in ["fax", "pr"]:
        hasFoundProblem.append(index)
     elif x["hiragana"] == "":
        hasFoundProblem.append(index)
#     assert x["hiragana"]  != "", tagged
  if len(hasFoundProblem)>0:
    assert len(line) > 1
 #   print(hasFoundProblem)
    if hasFoundProblem == [0]:
      assert len(line) > 1
      if line[1]["hiragana"].endswith(line[1]["word"]): 
         line[0]["hiragana"] = line[1]["hiragana"][:-len(line[1]["word"])]
         line[1]["hiragana"] = line[1]["word"]
      elif len(line[1]["hiragana"]) == len(line[0]["word"]) + len(line[1]["word"]):
         line[0]["hiragana"] = line[1]["hiragana"][:len(line[0]["word"])]
         line[1]["hiragana"] = line[1]["hiragana"][len(line[0]["word"]):]
      elif line[0]["word"] == "出":
         assert line[1]["hiragana"].startswith("で")
         line[0]["hiragana"] = "で"
         line[1]["hiragana"] = line[1]["hiragana"][1:]
      elif line[0]["word"] == "見":
         assert line[1]["hiragana"].startswith("み")
         line[0]["hiragana"] = "み"
         line[1]["hiragana"] = line[1]["hiragana"][1:]
      elif line[1]["hiragana"].count(line[0]["word"][-1]) == 1:
         line[0]["hiragana"] = line[1]["hiragana"][:line[1]["hiragana"].index(line[0]["word"][-1])]
         line[1]["hiragana"] = line[1]["hiragana"][line[1]["hiragana"].index(line[0]["word"][-1]):]
      else:
         assert False
    elif hasFoundProblem == [1]:
      if line[1]["word"] + line[2]["word"] == line[2]["hiragana"]:
        line[1]["hiragana"] = line[1]["word"]
        line[2]["hiragana"] = line[2]["word"]
      else:
        assert False
    elif hasFoundProblem == [2]:
      if line[2]["word"] + line[3]["word"] == line[3]["hiragana"]:
        line[2]["hiragana"] = line[2]["word"]
        line[3]["hiragana"] = line[3]["word"]
      else:
        assert False
    elif hasFoundProblem == [0,1]:
       if line[0]["word"] == "見":
         line[0]["hiragana"] = "み"
       if line[0]["word"] == "つ":
          line[0]["hiragana"] = line[0]["word"]
       if line[1]["word"] == "込":
         line[1]["hiragana"] = "こ"
       if line[1]["word"] == "け":
          line[1]["hiragana"] = line[1]["word"]
       line[2]["hiragana"] = line[2]["hiragana"][len(line[0]["hiragana"])+len(line[1]["hiragana"]):]
#  print(line)
#  assert "".join([x["hiragana"] for x in line]) == ("".join([x[2] for x in tagged])).replace("ＰＲ", "pr")
  for index, x in enumerate(line):
 #    print(x["word"], "==", x["hiragana"])
     if x["word"] != x["hiragana"] and max([ord(y) for y in x["word"]]) < 200 and x["word"] not in ["pr", "fax"]:
        assert False
     elif x["hiragana"] == "":
        assert False

cachedPhonemization = {}

def phonemize(x):
   if x not in cachedPhonemization:
      phonemized = romkan.to_roma(x)
      if max([ord(y) for y in phonemized]) > 200: # contains Kanji
         cachedPhonemization[x] = x
      else:
        if x.endswith("っ"):
          assert phonemized.endswith("xtsu")
          phonemized = phonemized.replace("xtsu", "G") # G for `geminate'
        phonemized = phonemized.replace("ch", "C")
        phonemized = phonemized.replace("sh", "S")
        phonemized = phonemized.replace("ts", "T")
        cachedPhonemization[x] = phonemized
   phonemized = cachedPhonemization[x]
   return phonemized

def calculateTradeoffForWeights(weights):
    dev = []
    for verb in data:
       affixes = verb[1:]
       affixes = sorted(affixes, key=lambda x:weights[getRepresentation(x["lemma"])])
       for ch in [verb[0]] + affixes:
         for char in phonemize(ch["hiragana"]):
           dev.append(char)
       #    print(char)
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
    
#    print(idev)
    
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
#       print(k)
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
    assert 7>memory
    auc += mi * (7-memory)
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
  coordinate=choice(itos)
  while affixFrequency.get(coordinate, 0) < 10 and random() < 0.95:
     coordinate = choice(itos)
  mostCorrect, mostCorrectValue = 0, None
  for newValue in [-1] + [2*x+1 for x in range(len(itos))] + [weights[coordinate]]:
     if random() < 0.9 and newValue != weights[coordinate]:
        continue
     print(newValue, mostCorrect, coordinate, affixFrequency[coordinate])
     weights_ = {x : y if x != coordinate else newValue for x, y in weights.items()}
     correctCount = calculateTradeoffForWeights(weights_)
#     print(weights_)
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
     if affixFrequency[x] < 10:
       continue
     print("\t".join([str(y) for y in [x, weights[x], affixFrequency[x]]]))
  if (iteration + 1) % 50 == 0:
     with open(TARGET_DIR+"/optimized_"+__file__+"_"+str(myID)+".tsv", "w") as outFile:
        print(iteration, mostCorrect, str(args), file=outFile)
        for key in itos_:
           print(key, weights[key], file=outFile)



