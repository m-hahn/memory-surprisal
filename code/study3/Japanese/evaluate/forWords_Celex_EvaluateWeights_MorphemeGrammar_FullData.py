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


affixFrequencies = {}
for verbWithAff in data:
  for affix in verbWithAff[1:]:
    affixLemma = getRepresentation(affix["lemma"])
    affixFrequencies[affixLemma] = affixFrequencies.get(affixLemma, 0)+1

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
if args.model == "RANDOM":
  weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))
#  weights['する'] = -1
else:
  weights = {}
  weights = {}
  import glob
  files = glob.glob(args.model)
  assert len(files) == 1
  with open(files[0], "r") as inFile:
     next(inFile)
     for line in inFile:
        if "extract" in files[0]:
           morpheme, weight, _ = line.strip().split("\t")
        else:
           morpheme, weight = line.strip().split(" ")
        weights[morpheme] = int(weight)

from collections import defaultdict

errors = defaultdict(int)

hasSeenType = set()

def getCorrectOrderCount(weights):
   correct = 0
   incorrect = 0
   correctFull = 0
   incorrectFull = 0

   correctTypes = 0
   incorrectTypes = 0
   correctFullTypes = 0
   incorrectFullTypes = 0
   for verb in data:
      keyForThisVerb = " ".join([x["lemma"] for x in verb])
      hasSeenThisVerb = (keyForThisVerb in hasSeenType)
      hasMadeMistake = False
      for i in range(1, len(verb)):
         for j in range(1, i):
             weightI = weights[getRepresentation(verb[i]["lemma"])]
             weightJ = weights[getRepresentation(verb[j]["lemma"])]
             if weightI == weightJ:
                continue
             if weightI > weightJ:
               correct+=1
               if not hasSeenThisVerb:
                 correctTypes += 1
             else:
               incorrect+=1
               if not hasSeenThisVerb:
                 incorrectTypes += 1
               hasMadeMistake = True
#               print("MISTAKE", verb[i]["lemma"], weights[getRepresentation(verb[i]["lemma"])], verb[j], weights[getRepresentation(verb[j]["lemma"])], [x["lemma"] for x in verb])
               errors[(getRepresentation(verb[j]["lemma"]), getRepresentation(verb[i]["lemma"]))] += 1
      if len(verb) > 2:
        if not hasMadeMistake:
            correctFull += 1
            if not hasSeenThisVerb:
              correctFullTypes += 1
        else:
            incorrectFull += 1
            if not hasSeenThisVerb:
              incorrectFullTypes += 1
      if not hasSeenThisVerb:
        hasSeenType.add(keyForThisVerb)
   return correct/(correct+incorrect), correctFull/(correctFull+incorrectFull),correctTypes/(correctTypes+incorrectTypes), correctFullTypes/(correctFullTypes+incorrectFullTypes)

result = getCorrectOrderCount(weights)
print(errors)
print(result)

model = args.model[args.model.rfind("_")+1:-4]   
with open("results/accuracy_"+__file__+"_"+str(myID)+"_"+model+".txt", "w") as outFile:
   print(result[0], file=outFile)
   print(result[1], file=outFile)
   print(result[2], file=outFile)
   print(result[3], file=outFile)
   errors = list(errors.items())
   errors.sort(key=lambda x:x[1], reverse=True)
   for x, y in errors:
      print(x[0], x[1], y, file=outFile)
print("ERRORS")
print(errors)
print(result)

print("results/accuracy_"+__file__+"_"+str(myID)+"_"+args.model+".txt")


