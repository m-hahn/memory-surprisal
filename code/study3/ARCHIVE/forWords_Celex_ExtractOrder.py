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


