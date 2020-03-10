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


def processVerb(verb):
    if len(verb) > 0:
      if "VERB" in [x["posUni"] for x in verb[1:]]:
        print([x["word"] for x in verb])
      data.append([x["lemma"] for x in verb])

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
if args.model == "RANDOM":
  weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))
#  weights['する'] = -1
else:
  weights = {}
  weights = {}
  import glob
  PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized"
  files = glob.glob(PATH+"/optimized_*.py_"+args.model+".tsv")
  assert len(files) == 1
  with open(files[0], "r") as inFile:
     next(inFile)
     for line in inFile:
        morpheme, weight = line.strip().split(" ")
        weights[morpheme] = int(weight)

from collections import defaultdict

mistakes = defaultdict(int)

def getCorrectOrderCount(weights):
   correct = 0
   incorrect = 0
   correctFull = 0
   incorrectFull = 0
   for verb in data:
      hasMadeMistake = False
      for i in range(1, len(verb)):
#         if verb[i] == 'する':
 #           continue
         for j in range(1, i):
  #           if verb[j] == 'する':
   #             continue
             weightI = weights[verb[i]]
             weightJ = weights[verb[j]]
             if weightI > weightJ:
               correct+=1
             else:
               incorrect+=1
               hasMadeMistake = True
               print("MISTAKE", verb[i], weights[verb[i]], verb[j], weights[verb[j]], verb)
               mistakes[(verb[j], verb[i])] += 1
      if len(verb) > 2:
        if not hasMadeMistake:
            correctFull += 1
        else:
            incorrectFull += 1
   return correct/(correct+incorrect), correctFull/(correctFull+incorrectFull)

result = getCorrectOrderCount(weights)
print(mistakes)
print(result)

with open("/u/scr/mhahn/deps/memory-need-ngrams-morphology-accuracy/accuracy_"+__file__+"_"+str(myID)+"_"+args.model+".txt", "w") as outFile:
   print(result[0], file=outFile)
   print(result[1], file=outFile)
   mistakes = list(mistakes.items())
   mistakes.sort(key=lambda x:x[1], reverse=True)
   for x, y in mistakes:
      print(x[0], x[1], y, file=outFile)
print("/u/scr/mhahn/deps/memory-need-ngrams-morphology-accuracy/accuracy_"+__file__+"_"+str(myID)+"_"+args.model+".txt")


