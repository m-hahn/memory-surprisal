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


from collections import defaultdict

verbsCounts = defaultdict(int)
verbsPerMorpheme = defaultdict(lambda : defaultdict(int))
affixCount = defaultdict(int)

for verbWithAff in data:
  verb = verbWithAff[0]
  verbsCounts[verb] += 1
  verbsCounts["_TOTAL_"] += 1

  suffixes = set(verbWithAff[1:])
  for affix in suffixes:
       verbsPerMorpheme[affix][verb] += 1
       affixCount[affix] += 1

results = []

for affix in itos:
#  print(affix)
 # print(verbsPerMorpheme[affix])
  # MI between `verb' and `was this affix found?'
  mi = 0
  verbFreqSum = 0
  for verb in verbsCounts:
     if verb == "_TOTAL_":
        continue
     verbFreq = verbsCounts[verb] / verbsCounts["_TOTAL_"]
     probOfAffix = verbsPerMorpheme[affix][verb] / verbsCounts[verb] 
     probOfAffixPrior = affixCount[affix] / verbsCounts["_TOTAL_"]
     assert probOfAffix <= 1, probOfAffix
     assert probOfAffixPrior <= 1
     verbFreqSum += verbFreq
     mi += verbFreq * (probOfAffix * log((probOfAffix+1e-10)/(probOfAffixPrior+1e-10)) + (1-probOfAffix) * log((1-probOfAffix+1e-10)/(1-probOfAffixPrior+1e-10)))
     #print(log(probOfAffix/probOfAffixPrior))

  probOfAffixPrior = affixCount[affix] / verbsCounts["_TOTAL_"]
  entropyOfHavingAffix = -(probOfAffixPrior * log(probOfAffixPrior+1e-10) + (1-probOfAffixPrior) * log(1-probOfAffixPrior+1e-10))
  print(affix, mi, verbFreqSum, entropyOfHavingAffix)
  assert mi <= 0.7
  assert mi >= 0
  results.append((affix, mi, entropyOfHavingAffix, entropyOfHavingAffix-mi))
results.sort(key=lambda x:x[2]-x[1], reverse=True)
print("==========")
for r in results:
   print(r)

