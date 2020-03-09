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
weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))
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
#weights = {'める': 0, 'てる': 2, '始める': 4, 'そうだ': 6, 'られる': 8, 'あう': 10, 'ざるを得る': 12, 'える': 14, '出来る': 16, 'まい': 18, 'きる': 20, 'だめ': 22, 'ちゃう': 24, 'できる': 26, 'がたい': 28, '易い': 30, 'させる': 32, 'べる': 34, 'たー': 36, 'かける': 38, 'みたいだ': 40, 'する': 42, 'れる': 44, 'せる': 46, 'くださる': 48, 'かもしれる': 50, 'ようだ': 52, 'でした': 54, 'らしい': 56, 'たい': 58, 'かねる': 60, 'ける': 62, '出す': 64, 'ざるをえる': 66, 'ない': 68, 'にくい': 70, 'やすい': 72, '済み': 74, 'なる': 76, 'ます': 78, 'う': 80, '続ける': 82, 'た': 84, 'だ': 86}

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
         for j in range(1, i):
             weightI = weights[verb[i]]
             weightJ = weights[verb[j]]
             if weightI > weightJ:
               correct+=1
             else:
               incorrect+=1
               hasMadeMistake = True
               print("MISTAKE", verb[i], weights[verb[i]], verb[j], weights[verb[j]], verb)
               mistakes[(verb[i], verb[j])] += 1
      if len(verb) > 2:
        if not hasMadeMistake:
            correctFull += 1
        else:
            incorrectFull += 1
   return correct/(correct+incorrect), correctFull/(correctFull+incorrectFull)

result = getCorrectOrderCount(weights)
print(mistakes)
print(result)

