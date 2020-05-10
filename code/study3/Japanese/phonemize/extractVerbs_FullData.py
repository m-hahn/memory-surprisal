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
    affixLemma = affix["lemma"]
    affixFrequency[affixLemma] = affixFrequency.get(affixLemma, 0)+1


itos = set()
for verbWithAff in data:
  for affix in verbWithAff[1:]:
    affixLemma = affix["lemma"]
    itos.add(affixLemma)
itos = sorted(list(itos))
stoi = dict(list(zip(itos, range(len(itos)))))


print(itos)
print(stoi)

itos_ = itos[::]
shuffle(itos_)
weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))


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

with open("../data/extractedVerbs.txt", "w") as outFile:
 for verb in data:
   print("".join([x["word"] for x in verb]), file=outFile)
#./kytea /juicier/scr120/scr/mhahn/CODE/memory-surprisal/code/ngram-control/create_models_ngrams/morph/Japanese/data/extractedVerbs.txt > /juicier/scr120/scr/mhahn/CODE/memory-surprisal/code/ngram-control/create_models_ngrams/morph/Japanese/data/extractedVerbs_tagged.txt
