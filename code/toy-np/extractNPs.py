
import random
import sys

objectiveName = "LM"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--model", dest="model", type=str, default="REAL")
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





myID = args.idForProcess


TARGET_DIR = "/u/scr/mhahn/deps/memory-need-ngrams/"



posUni = set() 

posFine = set() 





tuplesByDep = {"amod" : [], "nummod" : [], "det" : []}

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



def makeCoarse(x):
   if ":" in x:
      return x[:x.index(":")]
   return x

def orderSentence(sentence, dhLogits, printThings):

   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   if args.model == "REAL_REAL":
       # Collect tokens to be removed (i.e., punctuation)
      eliminated = []
   for line in sentence:

      line["coarse_dep"] = makeCoarse(line["dep"])

      if line["dep"] == "root":
          root = line["index"]
          continue
      # Exclude Punctuation
      if line["dep"].startswith("punct"):
         if args.model == "REAL_REAL":
            eliminated.append(line)
         continue
      # Determine ordering relative to head
      if line["coarse_dep"] in tuplesByDep:
          depName = line["coarse_dep"]
          if depName == "det":
#             if "PronType" in " ".join(line["morph"]) and "PronType=Dem" not in line["morph"]:
 #               continue
             print("\t".join([line["lemma"], line["posUni"], line["posFine"]])) #, line["morph"])

          if False and depName == "nummod" and all([ord(x) < 58 for x in line["word"]]):
              print(line["word"])          
          else:
             tuplesByDep[depName].append((line["word"], sentence[line["head"]-1]["word"]))
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



dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()



posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   
itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)


import os

if args.model == "REAL" or args.model == "REAL_REAL":
  originalCounter = "NA"
elif args.model == "RANDOM_BY_TYPE":
  dhByType = {}
  distByType = {}
  for dep in itos_pure_deps:
    dhByType[dep.split(":")[0]] = random() - 0.5
    distByType[dep.split(":")[0]] = random()
  for key in range(len(itos_deps)):
     dhWeights[key] = dhByType[itos_deps[key][1].split(":")[0]]
     distanceWeights[key] = distByType[itos_deps[key][1].split(":")[0]]
  originalCounter = "NA"
elif args.model == "GROUND":
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
     dhWeights[key] = dhByDependency[itos_deps[key][1].split(":")[0]]
     distanceWeights[key] = distByDependency[itos_deps[key][1].split(":")[0]]
  originalCounter = "NA"
else:
  assert False, args.model



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








corpusTrain = CorpusIterator(args.language,"train", storeMorph=True).iterator(rejectShortSentences = False)


count = 0
for s in corpusTrain:
    count += 1
    if count % 100 == 0:
       print(count)
    orderSentence(s, dhLogits, False)


def toCounts(l):
   c = {}
   for x in l:
     c[x] = c.get(x,0)+1.0
   return c

def getEntropy(counts):
   total = sum(y for x, y in counts.items())
#   print(counts, total)
 #  print([y for _, y in counts.items()])
   return -sum([y/total * log(y/total) for _, y in counts.items()])

def pmi(pairs):
   if len(pairs) == 0:
      return 0
   left, right = zip(*pairs)
   byLeft = {}
   for l, r in pairs:
      if l not in byLeft:
         byLeft[l] = {}
      if r not in byLeft[l]:
         byLeft[l][r]=0
      byLeft[l][r]+=1.0
   left = toCounts(left)
   totalCount = len(pairs)
   entropy = 0
   for l, count in left.items():
      entropy += float(count) / totalCount * getEntropy(byLeft[l])
   marginalEntropy = getEntropy(toCounts(right))
   return marginalEntropy - entropy

print(", ".join([y+" : "+str(z) for y, z in (toCounts([x[0] for x in tuplesByDep["det"]])).items()]))
print("amod", pmi(tuplesByDep["amod"]), len(tuplesByDep["amod"]))
print("det", pmi(tuplesByDep["det"]), len(tuplesByDep["det"]))
print("nummod", pmi(tuplesByDep["nummod"]), len(tuplesByDep["nummod"]))

