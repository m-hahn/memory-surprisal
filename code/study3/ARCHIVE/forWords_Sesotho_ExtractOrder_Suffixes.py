# based on yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py

import random
import sys

objectiveName = "LM"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="Sesotho_Acqdiv")
parser.add_argument("--model", dest="model", type=str)
parser.add_argument("--alpha", dest="alpha", type=float, default=0.0)
parser.add_argument("--gamma", dest="gamma", type=int, default=1)
parser.add_argument("--delta", dest="delta", type=float, default=1.0)
parser.add_argument("--cutoff", dest="cutoff", type=int, default=12)
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

words = []

header = ["form", "lemma", "analysis", "type1", "type2"]
header = dict(list(zip(header, range(len(header)))))

def processWord(word):
   nonAffix = [x[-3] for x in word if x[-3] not in ["sfx","pfx"]]
#   print(nonAffix, len(word))

   assert len(nonAffix) == 1
   if nonAffix[0] == "v":
#      print( [x[-3] for x in word if x[-3] in ["sfx","pfx"]])

      words.append([x[8:] for x in word])

posUni = set() 

posFine = set() 

currentWord = []
with open("/u/scr/mhahn/CODE/acqdiv-database/csv/morphemes5.csv", "r") as inFile:
  next(inFile)
  for line in inFile:
     line = line.strip().replace('","', ',').split(",")
     if len(line) > 14:
#       print(line)
#       assert len(line) == 15, len(line)
       line[9] = ",".join(line[9:-4])
       line = line[:10] + line[-4:]
 #      print(line)
       assert len(line) == 14, len(line)
#     print(len(line))
     assert len(line) == 14, line
     if line[4] == "Sesotho":
#       print(line)
       if line[-3] == "sfx":
         currentWord.append(line)
       elif line[-3] == "pfx" and len(currentWord) > 0 and currentWord[-1][-3] != "pfx":
         #print([x[-3] for x in currentWord])
         processWord(currentWord)
         currentWord = []
  #       print("---")
         currentWord.append(line)
       elif line[-3] == "pfx":
         currentWord.append(line)
       elif line[-3] != "sfx" and len(currentWord) > 0 and currentWord[-1][-3] == "sfx":
         #print([x[-3] for x in currentWord])
         processWord(currentWord)
         currentWord = []
 #        print("---")
         currentWord.append(line)
       elif line[-3] not in ["sfx", "pfx"] and len(currentWord) > 0 and currentWord[-1][-3] != "pfx":
         #print([x[-3] for x in currentWord])
         processWord(currentWord)
         currentWord = []
#         print("---")
         currentWord.append(line)
       else:
          currentWord.append(line)
   #    print(line[-3])

#print(words)
#quit()


from math import log, exp
from random import random, shuffle, randint, Random, choice


from corpusIterator_V import CorpusIterator_V

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}

pairs = set()
counter = 0
data = words

#quit()
import torch.nn as nn
import torch
from torch.autograd import Variable


import numpy.random



import torch.cuda
import torch.nn.functional


from collections import defaultdict

prefixFrequency = defaultdict(int)
suffixFrequency = defaultdict(int)
for verbWithAff in data:
  for affix in verbWithAff:
    affixLemma = affix[header["form"]]
    if affix[header["type1"]] == "pfx":
       prefixFrequency[affixLemma] += 1
    elif affix[header["type1"]] == "sfx":
       suffixFrequency[affixLemma] += 1

itos_pfx = sorted(list((prefixFrequency)))
stoi_pfx = dict(list(zip(itos_pfx, range(len(itos_pfx)))))

itos_sfx = sorted(list((suffixFrequency)))
stoi_sfx = dict(list(zip(itos_sfx, range(len(itos_sfx)))))

print(prefixFrequency)
print(suffixFrequency)

print(itos_pfx)
print(itos_sfx)

itos_pfx_ = itos_pfx[::]
shuffle(itos_pfx_)
weights_pfx = dict(list(zip(itos_pfx_, [2*x for x in range(len(itos_pfx_))])))

itos_sfx_ = itos_sfx[::]
shuffle(itos_sfx_)
weights_sfx = dict(list(zip(itos_sfx_, [2*x for x in range(len(itos_sfx_))])))

  

def getCorrectOrderCount(weights_pfx, weights_sfx, coordinate, newValue):
   correct = 0
   incorrect = 0
   if len(index_sfx[coordinate]) == 0:
     return 1.0
   for verb in index_sfx[coordinate]:
      #prefixes_keys = [x[header["form"]] for x in verb if x[header["type1"]] == "sfx"]
#      if coordinate not in prefixes_keys:
 #       assert False
  #      continue
   
      suffixes = [(x[header["form"]], weights_sfx[x[header["form"]]]) for x in verb if x[header["type1"]] == "sfx"]
      assert len(suffixes) > 1
#      suffixes = [(x[header["form"]], weights_sfx[x[header["form"]]]) for x in verb if x[header["type1"]] == "sfx"]

      for affixes in [suffixes]:    
        for i in range(0, len(affixes)):
           for j in range(0, i):
               if affixes[i][0] == coordinate:
                   weightI = newValue
               else:
                  weightI = affixes[i][1]
  
               if affixes[j][0] == coordinate:
                   weightJ = newValue
               else:
                  weightJ = affixes[j][1]
               if weightI > weightJ:
                 correct+=1
               else:
                 incorrect+=1
        assert correct+incorrect>0, affixes
   if correct+incorrect == 0:
      print("ERROR 19722: #", coordinate, "#")
      assert False, (index_sfx[coordinate], coordinate)
      return 1.0
   return correct/(correct+incorrect)

index_sfx = defaultdict(list)

for verb in data:
   suffixes_keys = [x[header["form"]] for x in verb if x[header["type1"]] == "sfx"]
   if len(suffixes_keys) <= 1:
     continue
   for sfx in suffixes_keys:
      index_sfx[sfx].append(verb)
   index_sfx[None].append(verb)



for iteration in range(2000):
  coordinate = choice(itos_sfx)
  while prefixFrequency[coordinate] < 10:
    coordinate = choice(itos_sfx)
  mostCorrect, mostCorrectValue = 0, None
  for newValue in [-1] + [2*x+1 for x in range(len(itos_sfx))]:
     #if random() > 0.3:
     #  continue
     correctCount = getCorrectOrderCount(weights_pfx, weights_sfx, coordinate, newValue)
#     print(coordinate, newValue, iteration, correctCount)
     if correctCount > mostCorrect:
        mostCorrectValue = newValue
        mostCorrect = correctCount
  assert not (mostCorrectValue is None)
  weights_sfx[coordinate] = mostCorrectValue
  itos_sfx_ = sorted(itos_sfx, key=lambda x:weights_sfx[x])
  weights_sfx = dict(list(zip(itos_sfx_, [2*x for x in range(len(itos_sfx_))])))
  print(weights_sfx)
  for x in itos_sfx_:
     print("\t".join([str(y) for y in [x, weights_sfx[x], suffixFrequency[x], len(index_sfx[x])]]))
  print(iteration, mostCorrect, prefixFrequency[coordinate], len(index_sfx[coordinate]))
  print("Total", getCorrectOrderCount(weights_pfx, weights_sfx, None, 0))
#with open("output/extracted_"+__file__+"_"+str(myID)+".tsv", "w") as outFile:
#  for x in itos_:
#  #   if affixFrequencies[x] < 10:
#   #    continue
#     print("\t".join([str(y) for y in [x, weights[x], affixFrequencies[x]]]), file=outFile)


