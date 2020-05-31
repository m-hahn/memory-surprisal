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
parser.add_argument("--cutoff", dest="cutoff", type=int, default=15)
parser.add_argument("--idForProcess", dest="idForProcess", type=int, default=random.randint(0,10000000))
import random



args=parser.parse_args()
print(args)


assert args.alpha >= 0
assert args.alpha <= 1
assert args.delta >= 0
assert args.gamma >= 1


#RELEVANT_KEY = "lemma"

def getKey(word):
  return word[header["lemma"]][:2]

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


#from corpusIterator_V import CorpusIterator_V

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
    affixLemma = getKey(affix) #[header[RELEVANT_KEY]]
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

itos_sfx_ = itos_sfx[::]
shuffle(itos_sfx_)
weights_sfx = dict(list(zip(itos_sfx_, [2*x for x in range(len(itos_sfx_))])))


if args.model == "RANDOM":
  weights_sfx = dict(list(zip(itos_sfx_, [2*x for x in range(len(itos_sfx_))])))
else:
  weights_sfx = {}
  import glob
  PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized"
  files = glob.glob(args.model)
  assert len(files) == 1
  assert "Suffixes" in files[0], files
  assert "Normalized" not in files[0]
  with open(files[0], "r") as inFile:
     next(inFile)
     next(inFile)
     next(inFile)
     for line in inFile:
        morpheme, weight = line.strip().split(" ")
        weights_sfx[morpheme] = int(weight)

errors = defaultdict(int)

hasSeenType = set()


AFFIX_KEY = "sfx"

def getCorrectOrderCount(weights_sfx):
   correct = 0
   incorrect = 0
   correctFull = 0
   incorrectFull = 0

   correctTypes = 0
   incorrectTypes = 0
   correctFullTypes = 0
   incorrectFullTypes = 0
   for q, verb in enumerate(data):
      affixes = [(getKey(x), weights_sfx[getKey(x)]) for x in verb if x[header["type1"]] == AFFIX_KEY]
      if len(affixes) <= 1:
        continue

      keyForThisVerb = " ".join([x[header["lemma"]] for x in verb])
      hasSeenThisVerb = (keyForThisVerb in hasSeenType)
      hasMadeMistake = False
      for i in range(0, len(affixes)):
         for j in range(0, i):
             weightI = affixes[i][1]
             weightJ = affixes[j][1]
             if weightI > weightJ:
               correct+=1
               if not hasSeenThisVerb:
                 correctTypes += 1
             else:
               incorrect+=1
               if not hasSeenThisVerb:
                 incorrectTypes += 1
               hasMadeMistake = True
               errors[(affixes[j][0], affixes[i][0])] += 1
      if len(affixes) > 1:
        if not hasMadeMistake:
            correctFull += 1
            if not hasSeenThisVerb:
              correctFullTypes += 1
        else:
            incorrectFull += 1
            if not hasSeenThisVerb:
              incorrectFullTypes += 1
      assert correct+incorrect>0, affixes
      if not hasSeenThisVerb:
        hasSeenType.add(keyForThisVerb)
   print((correctFull+incorrectFull))
   return correct/(correct+incorrect), correctFull/(correctFull+incorrectFull), correctTypes/(correctTypes+incorrectTypes), correctFullTypes/(correctFullTypes+incorrectFullTypes)




for q in range(len(data)):
   verb = data[q]
   affixes_keys = [getKey(x) for x in verb if x[header["type1"]] == AFFIX_KEY]
   if len(affixes_keys) <= 1:
     continue

   segmentation = []
   for j in range(len(verb)):
      # subject prefix?
      if ".SBJ" in verb[j][header["analysis"]]:
         segmentation.append([])
         segmentation[-1].append(verb[j])
      else:
         if len(segmentation) == 0:
      #     print(verb)
           segmentation.append([])
         segmentation[-1].append(verb[j])
   ###############################################################################   
   # Restrict to the last verb, chopping off initial auxiliaries and their affixes
   ###############################################################################

   #print(segmentation[-1])
#   if len(segmentation) > 1:
#    for w in range(len(segmentation)-1):
#     if len(segmentation[w]) == 2:
#         _ = 0
#     else:
#        print(segmentation[w])


   verb = segmentation[-1]

   affixes_keys = [getKey(x) for x in verb if x[header["type1"]] == AFFIX_KEY]


   # It is important to overwrite data[q] before continuing
   data[q] = verb

   if len(affixes_keys) <= 1:
     continue



result = getCorrectOrderCount(weights_sfx)
print(errors)
print(result)
if args.model.endswith(".tsv"):
   model = args.model[args.model.rfind("_")+1:-4]   
else:
   model = args.model
with open("results/accuracy_"+__file__+"_"+str(myID)+"_"+model+".txt", "w") as outFile:
   print(result[0], file=outFile)
   print(result[1], file=outFile)
   print(result[2], file=outFile)
   print(result[3], file=outFile)
   errors = list(errors.items())
   errors.sort(key=lambda x:x[1], reverse=True)
   for x, y in errors:
      print(x[0], x[1], y, file=outFile)
print("/u/scr/mhahn/deps/memory-need-ngrams-morphology-accuracy/accuracy_"+__file__+"_"+str(myID)+"_"+model+".txt")


