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

# for ordering
def getKey(word):
  return word[header["lemma"]][:2]

def getSegmentedForms(word): # return a list , preprocessing
   if "/" not in word[header["lemma"]] and "." not in word[header["lemma"]]:
     return [word]
   elif "/" in word[header["lemma"]]:
    assert word[header["lemma"]].count("/") == 1
    lemmas = word[header["lemma"]].split("/")
    word1 = word[::]
    word2 = word[::]
    word1[1] = lemmas[0]
    word2[1] = lemmas[1]

    word1[0] = "_"
    word2[0] = "_"
    if lemmas[0].startswith("sm") and lemmas[1].startswith("t^"): # merger between subject and tense/aspect marker (> 100 cases in the corpus)
        _ = 0
    elif word[header["analysis"]] == "NEG.POT": #keke, kebe. Compare Doke and Mokofeng, section 424. Seems to be better treated as an auxiliary, as it is followed by subject prefixes in the corpus.
       return None
    elif word[header["analysis"]] == "almost.PRF": # batlile = batla+ile. This is an auxiliary, not a prefix. Doke and Mokofeng, section 575.
       return None
    elif word[header["analysis"]] == "POT.PRF": # kile. This seems to be a prefix, as it is followed by subject prefixes in the corpus.
       return None
    elif word[header["analysis"]] == "be.PRF": # bile . Better treated as an auxiliary, for the same reason.
       return None
    elif word[header["analysis"]] == "do.indeed.PRF": # hlile. Same
       return None
    elif word[header["analysis"]] == "fill.belly.PRF": # Occurs a single time, excluded.
       return None
    else:
       print("SPLIT", word1, word2, word)
       assert False
    return [word1, word2]
   elif word[header["lemma"]] == "a.name" or word[header["lemma"]] == "a.place": #  exclude these data
     return None
   elif word[header["lemma"]].startswith("t^p.om"):
    # print(word)
     lemma1 = word[1][:3]
     lemma2 = word[1][4:]
     #print(lemma2)
     word1 = word[::]
     word2 = word[::]
     word1[1] = lemma1
     word2[1] = lemma2
 
     word1[0] = "_"
     word2[0] = "_"
     if lemma1.startswith("t^") and lemma2.startswith("om"):
   #      print(word)
         assert word[2].startswith("PRS")
         return [word2]
         _ = 0
     else:
        print("SPLIT", word1, word2, word)
        assert False
        return [word1, word2]
   elif word[header["lemma"]].startswith("t^p.rf"):
     lemma1 = word[1][:3]
     lemma2 = word[1][4:]
     #print(lemma2)
     word1 = word[::]
     word2 = word[::]
     word1[1] = lemma1
     word2[1] = lemma2
 
     word1[0] = "_"
     word2[0] = "_"
     if lemma1.startswith("t^") and lemma2.startswith("rf"):
         assert word[2].startswith("PRS")
         return [word2]
         _ = 0
     else:
        print("SPLIT", word1, word2, word)
        assert False
        return [word1, word2]
   else: # exclude these data
     return None

def getNormalizedForm(word): # for prediction
#   print(word)
   return stoi_words[word[header["lemma"]]]

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
dataChosen = []
for verbWithAff in data:
  prefixesResult = []
  for x in verbWithAff:
    if x[header["type1"]] == "pfx":
       segmented = getSegmentedForms(x)
       if segmented is None:
         prefixesResult = None
         break
       prefixesResult += segmented
    else:
       prefixesResult.append(x)
  if prefixesResult is None: # remove this datapoint (affects <20 datapoints)
     continue
  dataChosen.append(prefixesResult)
  for affix in prefixesResult:
    affixLemma = getKey(affix) #[header[RELEVANT_KEY]]
    if affix[header["type1"]] == "pfx":
       prefixFrequency[affixLemma] += 1
    elif affix[header["type1"]] == "sfx":
       suffixFrequency[affixLemma] += 1
data = dataChosen

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


if args.model == "RANDOM":
  weights_pfx = dict(list(zip(itos_pfx_, [2*x for x in range(len(itos_pfx_))])))
else:
  weights_pfx = {}
  import glob
  PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized"
  files = glob.glob(PATH+"/optimized_*.py_"+args.model+".tsv")
  assert len(files) == 1
  assert "Suffixes" not in files[0], files
  assert "Normalized" in files[0]
  with open(files[0], "r") as inFile:
     next(inFile)
     next(inFile)
     next(inFile)
     for line in inFile:
        morpheme, weight = line.strip().split(" ")
        weights_pfx[morpheme] = int(weight)

errors = defaultdict(int)

AFFIX_KEY = "pfx"

def getCorrectOrderCount(weights_pfx, coordinate, newValue):
   correct = 0
   incorrect = 0

   correctFull = 0
   incorrectFull = 0
   for q, verb in enumerate(data):
      #prefixes_keys = [x[header["form"]] for x in verb if x[header["type1"]] == "pfx"]
#      if coordinate not in prefixes_keys:
 #       assert False
  #      continue
   
      affixes = [(getKey(x), weights_pfx[getKey(x)]) for x in verb if x[header["type1"]] == AFFIX_KEY]
      if len(affixes) <= 1:
        continue
      #assert len(prefixes) > 1, verb
#      suffixes = [(x[header[RELEVANT_KEY]], weights_sfx[x[header[RELEVANT_KEY]]]) for x in verb if x[header["type1"]] == "sfx"]

      hasIncorrect = False
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
               #print(weightI, weightJ)
               if weightI > weightJ:
                 correct+=1
               else:
                 hasIncorrect = True
                 incorrect+=1
                 #print("==========")
                 #print(q)
                 #print(affixes)
                 #print("Error pair", (affixes[i][0], affixes[j][0]))
                 #print(verb)
 #                if affixes[i][0] == affixes[j][0]:
#                      assert False
                 errors[(affixes[j][0], affixes[i][0])] += 1
      if len(affixes) > 1:
        if hasIncorrect:
           incorrectFull += 1
        else:
           correctFull += 1
      assert correct+incorrect>0, affixes
   if correct+incorrect == 0:
      print("ERROR 19722: #", coordinate, "#")
      assert False, (index_sfx[coordinate], coordinate)
      return 1.0
   print((correctFull+incorrectFull))
   return correct/(correct+incorrect), correctFull/(correctFull+incorrectFull)




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



result = getCorrectOrderCount(weights_pfx, None, 0)
print(errors)
print(result)

with open("/u/scr/mhahn/deps/memory-need-ngrams-morphology-accuracy/accuracy_"+__file__+"_"+str(myID)+"_"+args.model+".txt", "w") as outFile:
   print(result[0], file=outFile)
   print(result[1], file=outFile)
   errors = list(errors.items())
   errors.sort(key=lambda x:x[1], reverse=True)
   for x, y in errors:
      print(x[0], x[1], y, file=outFile)
print("/u/scr/mhahn/deps/memory-need-ngrams-morphology-accuracy/accuracy_"+__file__+"_"+str(myID)+"_"+args.model+".txt")


