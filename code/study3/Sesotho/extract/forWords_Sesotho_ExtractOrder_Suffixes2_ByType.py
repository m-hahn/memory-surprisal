# based on yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py

import random
import sys

objectiveName = "LM"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="Sesotho_Acqdiv")
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


#RELEVANT_KEY = "lemma"

# for ordering
def getKey(word):
  return word[header["lemma"]][:2]

def getSegmentedFormsVerb(word):
   if "/" not in word[header["lemma"]] and "." not in word[header["lemma"]]:
     return [word]
   elif "/" in word[header["lemma"]]:
    lemmas = word[header["lemma"]].split("/")
    words = [word[::] for _ in lemmas]
    for i in range(len(lemmas)):
      words[i][0] = "_"
      words[i][1] = lemmas[i]
      words[i][3] = "v" if i == 0 else "sfx"      
    #print("SPLIT", words, word) # frequent: verb stem + past suffix merged
    return words
   else: # 
    print("TODO", word)
    assert False


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
    if lemmas[0] == "t^pf" and lemmas[1] == "m^in": # ~360 cases, mostly -e-. TODO think about the order of the morphemes in the allegedly merged morpheme.
      _ = 0
    elif word[header["analysis"]] == "REVERS.CAUS": # os (Doke and Mokofeng, section 345)
      _ = 0
    elif word[header["analysis"]] == "APPL.PRF": # ets (cf. Doke and Mokofeng, section 313?). Both APPL and PRF have relatively frequent suffix morphs of the form -ets- in the corpus.
      _ = 0
    elif word[header["analysis"]] == "PRF.CAUS": # dits. Also consider Doke and Mokofeng, section 369, rule 4.
      _ = 0
    elif word[header["analysis"]] == "DEP.PRF": #  e. DEP = participial mood (Doke and Mokofeng, section 431).
      _ = 0
    elif word[header["analysis"]] in ["PRF.PASS", "PRS.APPL", "cl.PRF", "IND.PRS", "PRF.REVERS", "NEG.PRF"]: # rare, together 10 data points
      _ = 0
    else:
      print("SPLIT", word1, word2, word)
    return [word1, word2]
   else: # 
    print("TODO", word)
    assert word[1] == "m..." # occurs 1 time
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

assert len(data) > 0
prefixFrequency = defaultdict(int)
suffixFrequency = defaultdict(int)
#dataChosen = []
for verbWithAff in data:
  for affix in verbWithAff:
    affixLemma = getKey(affix) #[header[RELEVANT_KEY]]
    if affix[header["type1"]] == "pfx":
       prefixFrequency[affixLemma] += 1
    elif affix[header["type1"]] == "sfx":
       suffixFrequency[affixLemma] += 1
#assert len(dataChosen) > 0
#data = dataChosen

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


def getCorrectOrderCount(weights_sfx, coordinate, newValue):
   correct = 0
   incorrect = 0
   if len(index_sfx[coordinate]) == 0:
     return 1.0
   for verb in index_sfx[coordinate]:
      #prefixes_keys = [x[header["form"]] for x in verb if x[header["type1"]] == "sfx"]
#      if coordinate not in prefixes_keys:
 #       assert False
  #      continue
   
      suffixes = [(getKey(x), weights_sfx[getKey(x)]) for x in verb if x[header["type1"]] == "sfx"]
      assert len(suffixes) > 1, verb
#      suffixes = [(x[header[RELEVANT_KEY]], weights_sfx[x[header[RELEVANT_KEY]]]) for x in verb if x[header["type1"]] == "sfx"]

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
 #                print("ERROR", q)
  #               print((affixes[i][0], affixes[j][0]))
   #              print(verb)
#                 if affixes[i][0] == affixes[j][0]:
 #                     assert False
               #  errors[(affixes[i][0], affixes[j][0])] += 1
        assert correct+incorrect>0, affixes
   if correct+incorrect == 0:
      print("ERROR 19722: #", coordinate, "#")
      assert False, (index_sfx[coordinate], coordinate)
      return 1.0
   return correct/(correct+incorrect)

index_sfx = defaultdict(list)

elementsOccurringBeforeSubject = defaultdict(int)


print(len(data))
#quit()

for q in range(len(data)):
   verb = data[q]
 #  print(verb)
#   prefixes_keys = [getKey(x) for x in verb if x[header["type1"]] == "pfx"]
 #  if len(prefixes_keys) <= 1:
  #   continue
#   print(verb)
   for i in range(len(verb)-1):
      if ".SBJ" in verb[i+1][header["analysis"]]:
         if not verb[i][header["lemma"]].startswith("t^"):
           if not verb[i][header["lemma"]] == "rl": # suffix -ng to the auxiliary
 #           print(verb[i], verb)
            elementsOccurringBeforeSubject[tuple(verb[i])] += 1
   # make it hierarchical by counting subj... subj...

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

#   print(segmentation[-1])
#   if len(segmentation) > 1:
#    for w in range(len(segmentation)-1):
#     if len(segmentation[w]) == 2:
#         _ = 0
#     else:
#        print(segmentation[w])


   verb = segmentation[-1]

   suffixes_keys = [getKey(x) for x in verb if x[header["type1"]] == "sfx"]

#   print(suffixes_keys)
   # It is important to overwrite data[q] before continuing
   data[q] = verb

   if len(suffixes_keys) <= 1:
     continue


   for sfx in suffixes_keys:
      index_sfx[sfx].append(verb)
   index_sfx[None].append(verb)




print(sorted(list(elementsOccurringBeforeSubject.items()), key=lambda x:x[1]))
#print(len(data))
#quit()
#quit()


for iteration in range(200):
  coordinate = choice(itos_sfx)
  while suffixFrequency[coordinate] < 10 and random() < 0.95:
    coordinate = choice(itos_sfx)
  mostCorrect, mostCorrectValue = 0, None
  for newValue in [-1] + [2*x+1 for x in range(len(itos_sfx))]:
     #if random() > 0.3:
     #  continue
     correctCount = getCorrectOrderCount(weights_sfx, coordinate, newValue)
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
  print(iteration, mostCorrect, suffixFrequency[coordinate], len(index_sfx[coordinate]))
  print("Total", getCorrectOrderCount(weights_sfx, None, 0))
with open("output/extracted_"+__file__+"_"+str(myID)+".tsv", "w") as outFile:
  for x in itos_sfx_:
  #   if affixFrequencies[x] < 10:
   #    continue
     print("\t".join([str(y) for y in [x, weights_sfx[x], suffixFrequency[x]]]), file=outFile)


