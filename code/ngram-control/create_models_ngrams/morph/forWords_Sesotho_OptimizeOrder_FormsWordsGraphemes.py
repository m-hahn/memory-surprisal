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
parser.add_argument("--cutoff", dest="cutoff", type=int, default=3)
parser.add_argument("--idForProcess", dest="idForProcess", type=int, default=random.randint(0,10000000))
import random



args=parser.parse_args()
print(args)


assert args.alpha >= 0
assert args.alpha <= 1
assert args.delta >= 0
assert args.gamma >= 1


RELEVANT_KEY = "lemma"


myID = args.idForProcess


TARGET_DIR = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"

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
    affixLemma = affix[header[RELEVANT_KEY]]
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

  


contextLength = args.cutoff
lengthForPrediction = args.cutoff

def calculateTradeoffForWeights(weights_pfx):
    wordsWithoutSentinels = 0
    dev = []
    for _ in range(args.cutoff+2):
      dev.append(stoi_words["PAD"])
    dev.append(stoi_words["SOS"])
    wordsWithoutSentinels += 1

    numberOfRelevantSentences = 0
    for verb in data:
       numberOfRelevantSentences += 1

       prefixes = [x for x in verb if x[header["type1"]] == "pfx"]
       suffixes = [x for x in verb if x[header["type1"]] == "sfx"]
       v = [x for x in verb if x[header["type1"]] == "v"]
       assert len(prefixes)+len(v)+len(suffixes)==len(verb)

       prefixes.sort(key=lambda x:weights_pfx[x[header[RELEVANT_KEY]]])
       ordered = prefixes + v + suffixes

       dev.append(stoi_words["SOS"])
       wordsWithoutSentinels += 1
       for ch in ordered:
         for char in ch[header["form"]]:
           dev.append(stoi_words[char])
           wordsWithoutSentinels += 1
       dev.append(stoi_words["EOS"])
       wordsWithoutSentinels += 1
       for _ in range(args.cutoff+2):
         dev.append(stoi_words["PAD"])
#    print(len([x for x in dev if x != 0]), wordsWithoutSentinels)
   
    array = dev    
    #print array
    
    print("Sorting "+str(numberOfRelevantSentences)+" sentences.")
    if numberOfRelevantSentences == 0:
        return 1.0
    indices = range(len(array))
    indices = sorted(indices, key=lambda x:array[x:x+args.cutoff])
    #print indices
    print("Now calculating information")
    
    # bigram surprisal
    startOfCurrentPrefix = None
    
    
    endPerStart = [None for _ in range(len(indices))]
    endPerStart[0] = len(endPerStart)-1
    
    lastCrossEntropy = 10000
    memory = 0
    
    devSurprisalTable = []
    
    for contextLength in range(0, args.cutoff-1):
          crossEntropy = 0
          totalSum = 0
          i = 0
          lengthsOfSuffixes = 0
          countTowardsSurprisal = 0
          while i < len(indices):
       #        if indices[i]+contextLength >= len(array):
       #          startOfCurrentSuffix += 1
       #          i += 1
               while endPerStart[i] is None:
                   i += 1
               endOfCurrentPrefix = endPerStart[i] # here, end means the last index (not the one where a new thing starts)
               endPerStart[i] = None
               assert endOfCurrentPrefix is not None, (i, len(indices))
               # now we know the range where the current prefix occurs
               countOfCurrentPrefix = ((endOfCurrentPrefix-i+1)) # if contextLength >= 1 else wordsWithoutSentinels)
               assert countOfCurrentPrefix >= 1
               startOfCurrentSuffix = i
       #        if indices[startOfCurrentSuffix]+contextLength >= len(array):
       #          startOfCurrentSuffix += 1
       #          i += 1
               j = i
               firstNonSentinelSuffixForThisPrefix = i # by default, will be modified in time in case sentinels show up
               probSumForThisPrefix = 0
               while j <= endOfCurrentPrefix:
                    # is j the last one?
       
                    assert j == endOfCurrentPrefix or j+1 < len(indices), (i,j)
       
                    assert j < len(indices)
       
                    # when there is nothing to predict
                    if indices[j]+contextLength >= len(array):
                        j+=1
                        startOfCurrentSuffix+=1
                        continue 
       #
                    # don't want to be predicting the final "0" tokens
       #             if array[indices[j]+contextLength] == 0:
       #                 j+=1
       #                 startOfCurrentSuffixExceptSentinels+=1
       #                 continue
       
                    assert indices[startOfCurrentSuffix]+contextLength < len(array), (i,j)
                    assert j >= i
                    assert endOfCurrentPrefix >= j
                    if j == endOfCurrentPrefix or indices[j+1]+contextLength >= len(array) or array[indices[j+1]+contextLength] != array[indices[startOfCurrentSuffix]+contextLength]:
                      endOfCurrentSuffix = j # here, end means the last index (not the one where a new thing starts)
       #               print (i, j)
                      lengthOfCurrentSuffix =  endOfCurrentSuffix - startOfCurrentSuffix + 1
                      lengthsOfSuffixes += lengthOfCurrentSuffix
       
                      if array[indices[startOfCurrentSuffix]+contextLength] != 0: # don't incur loss for predicting sentinel
                         countOfCurrentPrefixWithoutSentinelSuffix = endOfCurrentPrefix - firstNonSentinelSuffixForThisPrefix + 1 # here important that sentinel comes first when sorting (is 0)
                         assert countOfCurrentPrefixWithoutSentinelSuffix <= countOfCurrentPrefix, ["endOfCurrentPrefix", endOfCurrentPrefix, "firstNonSentinelSuffixForThisPrefix", firstNonSentinelSuffixForThisPrefix, "i", i]
                         conditionalProbability = float(lengthOfCurrentSuffix) / countOfCurrentPrefixWithoutSentinelSuffix
                         probSumForThisPrefix += conditionalProbability
                         surprisal = -log(conditionalProbability)
                         probabilityThatThisSurprisalIsIncurred = float(lengthOfCurrentSuffix) / wordsWithoutSentinels
                         crossEntropy += probabilityThatThisSurprisalIsIncurred * surprisal
                         totalSum += probabilityThatThisSurprisalIsIncurred
                         countTowardsSurprisal += lengthOfCurrentSuffix
                      else:
                         firstNonSentinelSuffixForThisPrefix = j+1
                      endPerStart[startOfCurrentSuffix] = endOfCurrentSuffix
                      startOfCurrentSuffix = j+1
                    if j == endOfCurrentPrefix:
                      break
                    if indices[j+1]+contextLength >= len(array):
                       startOfCurrentSuffix = j+2
                       j+=2
                    else:
                       j+=1
               i = endOfCurrentPrefix+1
               assert lengthsOfSuffixes >= i-contextLength
               assert min(abs(probSumForThisPrefix - 0.0), abs(probSumForThisPrefix - 1.0)) < 0.00001, probSumForThisPrefix
          assert i-lengthsOfSuffixes == contextLength
          #assert lastCrossEntropy >= crossEntropy
       
          print("==================================================countTowardsSurprisal", countTowardsSurprisal)
          memory += min(lengthForPrediction, contextLength) * (lastCrossEntropy-crossEntropy)
          print("CONTEXT LENGTH "+str(contextLength)+"   "+str( crossEntropy)+"  "+str((lastCrossEntropy-crossEntropy))+"   "+str(memory))
          assert abs(totalSum - 1.0) < 0.00001, totalSum
       
     #   print >> outFile, "\t".join(map(str,["FileName","ModelName","Counter", "Model", "Distance", "Entropy", "ConditionalMI", "Memory"]))
      
          #print("\t".join(map(str,[myID, __file__, 100000, model, originalCounter, contextLength, crossEntropy, (lastCrossEntropy-crossEntropy) if contextLength > 0 else crossEntropy, memory, lengthForPrediction, contextLength ])))
          lastCrossEntropy = crossEntropy
          devSurprisalTable.append(crossEntropy)
    #  lengthForPrediction = int(sys.argv[4]) #20
    #contextLength = int(sys.argv[5]) #20
    print(devSurprisalTable)
    mis = [devSurprisalTable[i] - devSurprisalTable[i+1] for i in range(len(devSurprisalTable)-1)]
    tmis = [mis[x]*(x+1) for x in range(len(mis))]
    #print(mis)
    #print(tmis)
    auc = 0
    memory = 0
    mi = 0
    for i in range(len(mis)):
       mi += mis[i]
       memory += tmis[i]
       auc += mi * tmis[i]
    #print("MaxMemory", memory)
    assert 20>memory, memory
    auc += mi * (20-memory)
    #print("AUC", auc)
    return auc
    #assert False
    
    #outpath = TARGET_DIR+"/estimates-"+args.language+"_"+__file__+"_model_"+str(myID)+"_"+args.model+".txt"
    #print(outpath)
    #with open(outpath, "w") as outFile:
    #         print >> outFile, str(args)
    #         print >> outFile, devSurprisalTable[-1]
    #         print >> outFile, " ".join(map(str,devSurprisalTable))
    #
    #
   


fullAUCs = []

words = set()

# For each verb form, select only the main verb form
for q in range(len(data)):
   verb = data[q]
   prefixes_keys = [x[header[RELEVANT_KEY]] for x in verb if x[header["type1"]] == "pfx"]

   segmentation = []
   for j in range(len(verb)):
      # subject prefix?
      if ".SBJ" in verb[j][header["analysis"]]:
         segmentation.append([])
         segmentation[-1].append(verb[j])
      else:
         if len(segmentation) == 0:
           segmentation.append([])
         segmentation[-1].append(verb[j])
   ###############################################################################   
   # Restrict to the last verb, chopping off initial auxiliaries and their affixes
   ###############################################################################

   verb = segmentation[-1]

   data[q] = verb
   for word in verb:
     for ch in word[header["form"]]:
       words.add(ch)


words = list(words)
itos_words = ["PAD", "SOS", "EOS"] + words
stoi_words = dict(zip(itos_words, range(len(itos_words))))
print(stoi_words)


for iteration in range(1000):
  coordinate = choice(itos_pfx)
  while prefixFrequency[coordinate] < 10 and random() < 0.95:
    coordinate = choice(itos_pfx)
  mostCorrect, mostCorrectValue = 0, None
  for newValue in [-1] + [2*x+1 for x in range(len(itos_pfx))] + [weights_pfx[coordinate]]:
     if random() < 0.8 and newValue != weights_pfx[coordinate]:
        continue
     print(newValue, mostCorrect, coordinate,prefixFrequency[coordinate])
     weights_ = {x : y if x != coordinate else newValue for x, y in weights_pfx.items()}
     correctCount = calculateTradeoffForWeights(weights_)
#     print(weights_)
#     print(coordinate, newValue, iteration, correctCount)
     if correctCount > mostCorrect:
        mostCorrectValue = newValue
        mostCorrect = correctCount
  assert not (mostCorrectValue is None)
  print(iteration, mostCorrect)
  weights_pfx[coordinate] = mostCorrectValue
  itos_pfx_ = sorted(itos_pfx, key=lambda x:weights_pfx[x])
  weights_pfx = dict(list(zip(itos_pfx_, [2*x for x in range(len(itos_pfx_))])))
  print(weights_pfx)
  for x in itos_pfx_:
     if prefixFrequency[x] < 10:
       continue
     print("\t".join([str(y) for y in [x, weights_pfx[x], prefixFrequency[x]]]))
  if (iteration + 1) % 50 == 0:
     fullAUCs.append(calculateTradeoffForWeights(weights_))
     with open(TARGET_DIR+"/optimized_"+args.language+"_"+__file__+"_"+str(myID)+".tsv", "w") as outFile:
        print(iteration, mostCorrect, file=outFile)
        print(" ".join([str(x) for x in fullAUCs]), file=outFile)
        print(str(args), file=outFile)
        for key in itos_pfx_:
           print(key, weights_pfx[key], file=outFile)



