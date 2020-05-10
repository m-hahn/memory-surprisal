

productions = {}

productions["S"] = [(0.5, ("Obj", "Subj", "V", ".")), (0.5, ("Subj", "Obj", "V", "."))]
productions["Obj"] = [(1.0, ("NP", "di"))]
productions["Subj"] = [(1.0, ("NP"))]
productions["NP"] = [(0.75, ("N")), (0.125, ("PP", "NP")), (0.125, ("Adj", "NP"))]
productions["PP"] = [(1.0, ("NP", "P"))]


import numpy as np
import random

def sample(x):
    if x in productions:
        probs = [y[0] for y in productions[x]]
        return " ".join([sample(z) for z in productions[x][np.random.choice(range(len(probs)), p=probs)][1]])
    return x

vocabulary = set()
for x in productions:
      for _, rhs in productions[x]:
          for word in rhs:
              if word not in vocabulary:
                  vocabulary.add(word)
itos = list(vocabulary)
stoi = dict(zip(itos, range(len(itos))))
print("Itos")
print(itos)

print(sample("S"))




import random
import sys
from collections import deque

lengthForPrediction = 20 #int(sys.argv[4]) #20
contextLength = 20 #int(sys.argv[5]) #20




myID = random.randint(0,10000000)

posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]




from math import log, exp
from random import random, shuffle
import os


originalDistanceWeights = {}

import numpy.random




batchSize = 1

lr_lm = 0.1


crossEntropy = 10.0

def encodeWord(w):
   return stoi[w]+3 if stoi[w] < vocab_size else 1

counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 


totalWordNum = 0
assert batchSize == 1
assert batchSize == 1

depLengths = []

array = []

wordsWithoutSentinels = 0

#corpus = CorpusIterator("train")
#corpusIterator = corpus.iterator()
#if corpus.length() == 0:
#   quit()
print("Building suffix array")
sentcounter = 0
#for _ in range(contextLength):
#   array.append(0)


def corpusIterator():
   for _ in range(10000):
       yield sample("S").split(" ")

for sentence in corpusIterator():
     sentcounter += 1
     for _ in range(contextLength):
        array.append(0)
     array.append(1)
     wordsWithoutSentinels += 1

     for word in sentence:
         array.append(stoi[word]+3)
         wordsWithoutSentinels += 1
     array.append(1)
     wordsWithoutSentinels += 1

#     if sentcounter == 1:
#       break
#array.append(1)
#for _ in range(contextLength):
#array.append(1)
for _ in range(contextLength):
   array.append(0)

#print array

print "Sorting"
indices = range(len(array))
indices = sorted(indices, key=lambda x:array[x:x+contextLength])
#print indices
print "Now calculating information"

# bigram surprisal
startOfCurrentPrefix = None


endPerStart = [None for _ in range(len(indices))]
endPerStart[0] = len(endPerStart)-1

lastCrossEntropy = 10000
memory = 0


print("Saving")
save_path = "/juicier/scr120/scr/mhahn/deps/"

my_fileName = __file__.split("/")[-1]

if True:

   for contextLength in range(0, contextLength-1):
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
   
      print(countTowardsSurprisal)
      memory += min(lengthForPrediction, contextLength) * (lastCrossEntropy-crossEntropy)
      print("CONTEXT LENGTH "+str(contextLength)+"   "+str( crossEntropy)+"  "+str((lastCrossEntropy-crossEntropy))+"   "+str(memory))
      assert abs(totalSum - 1.0) < 0.00001, totalSum
   
      lastCrossEntropy = crossEntropy

