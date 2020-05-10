

productions = {}

import random

basic = random.choice(["verb-final", "verb-medial", "half-half"])
case = random.choice(["before", "after"])
npOrder = random.choice(["APN", "ANP", "PAN", "PNA", "NAP", "NPA"])

productions["S"] = [(0.5, ("NP-heavy", "di", "N_a", "V", ".")), (0.5, ("NP-heavy", "N_a", "di", "V", "."))]
#productions["S"] += [(0.25, ("N_a", "di", "NP-heavy", "V", ".")), (0.25, ("N_a", "NP-heavy", "di" "V", "."))]

productions["NP-heavy"] = [(1.0, ("Adj", "N_i", "P", "N_a"))]



sentence1 = [{"index" : 1, "word" : "Adj", "head" : 2, "dep" : "amod"}, {"index" : 2, "word" : "N_i", "head" : 4, "dep" : "nmod"}, {"index" : 3, "word" : "P", "head" : 2, "dep" : "case"}, {"index" : 4, "word" : "N_a", "head" : 7, "dep" : "obj_heavy"}, {"index" : 5, "word" : "di", "head" : 4, "dep" : "case"}, {"index" : 6, "word" : "N_a", "head" : 7, "dep" : "nsubj"}, {"index" : 7, "word" : "V", "head" : -1, "dep" : "root"}, {"index" : 8, "word" : ".", "head" : 7, "dep" : "punct"}]
sentence2 = [{"index" : 1, "word" : "Adj", "head" : 2, "dep" : "amod"}, {"index" : 2, "word" : "N_i", "head" : 4, "dep" : "nmod"}, {"index" : 3, "word" : "P", "head" : 2, "dep" : "case"}, {"index" : 4, "word" : "N_a", "head" : 7, "dep" : "nsubj_heavy"}, {"index" : 5, "word" : "N_a", "head" : 7, "dep" : "obj"}, {"index" : 6, "word" : "di", "head" : 5, "dep" : "case"} ,    {"index" : 7, "word" : "V", "head" : -1, "dep" : "root"}, {"index" : 8, "word" : ".", "head" : 7, "dep" : "punct"}]


sentences = [sentence1, sentence2]

itos_deps = list(set([x["dep"] for x in sentence1+sentence2]))
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

import torch

dhWeights = torch.FloatTensor([0.0] * len(itos_deps))
distanceWeights = torch.FloatTensor([0.0] * len(itos_deps))

dhWeights[stoi_deps["amod"]] = random.random()-0.5 #random.choice([-10, 10])
dhWeights[stoi_deps["nmod"]] =random.random()-0.5 # random.choice([-10, 10])
dhWeights[stoi_deps["case"]] =random.random()-0.5 # random.choice([-10, 10])
OBJ = random.random()-0.5 #choice([-10, 10])
SUBJ = random.random()-0.5 #choice([-10, 10])
dhWeights[stoi_deps["obj"]] = OBJ #random.choice([-10, 10])
dhWeights[stoi_deps["nsubj"]] = SUBJ #random.choice([-10, 10])
dhWeights[stoi_deps["obj_heavy"]] = OBJ #random.choice([-10, 10])
dhWeights[stoi_deps["nsubj_heavy"]] = SUBJ #random.choice([-10, 10])

indices = list(range(len(itos_deps)))
random.shuffle(indices)
for x in range(len(indices)):
  distanceWeights[x] = random.random()
#if random.random() > 0.5:
#   LIGHT = -10
#   HEAVY = 10
#else:
#   LIGHT = 10
#   HEAVY = -10
#distanceWeights[stoi_deps["nsubj"]] = LIGHT
#distanceWeights[stoi_deps["obj"]] = LIGHT
#distanceWeights[stoi_deps["nsubj_heavy"]] = HEAVY
#distanceWeights[stoi_deps["obj_heavy"]] = HEAVY

#distanceWeights[stoi_deps["nmod"]] = 0
#distanceWeights[stoi_deps["case"]] = -10
#distanceWeights[stoi_deps["amod"]] = 10





import torch.nn as nn
import torch
from torch.autograd import Variable


def recursivelyLinearize(sentence, position, result):
   line = sentence[position-1]


   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         recursivelyLinearize(sentence, child, result)
   result.append(line)
   if "children_HD" in line:
      for child in line["children_HD"]:
         recursivelyLinearize(sentence, child, result)

import numpy.random
import numpy as np

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()
logsoftmaxLabels =  torch.nn.LogSoftmax(dim=2)



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       childrenLinearized = []
       while len(remainingChildren) > 0:
           logits = torch.cat([distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]].view(1) for x in remainingChildren])
           #print logits
           if reverseSoftmax:
              logits = -logits
           softmax = softmax_layer(logits.view(1,-1)).view(-1)
           selected = numpy.random.choice(range(0, len(remainingChildren)), p=softmax.data.numpy())
           log_probability = torch.log(softmax[selected])
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
       return childrenLinearized           


def orderSentence(sentence, printThings=False):
   root = None
   logits = [None]*len(sentence)
   for line in sentence:
      line["coarse_dep"] = line["dep"]
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      if line["coarse_dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         continue
      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      probability = 1/(1 + torch.exp(-dhLogit))
      dhSampled = (random.random() < probability.data.numpy())

      
     
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print("\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, (str(probability.data.numpy())+"      ")[:8], str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]].data.numpy())+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  )))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])



   for line in sentence:
      assert "haveVisited" not in  line, line
      line["haveVisited"] = True
      if "children_DH" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
         line["children_DH"] = childrenLinearized
      if "children_HD" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
         line["children_HD"] = childrenLinearized

#         shuffle(line["children_HD"])
   
   linearized = []
   recursivelyLinearize(sentence, root, linearized)
   if printThings or len(linearized) == 0:
     print( " ".join(map(lambda x:x["word"], sentence)))
     print( " ".join(map(lambda x:x["word"], linearized)))


#   # store new dependency links
#   moved = [None] * len(sentence)
#   for i, x in enumerate(linearized):
##      print x
#      moved[x["index"]-1] = i
# #  print moved
#   for i,x in enumerate(linearized):
#  #    print x
#      if x["head"] == 0: # root
#         x["reordered_head"] = 0
#      else:
#         x["reordered_head"] = 1+moved[x["head"]-1]
   return linearized, logits





import numpy as np

def sample(x):
    if x in productions:
        probs = [y[0] for y in productions[x]]
        return " ".join([sample(z) for z in productions[x][np.random.choice(range(len(probs)), p=probs)][1]])
    return x

vocabulary = set()
for x in productions:
      for _, rhs in productions[x]:
          for word in rhs:
              if word not in productions:
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
from random import shuffle
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



def deepCopy(y):
  return [dict(list(x.items())[::]) for x in y]

def corpusIterator():
   for i in range(10000):
       if i % 100 == 0:
          print(i)
       sent, _=  orderSentence(deepCopy(random.choice(sentences)))  #  sample("S").split(" ")
       yield [x["word"] for x in sent]

for sentence in corpusIterator():
#     print(sentence)
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

print("Sorting")
indices = range(len(array))
indices = sorted(indices, key=lambda x:array[x:x+contextLength])
#print indices
print("Now calculating information")

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

