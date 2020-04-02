# based on yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py

import random
import sys

objectiveName = "LM"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="Sesotho_Acqdiv")
parser.add_argument("--model_pfx", type=str)
parser.add_argument("--model_sfx", type=str)
parser.add_argument("--alpha", dest="alpha", type=float, default=1.0)
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
#    print("SPLIT", words, word) # frequent: verb stem + past suffix merged
    return words
   else: # 
    print("TODO", word)
    assert False


def getSegmentedForms_Suffix(word): # return a list , preprocessing
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



def getSegmentedForms_Prefix(word): # return a list , preprocessing
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
   return word[header["lemma"]]

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
Random(0).shuffle(words)
data_train = words[int(0.05*len(words)):]
data_dev = words[:int(0.05*len(words))]


#data = words




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
dataChosen_train = []
dataChosen_dev = []
for data_, dataChosen in [(data_train, dataChosen_train), (data_dev, dataChosen_dev)]:
  for verbWithAff in data_:
    affixesResult = []
    for x in verbWithAff:
      if x[header["type1"]] == "pfx":
         segmented = getSegmentedForms_Prefix(x)
         if segmented is None:
           prefixesResult = None
           break
         affixesResult += segmented
      elif x[header["type1"]] == "sfx":
         segmented = getSegmentedForms_Suffix(x)
         if segmented is None:
           suffixesResult = None
           break
         affixesResult += segmented
      elif x[header["type1"]] == "v":
         segmented = getSegmentedFormsVerb(x)
         affixesResult += segmented
      else:
         assert False
    if affixesResult is None: # remove this datapoint (affects <20 datapoints)
       continue
    dataChosen.append(affixesResult)
    for affix in affixesResult:
      affixLemma = getKey(affix) #[header[RELEVANT_KEY]]
      if affix[header["type1"]] == "pfx":
         prefixFrequency[affixLemma] += 1
      elif affix[header["type1"]] == "sfx":
         suffixFrequency[affixLemma] += 1
data_train = dataChosen_train
data_dev = dataChosen_dev


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

import glob
PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized"

if args.model_sfx not in ["REAL", "RANDOM", "REVERSE"]:
  files = glob.glob(PATH+"/optimized_*.py_"+args.model_sfx+".tsv")
  print(files, args.model_sfx)
  assert len(files) == 1
  assert "Suffixes" in files[0], files
  assert "Normalized" in files[0]
  with open(files[0], "r") as inFile:
     next(inFile)
     next(inFile)
     next(inFile)
     for line in inFile:
        morpheme, weight = line.strip().split(" ")
        weights_sfx[morpheme] = int(weight)
#
if args.model_pfx not in ["REAL", "RANDOM", "REVERSE"]:
  files = glob.glob(PATH+"/optimized_*.py_"+args.model_pfx+".tsv")
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

def calculateTradeoffForWeights(weights_pfx, weights_sfx):
    train = []
    dev = []
    for data, processed in [(data_train, train), (data_dev, dev)]:
      for verb in data:
         affixes = verb
         prefixes = [x for x in verb if x[header["type1"]] == "pfx"]
         suffixes = [x for x in verb if x[header["type1"]] == "sfx"]
         v = [x for x in verb if x[header["type1"]] == "v"]
         assert len(prefixes)+len(v)+len(suffixes)==len(verb)
         if args.model_sfx == "REAL":
            _ = 0
         elif args.model_sfx == "REVERSE":
            suffixes = suffixes[::-1]
         else:
            suffixes.sort(key=lambda x:weights_sfx[getKey(x)])
         if args.model_pfx == "REAL":
            _ = 0
         elif args.model_pfx == "REVERSE":
            prefixes = prefixes[::-1]
         else:
            prefixes.sort(key=lambda x:weights_pfx[getKey(x)])
         ordered = prefixes + v + suffixes
  
         for ch in ordered:
             processed.append(getNormalizedForm(ch))
         processed.append("EOS")
         for _ in range(args.cutoff+2):
           processed.append("PAD")
         processed.append("SOS")
    
    itos = list(set(dev) | set(train))
    
    
    dev = dev[::-1]
    #dev = list(createStreamContinuous(corpusDev))[::-1]
    
    
    #corpusTrain = CorpusIterator(args.language,"dev", storeMorph=True).iterator(rejectShortSentences = False)
    #train = list(createStreamContinuous(corpusTrain))[::-1]
    train = train[::-1]
    
    idev = range(len(dev))
    itrain = range(len(train))
    
    idev = sorted(idev, key=lambda i:dev[i:i+20])
    itrain = sorted(itrain, key=lambda i:train[i:i+20])
    
#    print(idev)
    
    idevInv = [x[1] for x in sorted(zip(idev, range(len(idev))), key=lambda x:x[0])]
    itrainInv = [x[1] for x in sorted(zip(itrain, range(len(itrain))), key=lambda x:x[0])]
    
    assert idev[idevInv[5]] == 5
    assert itrain[itrainInv[5]] == 5
    
    
    
    def getStartEnd(k):
       start = [0 for _ in dev]
       end = [len(train)-1 for _ in dev]
       if k == 0:
          return start, end
       # Start is the FIRST train place that is >=
       # End is the FIRST train place that is >
       l = 0
       l2 = 0
       for j in range(len(dev)):
         prefix = tuple(dev[idev[j]:idev[j]+k])
         while l2 < len(train):
            prefix2 = tuple(train[itrain[l2]:itrain[l2]+k])
            if prefix <= prefix2:
                 start[j] = l2
                 break
            l2 += 1
         if l2 == len(train):
            start[j] = l2
         while l < len(train):
            prefix2 = tuple(train[itrain[l]:itrain[l]+k])
            if prefix < prefix2:
                 end[j] = l
                 break
            l += 1
         if l == len(train):
            end[j] = l
         start2, end2 = start[j], end[j]
         assert start2 <= end2
         if start2 > 0 and end2 < len(train):
           assert prefix > tuple(train[itrain[start2-1]:itrain[start2-1]+k])
           assert prefix <= tuple(train[itrain[start2]:itrain[start2]+k])
           assert prefix >= tuple(train[itrain[end2-1]:itrain[end2-1]+k])
           assert prefix < tuple(train[itrain[end2]:itrain[end2]+k])
       return start, end
    
    
    lastProbability = [None for _ in idev]
    newProbability = [None for _ in idev]
    
    devSurprisalTable = []
    for k in range(0,args.cutoff):
#       print(k)
       startK, endK = getStartEnd(k) # Possible speed optimization: There is some redundant computation here, could be reused from the previous iteration. But the algorithm is very fast already.
       startK2, endK2 = getStartEnd(k+1)
       cachedFollowingCounts = {}
       for j in range(len(idev)):
    #      print(dev[idev[j]])
          if dev[idev[j]] in ["PAD", "SOS"]:
             continue
          start2, end2 = startK2[j], endK2[j]
          devPref = tuple(dev[idev[j]:idev[j]+k+1])
          if start2 > 0 and end2 < len(train):
            assert devPref > tuple(train[itrain[start2-1]:itrain[start2-1]+k+1]), (devPref, tuple(train[itrain[start2-1]:itrain[start2-1]+k+1]))
            assert devPref <= tuple(train[itrain[start2]:itrain[start2]+k+1]), (devPref, tuple(train[itrain[start2]:itrain[start2]+k+1]))
            assert devPref >= tuple(train[itrain[end2-1]:itrain[end2-1]+k+1])
            assert devPref < tuple(train[itrain[end2]:itrain[end2]+k+1])
    
          assert start2 <= end2
    
          countNgram = end2-start2
          if k >= 1:
             if idev[j]+1 < len(idevInv):
               prefixIndex = idevInv[idev[j]+1]
               assert dev[idev[prefixIndex]] == dev[idev[j]+1]
       
               prefixStart, prefixEnd = startK[prefixIndex], endK[prefixIndex]
               countPrefix = prefixEnd-prefixStart
               if countPrefix < args.gamma: # there is nothing to interpolate with, just back off
                  assert k > 0
                  newProbability[j] = lastProbability[j]
               else:
                  assert countPrefix >= countNgram, (countPrefix, countNgram)
       
                  following = set()
                  if (prefixStart, prefixEnd) in cachedFollowingCounts:
                      followingCount = cachedFollowingCounts[(prefixStart, prefixEnd)]
                  else:
                    for l in range(prefixStart, prefixEnd):
                      if k < itrain[l]+1:
                         following.add(train[itrain[l]-1])
                         assert devPref[1:] == tuple(train[itrain[l]-1:itrain[l]+k])[1:], (k, itrain[l], l, devPref , tuple(train[itrain[l]-1:itrain[l]+k]))
                    followingCount = len(following)
                    cachedFollowingCounts[(prefixStart, prefixEnd)] = followingCount
                  if followingCount == 0:
                      newProbability[j] = lastProbability[j]
                  else:
                      #assert countNgram > 0
                      probability = log(max(countNgram - args.alpha, 0.0) + args.alpha * followingCount * exp(lastProbability[j])) -  log(countPrefix)
                      newProbability[j] = probability
             else:
                newProbability[j] = lastProbability[j]
          elif k == 0:
                  probability = log(countNgram + args.delta) - log(len(train) + args.delta * len(itos))
                  newProbability[j] = probability
       lastProbability = newProbability 
       newProbability = [None for _ in idev]
       assert all([x is None or x <=0 for x in lastProbability])
       try:
           lastProbabilityFiltered = [x for x in lastProbability if x is not None]
           surprisal = - sum([x for x in lastProbabilityFiltered])/len(lastProbabilityFiltered)
       except ValueError:
           print >> sys.stderr, "PROBLEM"
           print >> sys.stderr, lastProbability
           surprisal = 1000
       devSurprisalTable.append(surprisal)
       print("Surprisal", surprisal, len(itos))
    print(devSurprisalTable)
    for k in range(len(devSurprisalTable)):
        devSurprisalTable[k] = min(devSurprisalTable[:k+1])
    mis = [devSurprisalTable[i] - devSurprisalTable[i+1] for i in range(len(devSurprisalTable)-1)]
    tmis = [mis[x]*(x+1) for x in range(len(mis))]
    #print(mis)
    #print(tmis)
    auc = 0
    memory = 0
    mi = 0
    print(mis)
    print(tmis)
    for i in range(len(mis)):
       mi += mis[i]
       memory += tmis[i]
       auc += mi * tmis[i]
    print("MaxMemory", memory)
    assert 7>memory
    auc += mi * (7-memory)
    print("AUC", auc)
    #assert False
    
    outpath = TARGET_DIR+"/estimates-"+args.language+"_"+__file__+"_model_"+str(myID)+"_"+args.model_pfx+"_"+args.model_sfx+".txt"
    print(outpath)
    with open(outpath, "w") as outFile:
       print(str(args), file=outFile)
       print(" ".join(map(str,devSurprisalTable)), file=outFile)
    
    #
    #
    return auc
   
calculateTradeoffForWeights(weights_pfx, weights_sfx)

