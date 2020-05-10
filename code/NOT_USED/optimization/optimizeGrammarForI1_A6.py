# Uses Kneser-Ney bigram model. Doing some experimenting also.



# optimizeGrammarForI1.py
# yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_FullVocab.py
# readDataDistCrossGPUFreeAllTwoEqual_NoClip_ByCoarseOnly_FixObj_OnlyLangmod_Replication_Best.py


#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"
averageTime = 1.1
lasttime = 0
import time

# TODO also try other optimizers

import random
import sys
import math

objectiveName = "LM"


import argparse

parser = argparse.ArgumentParser()

# 


parser.add_argument('--language', type=str, dest="language")
parser.add_argument('--entropy_weight', type=float, default=0.001, dest="entropy_weight")
parser.add_argument('--lr_policy', type=float, default=0.0001, dest="lr_policy")
parser.add_argument('--momentum_policy', type=float, default=0.9, dest="momentum_policy")
parser.add_argument('--lr_baseline', type=float, default=0.01, dest="lr_baseline")
parser.add_argument('--batchSize', type=int, default=1, dest="batchSize")
parser.add_argument('--prescribedID', type=int, default=random.randint(0,10000000000), dest="prescribedID")
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--delta', type=float, default=0.5)

args = parser.parse_args()






assert args.lr_policy < 1.0
assert args.momentum_policy < 1.0
assert args.entropy_weight >= 0

maxNumberOfUpdates = int(sys.argv[20]) if len(sys.argv) > 20 else 20000

model = "REINFORCE"





myID = args.prescribedID
random.seed(a=myID)

#



posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]

deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp, sqrt
from random import shuffle, randint
import os

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator import CorpusIterator

originalDistanceWeights = {}


def makeCoarse(x):
   if ":" in x:
      return x[:x.index(":")]
   return x

def initializeOrderTable():
   orderTable = {}
   keys = set()
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in CorpusIterator(args.language,partition).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          line["coarse_dep"] = makeCoarse(line["dep"])
          depsVocab.add(line["coarse_dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          if line["coarse_dep"] == "root":
             continue
          posHere = line["posUni"]
          posHead = sentence[line["head"]-1]["posUni"]
          dep = line["coarse_dep"]
          direction = "HD" if line["head"] < line["index"] else "DH"
          key = dep
          keyWithDir = (dep, direction)
          orderTable[keyWithDir] = orderTable.get(keyWithDir, 0) + 1
          keys.add(key)
          distanceCounts[key] = distanceCounts.get(key,0.0) + 1.0
          distanceSum[key] = distanceSum.get(key,0.0) + abs(line["index"] - line["head"])
   #print orderTable
   dhLogits = {}
   for key in keys:
      hd = orderTable.get((key, "HD"), 0) + 1.0
      dh = orderTable.get((key, "DH"), 0) + 1.0
      dhLogit = log(dh) - log(hd)
      dhLogits[key] = dhLogit
      originalDistanceWeights[key] = (distanceSum[key] / distanceCounts[key])
   return dhLogits, vocab, keys, depsVocab

#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable

torch.manual_seed(myID)

# "linearization_logprobability"
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   # Loop Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum #+ sum(line.get("children_decisions_logprobs",[]))

   if "linearization_logprobability" in line:
      allGradients += line["linearization_logprobability"] # the linearization of this element relative to its siblings affects everything starting at the start of the constituent, but nothing to the left of it
   else:
      assert line["coarse_dep"] == "root"


   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   result.append(line)
#   print ["DECISIONS_PREPARED", line["index"], line["word"], line["dep"], line["head"], allGradients.data.numpy()]
   line["relevant_logprob_sum"] = allGradients
   if "children_HD" in line:
      for child in line["children_HD"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   return allGradients

import numpy.random
import numpy as np

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()
logsoftmaxLabels =  torch.nn.LogSoftmax(dim=2)



def orderChildrenRelative(sentence, remainingChildren):
       childrenLinearized = []
       # quick optimization
#       if len(remainingChildren) == 1:
#           del remainingChildren[0]          
#           childrenLinearized.append(0)
#           return childrenLinearized
#       print(remainingChildren)
       while len(remainingChildren) > 0:
           if len(remainingChildren) == 1: # quick optimization
               selected = 0
               log_probability = 0
           else:
              relationIndices = torch.LongTensor([stoi_deps[sentence[x-1]["dependency_key"]] for x in remainingChildren])
   #           print(relationIndices)
              logits = torch.index_select(distanceWeights, 0, relationIndices)
#              logits = torch.cat([distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]].view(1) for x in remainingChildren])
              softmax = softmax_layer(logits.view(1,-1)).view(-1)
#              print(softmax)
#              print(softmax)
              selected = numpy.random.choice(range(0, len(remainingChildren)), p=softmax.data.numpy())
              log_probability = torch.log(softmax[selected])
           assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
           sentence[remainingChildren[selected]-1]["linearization_logprobability"] = log_probability
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
 #      print(["  ", childrenLinearized])
       return childrenLinearized           

import hashlib

def orderSentence(sentence, dhLogits, printThings):
   for j, line in enumerate(sentence):
      if line["dep"] == "root":
          root = line
          continue
      if line["dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         continue
      headIndex = line["head"]-1
      sentence[headIndex]["children"] = (sentence[headIndex].get("children", []) + [line])
   sentence= [x for x in sentence if not x["dep"].startswith("punct")]
   sentHash = hashlib.sha224(" ".join([x["word"] for x in sentence])).hexdigest()

   for i, line in enumerate(sentence):
         line["new_index"] = i

   return sentence, root, sentHash

dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()

posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

posFine = list(posFine)
itos_pos_ptb = posFine
stoi_pos_ptb = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

print itos_deps




import os


dhWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
distanceWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
for i, key in enumerate(itos_deps):
   dhLogits[key] = 0.1*(random.random()-0.5)
   dhWeights.data[i] = dhLogits[key]

   originalDistanceWeights[key] = 0.0 #0.1*(random()-0.5)
   distanceWeights.data[i] = originalDistanceWeights[key]



words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))

if len(itos) > 6:
   assert stoi[itos[5]] == 5



vocab_size = 50000
vocab_size = min(len(itos),vocab_size)
import sys
print >> sys.stderr, ("VOCAB_SIZE", vocab_size)




baseline = torch.nn.Embedding(num_embeddings = len(itos_pos_uni)+vocab_size+3, embedding_dim=1, padding_idx=2)
print posUni
#print posFine
print "VOCABULARY "+str(vocab_size+3)
outVocabSize = len(itos_pos_uni)+vocab_size+3
#quit()

# counts per target
# current state of the denominator sum WITHOUT the decay terms
# the sum of all the decay terms
counts = [({-1 : 1}, [1]) for _ in range(outVocabSize)] # args.epsilon*outVocabSize

unigramCounts = [args.epsilon for _ in range(outVocabSize)] + [args.epsilon]
totalUnigramCount = args.epsilon*(outVocabSize+1)


itos_total = ["EOS", "EOW", "SOS"] + itos_pos_uni + itos[:vocab_size]
assert len(itos_total) == outVocabSize



from torch import optim




#################################################################################3
# Language Model
#################################################################################3





components = [baseline]
# word_embeddings, pos_u_embeddings, pos_p_embeddings, 
#baseline, 

def parameters():
 for c in components:
   for param in c.parameters():
      yield param
# yield dhWeights
# yield distanceWeights

#for pa in parameters():
#  print pa

initrange = 0.1

baseline.weight.data.fill_(math.log(vocab_size)) #uniform_(-initrange, initrange)
#baseline.weight[2].fill_(0.0)



#################################################################################3
# Policy
#################################################################################3


def parameters_policy():
 yield dhWeights
 yield distanceWeights


parameters_cached = [x for x in parameters()]

parameters_policy_cached = [x for x in parameters_policy()]

#################################################################################3
#################################################################################3
#################################################################################3


crossEntropy = 10.0




import torch.nn.functional



counter = 0


lastDevLoss = None

failedDevRuns = 0

devLosses = [] 

lossModuleTest = nn.CrossEntropyLoss(size_average=False, reduce=False, reduction="none", ignore_index=2)


baselineAverageLoss = 1.0


def knProb(source, target):
      #unigramLogProb = math.log(unigramCounts[target]) - math.log(totalUnigramCount)

      countsHere = counts[source]

      targetCount = countsHere[0].get(target, 0)
      sumOfCounts = countsHere[1][0]

      #log_probability = math.log(targetCount) - math.log(sumOfCounts)

      #print(log_probability, unigramLogProb)

      # increment logit

      probKneserNey = (max(targetCount-args.delta, 0.0) + len(countsHere[0]) * (args.delta * unigramCounts[target])/totalUnigramCount)/sumOfCounts
#      if probKneserNey > 1 or probKneserNey <= 0:
 #         print("targetCount", targetCount)
  #        print(len(countsHere[0]), sumOfCounts, counts[source])
   #       print(unigramCounts[target], totalUnigramCount)
      assert 0.0 < probKneserNey, probKneserNey
      assert probKneserNey < 1.0, probKneserNey
      return probKneserNey


def numerify(line):
     if line == "EOS":
       return 0
     else:
       targetWord = stoi[line["word"]]
       if targetWord >= vocab_size:
          return stoi_pos_uni[line["posUni"]]+3
       else:
          return targetWord+3+len(itos_pos_uni)



orders = {}


def recursivelyLinearizeRandomly(node):
    children = [node]
    if "children" in node:
       children += node["children"]
    linearized = []
    random.shuffle(children)
    for child in children:
       if child == node:
           linearized += [node]
       else:
           linearized += recursivelyLinearizeRandomly(child)
    return linearized



def decrementKN(source, target):
      global totalUnigramCount

      countsHere = counts[source]


      countsHere[0][target] -= 1
      countsHere[1][0] -= 1
      if countsHere[0][target] == 0:
          del countsHere[0][target]

      unigramCounts[target] -= 1
      totalUnigramCount -= 1




def incrementKN(source, target):
      global totalUnigramCount

      countsHere = counts[source]

      if target not in countsHere[0]:
        countsHere[0][target] = 0

      countsHere[0][target] += 1
      countsHere[1][0] += 1

      unigramCounts[target] += 1
      totalUnigramCount += 1


def createRandomLinearization(sentence, root, sentHash):
      linearization = [None for _ in range(len(sentence))]
      linearized = recursivelyLinearizeRandomly(root)
      orders[sentHash] = [x["new_index"] for x in linearized]


      input_indices = [numerify(x) for x in ["EOS"]+linearized+["EOS"]]

      sequenceLength = len(input_indices)
      wordNum = sequenceLength-1

  #    lossesWord = torch.zeros(sequenceLength-1)

      for j in range(1,len(input_indices)):
              source = input_indices[j-1]
              target = input_indices[j]
              incrementKN(source, target)
  #            probKneserNey = knProb(source, target)
 #             lossesWord[j-1] = -math.log(probKneserNey)
#              print(probKneserNey, lossesWord[j-1])



def findOrderingsDP(sentence, node):
   orderingsPerChildren = [(node["numerified"], node["numerified"], [node["new_index"]])]
   for child in node.get("children", []):
      orderingsPerChildren.append(findOrderingsDP(sentence, child))
   newOrderings = []
   createAllOrderings TODO
   print(orderingsPerChildren)
   return orderingsPerChildren


def doForwardPass(sentence, root, sentHash, doDropout=True, batchSizeHere=1, relevant_logprob_sum=None):
#       print("forward")
       global counter
       global crossEntropy
       global printHere
       global devLosses
       global baselineAverageLoss


       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()), Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()))
       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0

       #############################################################################
       # Language Model 
       #############################################################################


       totalQuality = 0.0


       ordered = orders[sentHash] #= [x["new_index"] for x in linearized]
       previous = [sentence[x] for x in ordered]


       input_indices = [numerify(x) for x in ["EOS"]+previous+["EOS"]]

       sequenceLength = len(input_indices)
       wordNum = sequenceLength-1

       global totalUnigramCount
       lossesWord = torch.zeros(sequenceLength-1)



       for j in range(1,len(input_indices)):
               source = input_indices[j-1]
               target = input_indices[j]
               print(knProb(source, target))
               decrementKN(source, target)
               incrementKN(source, target)
               print(knProb(source, target))

       for x in sentence:
          x["numerified"] = numerify(x)

#       input_indices = [numerify(x) for x in sentence]
             
       orderings = findOrderingsDP(sentence, root)


  
       if wordNum == 0:
         print input_words
         print batchOrdered
         return 0,0,0,0,0
       if printHere:
         print loss/wordNum
         print lossWords/wordNum
         print ["CROSS ENTROPY LM", crossEntropy, exp(crossEntropy)]
         print baselineAverageLoss
       crossEntropy = 0.99 * crossEntropy + 0.01 * float(loss/wordNum)
       numberOfWords = sequenceLength
#       print(doDropout, float(loss/wordNum))

       return (loss/batchSizeHere, _, _, _, numberOfWords)



def computeDevLoss():
   devBatchSize = args.batchSize
   global printHere
   devLoss = 0.0
   devWords = 0
   corpusDev = CorpusIterator(args.language,"dev").iterator(rejectShortSentences = False)

   surprisalTable = [0 for _ in range(2)]
   devCounter = 0
   devCounterTimesBatchSize = 0
   while True:
     try:
        input_indices_list = []
        wordStartIndices_list = []
        for _ in range(devBatchSize):
           input_indices, wordStartIndices, _ = next(stream)
           input_indices_list.append(input_indices)
           wordStartIndices_list.append(wordStartIndices)
     except StopIteration:
        devBatchSize = len(input_indices_list)
     if devBatchSize == 0:
       break
     devCounter += 1
     printHere = (devCounter % 100 == 0)
     _, _, _, newLoss, newWords = doForwardPass(input_indices_list, wordStartIndices_list, surprisalTable = surprisalTable, doDropout=False, batchSizeHere=devBatchSize,relevant_logprob_sum=None )
     devLoss += newLoss
     devWords += newWords
     if printHere:
         print "Dev examples "+str(devCounter)
     devCounterTimesBatchSize += devBatchSize
   return devLoss/devWords, None #devSurprisalTableHere



def createStream(corpus, training=True):
    global crossEntropy
    global printHere
    global devLosses
    input_indices = [2] # Start of Segment
    wordStartIndices = []
    sentCount = 0
    for sentence in corpus:
       #if len(sentence) > 5:
       #   continue
       sentCount += 1
 #      if sentCount % 100 == 0:
#          print("\t".join([str(sentCount), "Sentences"]))
       ordered, root = orderSentence(sentence, dhLogits, printHere)
       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
            targetWord = stoi[line["word"]]
            if targetWord >= vocab_size:
               input_indices.append(stoi_pos_uni[line["posUni"]]+3)
            else:
               input_indices.append(targetWord+3+len(itos_pos_uni))

       yield input_indices, ordered, root
       input_indices = [2] # Start of Segment (makes sure that first word can be predicted from this token)
       wordStartIndices = []





DEV_PERIOD = 5000
epochCount = 0
corpusBase = CorpusIterator(args.language, storeMorph=True)


corpus = corpusBase.iterator(rejectShortSentences = False)

for sentence in corpus:
     printHere = (counter % 100 == 0)

     counter += 1
     if counter % 100 == 0:
        print(counter)
     sentence, root, sentHash = orderSentence(sentence, dhLogits, printHere)

     createRandomLinearization(sentence, root, sentHash)


while failedDevRuns < 5:
  epochCount += 1
  print >> sys.stderr, "Epoch "+str(epochCount)
  print "Starting new epoch, permuting corpus"
  corpusBase.permute()
#  corpus = getNextSentence("train")


  if counter > 5 and False:
          newDevLoss, _ = computeDevLoss()
          devLosses.append(newDevLoss)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
          if newDevLoss > 20 or len(devLosses) > 99:
              print "Abort, training too slow?"
              devLosses.append(newDevLoss+0.001)

#          if lastDevLoss is None or newDevLoss < lastDevLoss:
              #devSurprisalTable = devSurprisalTableHere
          DOING_PARAMETER_SEARCH = True
          TARGET_DIR = "/u/scr/mhahn/deps/locality_optimized_i1/"
          with open(TARGET_DIR+"/estimates-"+args.language+"_"+__file__+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
              print >> outFile, " ".join(sys.argv)
              print >> outFile, " ".join(map(str,devLosses))
#              print >> outFile, " ".join(map(str,devSurprisalTable))
              print >> outFile, "PARAMETER_SEARCH" if DOING_PARAMETER_SEARCH else "RUNNING"


          q= float(dhWeights[8])
          assert q == q


          print "Saving"
          if False:
            with open(TARGET_DIR+"/"+args.language+"_"+__file__+"_model_"+str(myID)+".tsv", "w") as outFile:
               print >> outFile, "\t".join(map(str,["FileName","DH_Weight","CoarseDependency","DistanceWeight"]))
               for i in range(len(itos_deps)):
                  key = itos_deps[i]
                  dhWeight = dhWeights[i].data.numpy()
                  distanceWeight = distanceWeights[i].data.numpy()
                  dependency = key
                  print >> outFile, "\t".join(map(str,[myID, dhWeight, dependency, distanceWeight]))




          if newDevLoss > 20 or len(devLosses) > 100:
              print "Abort, training too slow?"
              failedDevRuns = 1
              break


          if lastDevLoss is None or newDevLoss < lastDevLoss:
             lastDevLoss = newDevLoss
             failedDevRuns = 0
          else:
             failedDevRuns += 1
             print "Skip saving, hoping for better model"
             print devLosses
             print "Epoch "+str(epochCount)+" "+str(counter)
#             print zip(range(1,horizon+1), devSurprisalTable)
#             print devSurprisalTable


             #break

  if failedDevRuns >= 5:
     break
  lasttime = time.time()
  corpus = corpusBase.iterator(rejectShortSentences = False)
 
  for sentence in corpus:
       printHere = (counter % 100 == 0)

       counter += 1

       ordered, root, sentHash = orderSentence(sentence, dhLogits, printHere)

       loss, baselineLoss, policy_related_loss, _, wordNumInPass = doForwardPass(ordered, root, sentHash)
       if printHere:
          print "Epoch "+str(epochCount)+" "+str(counter)


