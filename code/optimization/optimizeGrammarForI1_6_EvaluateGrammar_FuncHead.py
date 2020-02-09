# Deterministically evaluates grammar.


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
parser.add_argument('--batchSize', type=int, default=1, dest="batchSize")
parser.add_argument('--prescribedID', type=int, default=random.randint(0,10000000000), dest="prescribedID")
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--delta', type=float, default=0.5)
parser.add_argument('--model', type=str)

args = parser.parse_args()







maxNumberOfUpdates = int(sys.argv[20]) if len(sys.argv) > 20 else 20000






myID = args.prescribedID
random.seed(a=myID)

#



posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]

deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp, sqrt
from random import random, shuffle, randint
import os

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator_FuncHead import CorpusIteratorFuncHead as CorpusIterator

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
              selected = numpy.argmax(softmax.data.numpy())
              log_probability = torch.log(softmax[selected])
           assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
           sentence[remainingChildren[selected]-1]["linearization_logprobability"] = log_probability
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
 #      print(["  ", childrenLinearized])
       return childrenLinearized           



def orderSentence(sentence, dhLogits, printThings):
   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0

   for line in sentence:
       line["coarse_dep"] = makeCoarse(line["dep"])
       line["excluded"] = (line["coarse_dep"] == "root" or line["coarse_dep"].startswith("punct"))
   dependencyKeys = torch.LongTensor([0 if line["excluded"] else stoi_deps[line["coarse_dep"]] for line in sentence])
   dhLogitsHere = torch.index_select(dhWeights, 0, dependencyKeys)
   dhProbabilities = torch.nn.functional.sigmoid(dhLogitsHere).data.numpy().tolist()
   for j, line in enumerate(sentence):
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      if line["coarse_dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         continue
      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      probability = dhProbabilities[j] #1/(1 + torch.exp(-dhLogit))
      dhSampled = (probability > 0.5)
      direction = "DH" if dhSampled else "HD"
      line["direction"] = direction
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, (str(probability)+"      ")[:8], str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]].data.numpy())+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])
   dh_sign = torch.FloatTensor([0 if line["excluded"] else (1 if line["direction"] == "DH" else -1) for line in sentence])
   dhLogProb = torch.nn.functional.logsigmoid(dh_sign * dhLogitsHere)

   for line in sentence:
      if "children_DH" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:])
         line["children_DH"] = childrenLinearized
      if "children_HD" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:])[::-1]
         line["children_HD"] = childrenLinearized

   
   linearized = []
   logprob_sum = recursivelyLinearize(sentence, root, linearized, dhLogProb.sum())
   if printThings or len(linearized) == 0:
     print " ".join(map(lambda x:x["word"], sentence))
     print " ".join(map(lambda x:x["word"], linearized))


   # store new dependency links
   moved = [None] * len(sentence)
   for i, x in enumerate(linearized):
      moved[x["index"]-1] = i
   for i,x in enumerate(linearized):
      if x["head"] == 0: # root
         x["reordered_head"] = 0
      else:
         x["reordered_head"] = 1+moved[x["head"]-1]
   return linearized, logits, logprob_sum


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
   dhLogits[key] = 0.1*(random()-0.5)
   dhWeights.data[i] = dhLogits[key]

   originalDistanceWeights[key] = 0.0 #0.1*(random()-0.5)
   distanceWeights.data[i] = originalDistanceWeights[key]


if args.model == "RANDOM_MODEL":
  for key in range(len(itos_deps)):
     dhWeights[key] = random() - 0.5
     distanceWeights[key] = random()
  originalCounter = "NA"
elif args.model == "RANDOM_BY_TYPE":
  dhByType = {}
  distByType = {}
  for dep in itos_pure_deps:
    dhByType[dep.split(":")[0]] = random() - 0.5
    distByType[dep.split(":")[0]] = random()
  for key in range(len(itos_deps)):
     dhWeights[key] = dhByType[itos_deps[key].split(":")[0]]
     distanceWeights[key] = distByType[itos_deps[key].split(":")[0]]
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
#         if line[headerGrammar.index("Language")] == language:
#           print(line)
         dependency = line[headerGrammar.index("Dependency")]
         dhHere = float(line[headerGrammar.index("DH_Mean_NoPunct")])
         distHere = float(line[headerGrammar.index("Distance_Mean_NoPunct")])
  #       if dhHere < 0:
   #        distHere = -distHere
         print(dependency, dhHere, distHere)
         dhByDependency[dependency] = dhHere
         distByDependency[dependency] = distHere
  for key in range(len(itos_deps)):
     dhWeights[key] = dhByDependency[itos_deps[key].split(":")[0]]
     distanceWeights[key] = distByDependency[itos_deps[key].split(":")[0]]
  originalCounter = "NA"
else:
#  with open("/u/scr/mhahn/deps/locality_optimized_i1/Chinese_optimizeGrammarForI1_3.py_model_675523898.tsv", "r") as inFile:
  with open("/u/scr/mhahn/deps/locality_optimized_i1/"+args.model, "r") as inFile:

     headerGrammar = next(inFile).strip().split("\t")
     print(headerGrammar)
     dhByDependency = {}
     distByDependency = {}
     for line in inFile:
         line = line.strip().split("\t")
         dependency = line[headerGrammar.index("CoarseDependency")]
         dhHere = float(line[headerGrammar.index("DH_Weight")])
         distHere = float(line[headerGrammar.index("DistanceWeight")])
         print(dependency, dhHere, distHere)
         dhByDependency[dependency] = dhHere
         distByDependency[dependency] = distHere
  for key in range(len(itos_deps)):
     dhWeights[key] = dhByDependency[itos_deps[key].split(":")[0]]
     distanceWeights[key] = distByDependency[itos_deps[key].split(":")[0]]
  originalCounter = "NA"



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

def doForwardPass(input_indices, wordStartIndices, surprisalTable=None, doDropout=True, batchSizeHere=1, relevant_logprob_sum=None):
#       print("forward")
       global counter
       global crossEntropy
       global printHere
       global devLosses
       global baselineAverageLoss

       assert doDropout == (relevant_logprob_sum is not None)
       if doDropout:
          for c in components:
             c.zero_grad()
          for p in  [dhWeights, distanceWeights]:
             if p.grad is not None:
                p.grad.data = p.grad.data.mul(args.momentum_policy)

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

       sequenceLength = (map(len, input_indices))
       wordNum = sum(sequenceLength) - len(input_indices) # subtract 1 per batch
       sequenceLength = max(sequenceLength)
#       print(input_indices)

       global totalUnigramCount
       lossesWord = torch.zeros(sequenceLength-1, batchSizeHere)
       mask = torch.zeros(sequenceLength-1, batchSizeHere)
       for i in range(batchSizeHere):
          for j in range(1,len(input_indices[i])):
               source = input_indices[i][j-1]
               countsHere = counts[source]

               target = input_indices[i][j]


               countsHere = counts[source]

               targetCount = countsHere[0].get(target, 0)
               sumOfCounts = countsHere[1][0]


               probKneserNey = (max(targetCount-args.delta, 0.0) + len(countsHere[0]) * (args.delta * unigramCounts[target])/totalUnigramCount)/sumOfCounts
               assert 0.0 < probKneserNey, probKneserNey
               assert probKneserNey < 1.0, probKneserNey

               # TRAINING
               if doDropout and target not in countsHere[0]:
                 countsHere[0][target] = 0

               if doDropout:
                 countsHere[0][target] += 1
                 countsHere[1][0] += 1

               if doDropout:
                 unigramCounts[target] += 1
                 totalUnigramCount += 1

               lossesWord[j-1][i] = -math.log(probKneserNey)
               mask[j-1][i] = 1

       for i in range(batchSizeHere):
          input_indices[i] = input_indices[i][:]
          while len(input_indices[i]) < sequenceLength:
             input_indices[i].append(2)

       inputTensor = Variable(torch.LongTensor(input_indices).transpose(0,1).contiguous()) # so it will be sequence_length x batchSizeHere
       inputTensorIn = inputTensor[:-1].view(1, -1)
       inputTensorOut = inputTensor[1:].view(1, -1)

       loss = lossesWord.sum()


          
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
       totalQuality = loss.data.cpu().numpy() # consists of lossesWord + lossesPOS
       numberOfWords = wordNum
#       print(doDropout, float(loss/wordNum))

       return (loss/batchSizeHere, baselineLoss/batchSizeHere, None, totalQuality, numberOfWords)





def computeDevLoss():
   devBatchSize = args.batchSize
   global printHere
   devLoss = 0.0
   devWords = 0
   corpusDev = CorpusIterator(args.language,"dev").iterator(rejectShortSentences = False)
   stream = createStream(corpusDev, training=False)

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
       ordered, _, relevant_logprob_sum = orderSentence(sentence, dhLogits, printHere)
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

       yield input_indices, wordStartIndices+[len(input_indices)], relevant_logprob_sum
       input_indices = [2] # Start of Segment (makes sure that first word can be predicted from this token)
       wordStartIndices = []





DEV_PERIOD = 5000
epochCount = 0
corpusBase = CorpusIterator(args.language, storeMorph=True)
if True:
  epochCount += 1
  print >> sys.stderr, "Epoch "+str(epochCount)
  print "Starting new epoch, permuting corpus"
  corpusBase.permute()
#  corpus = getNextSentence("train")
  corpus = corpusBase.iterator(rejectShortSentences = False)
  stream = createStream(corpus)



  lasttime = time.time()
  while True:
       counter += 1
       printHere = (counter % 100 == 0)

       devBatchSize = args.batchSize
       try:
          input_indices_list = []
          wordStartIndices_list = []
          relevant_logprob_sum = []
          for _ in range(args.batchSize):
             input_indices, wordStartIndices, logProbSum = next(stream)
             input_indices_list.append(input_indices)
             wordStartIndices_list.append(wordStartIndices)
             relevant_logprob_sum.append(logProbSum)
       except StopIteration:
          devBatchSize = len(input_indices_list)
       if devBatchSize == 0:
         break
       loss, baselineLoss, _, _, wordNumInPass = doForwardPass(input_indices_list, wordStartIndices_list, batchSizeHere=devBatchSize, relevant_logprob_sum=relevant_logprob_sum)
       if printHere:
          print "Epoch "+str(epochCount)+" "+str(counter)



newDevLoss, _ = computeDevLoss()
devLosses.append(newDevLoss)


print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)

with open("/u/scr/mhahn/deps/locality_optimized_i1/report_"+__file__+"_"+args.model, "w") as outFile:
   print >> outFile, newDevLoss
