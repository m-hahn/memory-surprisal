# Based on readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py


# TODO also try other optimizers

import random
import sys

objectiveName = "LM"


import argparse

parser = argparse.ArgumentParser()

# 


parser.add_argument('--language', type=str)
parser.add_argument('--entropy_weight', type=float, default=0.001)
parser.add_argument('--lr_policy', type=float, default=0.0001)
parser.add_argument('--momentum_policy', type=float, default=0.9)
parser.add_argument('--lr_baseline', type=float, default=0.01, dest="lr_baseline")
parser.add_argument('--optimizer_name', type=str, default="SGD")
parser.add_argument('--prescribedID', type=int, default=random.randint(0,10000000000))

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
from random import random, shuffle, randint
import os

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator import CorpusIterator

originalDistanceWeights = {}



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
          line["fine_dep"] = line["dep"]
          depsVocab.add(line["dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          if line["dep"] == "root":
             continue
          posHere = line["posUni"]
          posHead = sentence[line["head"]-1]["posUni"]
          dep = line["dep"]
          direction = "HD" if line["head"] < line["index"] else "DH"
          key = (posHead, dep, posHere)
          keyWithDir = (posHead, dep, posHere, direction)
          orderTable[keyWithDir] = orderTable.get(keyWithDir, 0) + 1
          keys.add(key)
          distanceCounts[key] = distanceCounts.get(key,0.0) + 1.0
          distanceSum[key] = distanceSum.get(key,0.0) + abs(line["index"] - line["head"])
   #print orderTable
   dhLogits = {}
   for key in keys:
      hd = orderTable.get((key[0], key[1], key[2], "HD"), 0) + 1.0
      dh = orderTable.get((key[0], key[1], key[2], "DH"), 0) + 1.0
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
   allGradients = gradients_from_the_left_sum + sum(line.get("children_decisions_logprobs",[]))
   if "linearization_logprobability" in line:
      allGradients += line["linearization_logprobability"] # the linearization of this element relative to its siblings affects everything starting at this word, but nothing to the left of it
   else:
      assert line["dep"] == "root"

   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   result.append(line)
#   print ["DECISIONS_PREPARED", line["index"], line["word"], line["dep"], line["head"], allGradients.data.numpy()[0]]
   line["relevant_logprob_sum"] = allGradients
   if "children_HD" in line:
      for child in line["children_HD"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   return allGradients

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       childrenLinearized = []
       while len(remainingChildren) > 0:
           logits = torch.cat([distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]].view(1) for x in remainingChildren])
           #print logits
           if reverseSoftmax:
              logits = -logits
           #print (reverseSoftmax, logits)
           softmax = softmax_layer(logits.view(1,-1)).view(-1)
           selected = numpy.random.choice(range(0, len(remainingChildren)), p=softmax.data.numpy())
           log_probability = torch.log(softmax[selected])
           assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
           sentence[remainingChildren[selected]-1]["linearization_logprobability"] = log_probability
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
       return childrenLinearized           



def orderSentence(sentence, dhLogits, printThings):
   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   for line in sentence:
      if line["dep"] == "root":
          root = line["index"]
          continue
      if line["dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         continue
      key = (sentence[line["head"]-1]["posUni"], line["dep"], line["posUni"])
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      probability = 1/(1 + torch.exp(-dhLogit))
      dhSampled = (random() < probability.data.numpy())
#      logProbabilityGradient = (1 if dhSampled else -1) * (1-probability)
#      line["ordering_decision_gradient"] = logProbabilityGradient
      line["ordering_decision_log_probability"] = torch.log(1/(1 + torch.exp(- (1 if dhSampled else -1) * dhLogit)))

      
     
      direction = "DH" if dhSampled else "HD"
#torch.exp(line["ordering_decision_log_probability"]).data.numpy()[0],
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("->".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, (str(probability.data.numpy())+"      ")[:8], str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]].data.numpy())+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])
      sentence[headIndex]["children_decisions_logprobs"] = (sentence[headIndex].get("children_decisions_logprobs", []) + [line["ordering_decision_log_probability"]])



   for line in sentence:
      if "children_DH" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
         line["children_DH"] = childrenLinearized
      if "children_HD" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
         line["children_HD"] = childrenLinearized

#         shuffle(line["children_HD"])
   
   linearized = []
   recursivelyLinearize(sentence, root, linearized, Variable(torch.FloatTensor([0.0])))
   if printThings or len(linearized) == 0:
     print " ".join(map(lambda x:x["word"], sentence))
     print " ".join(map(lambda x:x["word"], linearized))
   return linearized, logits


dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()

posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

posFine = list(posFine)
itos_pos_ptb = posFine
stoi_pos_ptb = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = sorted(vocab_deps, key=lambda x:x[1])
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

from torch import optim
if args.optimizer_name == "SGD":
  policy_optimizer = optim.SGD([dhWeights, distanceWeights], lr=args.lr_policy, momentum=args.momentum_policy)
elif args.optimizer_name == "Adagrad":
  policy_optimizer = optim.Adagrad([dhWeights, distanceWeights], lr=args.lr_policy)
elif args.optimizer_name == "Adam":
  policy_optimizer = optim.Adam([dhWeights, distanceWeights], lr=args.lr_policy)
else:
  assert False

words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))

assert stoi[itos[5]] == 5



vocab_size = 50000



batchSize = 1



crossEntropy = 10.0

def encodeWord(w):
   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)



import torch.cuda
import torch.nn.functional




counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 


lm_counts = {}

baseline = [10 for _ in range(len(words)+5)]

def doForwardPass(current):
       global counter
       global crossEntropy
       global printHere
       global devLosses
       batchOrderedLogits = zip(*map(lambda (y,x):orderSentence(x, dhLogits, y==0 and printHere), zip(range(len(current)),current)))
      
       batchOrdered = batchOrderedLogits[0]
       logits = batchOrderedLogits[1]
   
       lengths = map(len, current)
       # current is already sorted by length
       maxLength = lengths[int(0.8*batchSize)]
       input_words = []
       input_pos_u = []
       input_pos_p = []
       for i in range(maxLength+2):
          input_words.append(map(lambda x: 2 if i == 0 else (encodeWord(x[i-1]["word"]) if i <= len(x) else 0), batchOrdered))
          input_pos_u.append(map(lambda x: 2 if i == 0 else (stoi_pos_uni[x[i-1]["posUni"]]+3 if i <= len(x) else 0), batchOrdered))
          input_pos_p.append(map(lambda x: 2 if i == 0 else (stoi_pos_ptb[x[i-1]["posFine"]]+3 if i <= len(x) else 0), batchOrdered))

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()), Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()))
       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0

       policy_optimizer.zero_grad()
#       momentum = 0.0
#       assert momentum == 0.0 # here taking offline estimate, so no momentum
#       for p in  [dhWeights, distanceWeights]:
#          if p.grad is not None:
#             p.grad.data = p.grad.data.mul(momentum)


       totalQuality = 0.0

       if True:



 
           lossesWord = [[None]*batchSize for i in range(maxLength+1)]

           for i in range(1,len(input_words)-1): #range(1,maxLength+1): # don't include i==0
              for j in range(batchSize):
                 if i+1 < len(input_words) and input_words[i][j] != 0:
                    left = input_words[i][j]
                    right = input_words[i+1][j]

                    if left in lm_counts and right in lm_counts:
                        delta = 0.5
                        prob = max(lm_counts[left].get(right,0)-delta, 0) / lm_counts[left]["_TOTAL_"] + ((lm_counts[right]["_TOTAL_"] + 0.0) / lm_counts["_TOTAL_"]) * delta * (len(lm_counts[left]) - 1.0) / lm_counts[left]["_TOTAL_"]
                        assert prob <= 1.0, prob
                        lossesWord[i][j] = - log(prob)

                        loss += lossesWord[i][j] # + lossesPOS[i][j]
                        lossWords += lossesWord[i][j]
                        policyGradientLoss += batchOrdered[j][(i if i < len(batchOrdered[j]) else i-1)]["relevant_logprob_sum"] * (lossesWord[i][j] - baseline[right]) 
                        baseline[right] = 0.95 * baseline[right] + (1-0.95) * lossesWord[i][j]
                    lm_counts["_TOTAL_"] = lm_counts.get("_TOTAL_", 0) + 1
                    if left not in lm_counts:
                        lm_counts[left] = {"_TOTAL_" : 0}
                    lm_counts[left]["_TOTAL_"] =  lm_counts[left]["_TOTAL_"] + 1
                    lm_counts[left][right] =  lm_counts[left].get(right, 0) + 1
                    if input_words[i+1] == 0: # EOS
                        lm_counts[0] = lm_counts.get(0,0) + 1
#                    if j == 0:
 #                     print ["DECISION_PROB",batchOrdered[j][i]["relevant_logprob_sum"].data.numpy()[0] ]
                    if input_words[i+1] > 2 and j == 0 and printHere:
                       print [itos[input_words[i+1][j]-3],  lossesWord[i][j], baseline[right]] # itos_pos_ptb[input_pos_p[i+1][j]-3],
                    wordNum += 1
       if wordNum == 0:
         print input_words
         print batchOrdered
         return 0,0,0,0,0
       if printHere:
         print loss/wordNum
         print lossWords/wordNum
         print ["CROSS ENTROPY", crossEntropy, exp(crossEntropy)]
       crossEntropy = 0.99 * crossEntropy + 0.01 * (lossWords/wordNum) #.data.cpu().numpy()[0]
       totalQuality = loss #.data.cpu().numpy()[0] # consists of lossesWord + lossesPOS
       numberOfWords = wordNum
       probabilities = torch.sigmoid(dhWeights)

       neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))

       policy_related_loss = args.entropy_weight * neg_entropy + policyGradientLoss # lives on CPU
       return loss, baselineLoss, policy_related_loss, totalQuality, numberOfWords


def  doBackwardPass(loss, baselineLoss, policy_related_loss):
       global lastDevLoss
       global failedDevRuns
       if printHere:
         print "BACKWARD 1"
       policy_related_loss.backward()
       if printHere:
         print "BACKWARD 2"



       if printHere:
         print "BACKWARD 3 "+__file__+" "+args.language+" "+str(myID)+" "+str(counter)+" "+str(lastDevLoss)+" "+str(failedDevRuns)+"  "+(" ".join(map(str,["ENTROPY", args.entropy_weight, "LR_POLICY", args.lr_policy, "MOMENTUM", args.momentum_policy ])))
         print devLosses
       torch.nn.utils.clip_grad_norm([dhWeights, distanceWeights], 5.0, norm_type='inf')

       policy_optimizer.step()

def computeDevLoss():
   global printHere
   global counter
   devLoss = 0.0
   devWords = 0
#   corpusDev = getNextSentence("dev")
   corpusDev = CorpusIterator(args.language,"dev").iterator(rejectShortSentences = True)

   while True:
     try:
        batch = map(lambda x:next(corpusDev), 10*range(batchSize))
     except StopIteration:
        break
     batch = sorted(batch, key=len)
     partitions = range(10)
     shuffle(partitions)
     for partition in partitions:
        counter += 1
        printHere = (counter % 5 == 0)
        current = batch[partition*batchSize:(partition+1)*batchSize]
 
        _, _, _, newLoss, newWords = doForwardPass(current)
        devLoss += newLoss
        devWords += newWords
   return devLoss/devWords

while True:
#  corpus = getNextSentence("train")
  corpus = CorpusIterator(args.language).iterator(rejectShortSentences = True)


  while True:
    try:
       batch = map(lambda x:next(corpus), 10*range(batchSize))
    except StopIteration:
       break
    batch = sorted(batch, key=len)
    partitions = range(10)
    shuffle(partitions)
    for partition in partitions:
       counter += 1
       printHere = (counter % 20 == 0)
       current = batch[partition*batchSize:(partition+1)*batchSize]

       loss, baselineLoss, policy_related_loss, _, wordNumInPass = doForwardPass(current)
       if wordNumInPass > 0:
         doBackwardPass(loss, baselineLoss, policy_related_loss)
       else:
         print "No words, skipped backward"

       if counter % 50000 == 0:
          newDevLoss = computeDevLoss()
          devLosses.append(newDevLoss)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
          if lastDevLoss is None or newDevLoss < lastDevLoss:
             lastDevLoss = newDevLoss
             failedDevRuns = 0
          else:
             failedDevRuns += 1
             print "Skip saving, hoping for better model"
             #lr_lm *= 0.5
             continue
          print "Saving"
          TARGET_DIR = "/u/scr/mhahn/deps/locality_optimized_i1/"
          with open(TARGET_DIR+"/"+args.language+"_"+__file__+"_model_"+str(myID)+".tsv", "w") as outFile:
             print >> outFile, "\t".join(map(str,["HeadPOS","DH_Weight","CoarseDependency","DependentPOS","DistanceWeight"]))
             for i in range(len(itos_deps)):
                key = itos_deps[i]
                dhWeight = dhWeights[i].data.numpy()
                distanceWeight = distanceWeights[i].data.numpy()
                head, dependency, dependent = key
                print >> outFile, "\t".join(map(str,[head, dhWeight, dependency, dependent, distanceWeight]))








       if counter >= 1000000:
          print "Stop after "+str(1000000)+" iterations"
          quit()


