#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"


# ./python27 yHyperParamSearchGPUs_CorPost_Automated_OnlyWordForms_Slurm_Ngrams.py Breton-Adap 1 3 NONE RANDOM_BY_TYPE 0.02 100

#
#.35, 15, 0.5, 3], 6.2189246648251935, 6.521915246038937), (6.342939013534792, [6.3037920149], [0.9, 30, 0.2, 9], 6.138116416282256, 6.547761610787329), (6.33609884027768, [6.4674
#8277347], [0.2, 2, 5.0, 4], 6.114406374589717, 6.557791305965643), (6.4281052296959, [6.33108654996], [0.15, 20, 1.0, 2], 6.224660603401713, 6.631549855990087), (6.522439208201903, [6.54414465115], [0.35,
# 4, 0.1, 7], 6.352480445062022, 6.6923979713417845), (6.8038328589974935, [6.85282802645], [0.15, 10, 0.1, 8], 6.622163716863473, 6.985502001131514), (6.846745014742145, [6.97154534213], [0.1, 5, 0.2, 10]
#, 6.640600072694959, 7.05288995678933)]
#/u/scr/mhahn/deps/memory-need-ngrams/search-Breton-Adap_yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py_model_847439581_RANDOM_BY_TYPE.txt
#Traceback (most recent call last):
#  File "yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py", line 613, in <module>
#    probability = log(max(countNgram - args.alpha, 0.0) + args.alpha * followingCount * exp(lastProbability[j])) -  log(countPrefix)
#ValueError: math domain error
#OBTAINED RESULT
#


# TODO also try other optimizers

import random
import sys

objectiveName = "LM"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--model", dest="model", type=str)
parser.add_argument("--alpha", dest="alpha", type=float, default=1.0)
parser.add_argument("--gamma", dest="gamma", type=int, default=1)
parser.add_argument("--delta", dest="delta", type=float, default=1.0)
parser.add_argument("--cutoff", dest="cutoff", type=int, default=10)
parser.add_argument("--idForProcess", dest="idForProcess", type=int, default=random.randint(0,10000000))
import random



args=parser.parse_args()
print(args)

#assert args.cutoff == 20

assert args.alpha >= 0
assert args.alpha <= 1
assert args.delta >= 0
assert args.gamma >= 1





myID = args.idForProcess


TARGET_DIR = "/u/scr/mhahn/deps/memory-need-ngrams/"



posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]



deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp
from random import random, shuffle, randint

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator import CorpusIterator

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}

def initializeOrderTable():
   orderTable = {}
   keys = set()
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in CorpusIterator(args.language,partition, storeMorph=True).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          vocab_lemmas[line["lemma"]] = vocab_lemmas.get(line["lemma"], 0) + 1

          depsVocab.add(line["dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          for morph in line["morph"]:
              morphKeyValuePairs.add(morph)
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


# "linearization_logprobability"
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   # Loop Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum #+ sum(line.get("children_decisions_logprobs",[]))

#   if "linearization_logprobability" in line:
#      allGradients += line["linearization_logprobability"] # the linearization of this element relative to its siblings affects everything starting at the start of the constituent, but nothing to the left of it
#   else:
#      assert line["dep"] == "root"


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
#       childrenLinearized = []
#       while len(remainingChildren) > 0:
       if args.model == "REAL":
          return remainingChildren
       logits = [(x, distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]]) for x in remainingChildren]
       logits = sorted(logits, key=lambda x:x[1], reverse=(not reverseSoftmax))
       childrenLinearized = map(lambda x:x[0], logits)
           
#           #print logits
#           if reverseSoftmax:
#              
#              logits = -logits
#           #print (reverseSoftmax, logits)
#           softmax = softmax_layer(logits.view(1,-1)).view(-1)
#           selected = numpy.random.choice(range(0, len(remainingChildren)), p=softmax.data.numpy())
#    #       log_probability = torch.log(softmax[selected])
#   #        assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
#  #         sentence[remainingChildren[selected]-1]["linearization_logprobability"] = log_probability
#           childrenLinearized.append(remainingChildren[selected])
#           del remainingChildren[selected]
       return childrenLinearized           
#           softmax = torch.distributions.Categorical(logits=logits)
#           selected = softmax.sample()
#           print selected
#           quit()
#           softmax = torch.cat(logits)



def orderSentence(sentence, dhLogits, printThings):

   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   if args.model == "REAL_REAL":
      eliminated = []
   for line in sentence:
      if line["dep"] == "root":
          root = line["index"]
          continue
      if line["dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         if args.model == "REAL_REAL":
            eliminated.append(line)
         continue
      key = (sentence[line["head"]-1]["posUni"], line["dep"], line["posUni"])
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
#      probability = 1/(1 + torch.exp(-dhLogit))
      if args.model == "REAL":
         dhSampled = (line["head"] > line["index"]) #(random() < probability.data.numpy()[0])
      else:
         dhSampled = (dhLogit > 0) #(random() < probability.data.numpy())
#      logProbabilityGradient = (1 if dhSampled else -1) * (1-probability)
#      line["ordering_decision_gradient"] = logProbabilityGradient
      #line["ordering_decision_log_probability"] = torch.log(1/(1 + torch.exp(- (1 if dhSampled else -1) * dhLogit)))

      
     
      direction = "DH" if dhSampled else "HD"
#torch.exp(line["ordering_decision_log_probability"]).data.numpy()[0],
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], ("|".join(line["morph"])+"           ")[:10], ("->".join(list(key)) + "         ")[:22], line["head"], dhLogit, dhSampled, direction]))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])
      #sentence[headIndex]["children_decisions_logprobs"] = (sentence[headIndex].get("children_decisions_logprobs", []) + [line["ordering_decision_log_probability"]])


   if args.model != "REAL_REAL":
      for line in sentence:
         if "children_DH" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
            line["children_DH"] = childrenLinearized
         if "children_HD" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
            line["children_HD"] = childrenLinearized
   if args.model == "REAL_REAL":
       while len(eliminated) > 0:
          line = eliminated[0]
          del eliminated[0]
          if "removed" in line:
             continue
          line["removed"] = True
          if "children_DH" in line:
            assert 0 not in line["children_DH"]
            eliminated = eliminated + [sentence[x-1] for x in line["children_DH"]]
          if "children_HD" in line:
            assert 0 not in line["children_HD"]
            eliminated = eliminated + [sentence[x-1] for x in line["children_HD"]]

#         shuffle(line["children_HD"])
   
   linearized = []
   recursivelyLinearize(sentence, root, linearized, 0)
   if args.model == "REAL_REAL":
      linearized = filter(lambda x:"removed" not in x, sentence)
   if printThings or len(linearized) == 0:
     print " ".join(map(lambda x:x["word"], sentence))
     print " ".join(map(lambda x:x["word"], linearized))
   return linearized, logits


dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()
#print morphKeyValuePairs
#quit()

morphKeyValuePairs = list(morphKeyValuePairs)
itos_morph = morphKeyValuePairs
stoi_morph = dict(zip(itos_morph, range(len(itos_morph))))


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

#print itos_deps

dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)

#dhWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
#distanceWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
#for i, key in enumerate(itos_deps):
#
#   # take from treebank, or randomize
#   dhLogits[key] = 2*(random()-0.5)
#   dhWeights.data[i] = dhLogits[key]
#
#   originalDistanceWeights[key] = random()  
#   distanceWeights.data[i] = originalDistanceWeights[key]

import os

#if model != "RANDOM_MODEL" and args.model != "REAL" and args.model != "RANDOM_BY_TYPE":
#   inpModels_path = "/juicier/scr120/scr/mhahn/deps/"+"/manual_output/"
#   args.models = os.listdir(inpModels_path)
#   args.models = filter(lambda x:"_"+args.model+".tsv" in x, args.models)
#   if len(args.models) == 0:
#     assert False, "No args.model exists"
#   if len(args.models) > 1:
#     assert False, [args.models, "Multiple args.models exist"]
#   
#   with open(inpModels_path+args.models[0], "r") as inFile:
#      data = map(lambda x:x.split("\t"), inFile.read().strip().split("\n"))
#      header = data[0]
#      data = data[1:]
#    
#   for line in data:
#      head = line[header.index("Head")]
#      dependent = line[header.index("Dependent")]
#      dependency = line[header.index("Dependency")]
#      key = (head, dependency, dependent)
#      dhWeights[stoi_deps[key]] = float(line[header.index("DH_Weight")].replace("[", "").replace("]",""))
#      distanceWeights[stoi_deps[key]] = float(line[header.index("DistanceWeight")].replace("[", "").replace("]",""))
#      originalCounter = int(line[header.index("Counter")])
if args.model == "RANDOM_MODEL":
  for key in range(len(itos_deps)):
     dhWeights[key] = random() - 0.5
     distanceWeights[key] = random()
  originalCounter = "NA"
elif args.model == "REAL" or args.model == "REAL_REAL":
  originalCounter = "NA"
elif args.model == "RANDOM_BY_TYPE":
  dhByType = {}
  distByType = {}
  for dep in itos_pure_deps:
    dhByType[dep.split(":")[0]] = random() - 0.5
    distByType[dep.split(":")[0]] = random()
  for key in range(len(itos_deps)):
     dhWeights[key] = dhByType[itos_deps[key][1].split(":")[0]]
     distanceWeights[key] = distByType[itos_deps[key][1].split(":")[0]]
  originalCounter = "NA"
elif args.model == "RANDOM_BY_TYPE_CONS":
  distByType = {}
  for dep in itos_pure_deps:
    distByType[dep.split(":")[0]] = random()
  for key in range(len(itos_deps)):
     dhWeights[key] = 1.0
     distanceWeights[key] = distByType[itos_deps[key][1].split(":")[0]]
  originalCounter = "NA"
elif args.model == "RANDOM_MODEL_CONS":
  for key in range(len(itos_deps)):
     dhWeights[key] = 1.0
     distanceWeights[key] = random()
  originalCounter = "NA"


lemmas = list(vocab_lemmas.iteritems())
lemmas = sorted(lemmas, key = lambda x:x[1], reverse=True)
itos_lemmas = map(lambda x:x[0], lemmas)
stoi_lemmas = dict(zip(itos_lemmas, range(len(itos_lemmas))))

words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))
#print stoi
#print itos[5]
#print stoi[itos[5]]

if len(itos) > 6:
   assert stoi[itos[5]] == 5

#print dhLogits

#for sentence in getNextSentence():
#   print orderSentence(sentence, dhLogits)

vocab_size = len(itos)
#vocab_size = min(len(itos),vocab_size)
#print itos[:vocab_size]
#quit()

# 0 EOS, 1 UNK, 2 BOS
#word_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = emb_dim).cuda()
#pos_u_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = 10).cuda()
#pos_p_embeddings = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=10).cuda()
#morph_embeddings = torch.nn.Embedding(num_embeddings = len(morphKeyValuePairs)+3, embedding_dim=100).cuda()


#
#torch.cuda.set_device(gpuNumber)
#
#word_pos_morph_embeddings = torch.nn.Embedding(num_embeddings = len(itos_pos_uni)+vocab_size+3, embedding_dim=emb_dim).cuda()
#print posUni
##print posFine
#print "VOCABULARY "+str(vocab_size+3)
#outVocabSize = len(itos_pos_uni)+vocab_size+3
##quit()
#
#
#itos_total = ["EOS", "EOW", "SOS"] + itos_pos_uni + itos[:vocab_size]
#assert len(itos_total) == outVocabSize
## could also provide per-word subcategorization frames from the treebank as input???
#
#
##baseline = nn.Linear(emb_dim, 1).cuda()
#
#dropout = nn.Dropout(dropout_rate).cuda()
#
#rnn = nn.LSTM(emb_dim, rnn_dim, rnn_layers).cuda()
#for name, param in rnn.named_parameters():
#  if 'bias' in name:
#     nn.init.constant(param, 0.0)
#  elif 'weight' in name:
#     nn.init.xavier_normal(param)
#
#decoder = nn.Linear(rnn_dim,outVocabSize).cuda()
##pos_ptb_decoder = nn.Linear(128,len(posFine)+3).cuda()
#
#components = [rnn, decoder, word_pos_morph_embeddings]
## word_embeddings, pos_u_embeddings, pos_p_embeddings, 
##baseline, 
#
#def parameters():
# for c in components:
#   for param in c.parameters():
#      yield param
## yield dhWeights
## yield distanceWeights
#
##for pa in parameters():
##  print pa
#
#initrange = 0.1
##word_embeddings.weight.data.uniform_(-initrange, initrange)
##pos_u_embeddings.weight.data.uniform_(-initrange, initrange)
##pos_p_embeddings.weight.data.uniform_(-initrange, initrange)
##morph_embeddings.weight.data.uniform_(-initrange, initrange)
#word_pos_morph_embeddings.weight.data.uniform_(-initrange, initrange)
#
#decoder.bias.data.fill_(0)
#decoder.weight.data.uniform_(-initrange, initrange)
##pos_ptb_decoder.bias.data.fill_(0)
##pos_ptb_decoder.weight.data.uniform_(-initrange, initrange)
##baseline.bias.data.fill_(0)
##baseline.weight.data.uniform_(-initrange, initrange)
#
#


crossEntropy = 10.0

#def encodeWord(w):
#   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)



import torch.cuda
import torch.nn.functional



counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 




def createStreamContinuous(corpus):
#    global counter
    global crossEntropy
    global devLosses

    input_indices = [2] # Start of Segment
    wordStartIndices = []
#    sentenceStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       if sentCount % 10 == 0:
         print ["DEV SENTENCES", sentCount]

#       if sentCount == 100:
       #printHere = (sentCount % 10 == 0)
       ordered, _ = orderSentence(sentence, dhLogits, sentCount % 500 == 0)

#       sentenceStartIndices.append(len(input_indices))
       for line in ordered+["EOS"]:
#          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            yield "EOS"
          else:
#            targetWord = stoi[]
#            if targetWord >= vocab_size:
 #              yield line["posUni"]
  #          else:
            yield line["word"]



corpusDev = CorpusIterator(args.language,"dev", storeMorph=True).iterator(rejectShortSentences = False)
dev = list(createStreamContinuous(corpusDev))[::-1]


corpusTrain = CorpusIterator(args.language,"train", storeMorph=True).iterator(rejectShortSentences = False)
train = list(createStreamContinuous(corpusTrain))[::-1]


idev = range(len(dev))
itrain = range(len(train))

idev = sorted(idev, key=lambda i:dev[i:i+20])
itrain = sorted(itrain, key=lambda i:train[i:i+20])

print(idev)

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
   # Start is the FIRST train place that is >
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
   print(k)
   startK, endK = getStartEnd(k)
   startK2, endK2 = getStartEnd(k+1)
   cachedFollowingCounts = {}
   for j in range(len(idev)):
      start2, end2 = startK2[j], endK2[j]
      devPref = tuple(dev[idev[j]:idev[j]+k+1])
   #   print(devPref, start2, end2)
      if start2 > 0 and end2 < len(train):
        assert devPref > tuple(train[itrain[start2-1]:itrain[start2-1]+k+1]), (devPref, tuple(train[itrain[start2-1]:itrain[start2-1]+k+1]))
        assert devPref <= tuple(train[itrain[start2]:itrain[start2]+k+1]), (devPref, tuple(train[itrain[start2]:itrain[start2]+k+1]))
        assert devPref >= tuple(train[itrain[end2-1]:itrain[end2-1]+k+1])
        assert devPref < tuple(train[itrain[end2]:itrain[end2]+k+1])
#      for j2 in range(start2, end2):
 #              trainPref = tuple(train[itrain[j2]:itrain[j2]+k+1])
  #      #       print("..........", j2, trainPref)
   #            assert trainPref == devPref

      #assert start <= end
      assert start2 <= end2

      countNgram = end2-start2
      if k >= 1:
   #      print(k,countNgram)
         if idev[j]+1 < len(idevInv):
           prefixIndex = idevInv[idev[j]+1]
           assert dev[idev[prefixIndex]] == dev[idev[j]+1]
   
           prefixStart, prefixEnd = startK[prefixIndex], endK[prefixIndex]
           countPrefix = prefixEnd-prefixStart
           if countPrefix < args.gamma: # there is nothing to interpolate with, just back off
              assert k > 0
              newProbability[j] = lastProbability[j]
           else:
     #         print(k, devPref, (start2, end2), dev[idev[prefixIndex]:idev[prefixIndex]+k], (prefixStart, prefixEnd))
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
    #          print(followingCount)
              if followingCount == 0:
                  newProbability[j] = lastProbability[j]
              else:
                  #assert (followingCount == 0) == (prefixStart == prefixEnd) # This isn't fully true
          
                  probability = log(max(countNgram - args.alpha, 0.0) + args.alpha * followingCount * exp(lastProbability[j])) -  log(countPrefix)
       #           print(probability)
                  newProbability[j] = probability
         else:
            newProbability[j] = lastProbability[j]
      elif k == 0:
              probability = log(countNgram + args.delta) - log(len(train) + args.delta * len(itos))
              newProbability[j] = probability
   lastProbability = newProbability 
   newProbability = [None for _ in idev]
   assert all([x <=0 for x in lastProbability])
   try:
       surprisal = - sum([x for x in lastProbability])/len(lastProbability)
   except ValueError:
       print >> sys.stderr, "PROBLEM"
       print >> sys.stderr, lastProbability
       surprisal = 1000
   devSurprisalTable.append(surprisal)
   print("Surprisal", surprisal, len(itos))

outpath = TARGET_DIR+"/estimates-"+args.language+"_"+__file__+"_model_"+str(myID)+"_"+args.model+".txt"
print(outpath)
with open(outpath, "w") as outFile:
         print >> outFile, " ".join(sys.argv)
         print >> outFile, devSurprisalTable[-1]
         print >> outFile, " ".join(map(str,devSurprisalTable))



