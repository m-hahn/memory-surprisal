#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

import random
import sys

import torch.nn as nn
import torch
from torch.autograd import Variable
import math


from pyro.infer import SVI
from pyro.optim import Adam


import pyro
import pyro.distributions as dist

import pyro
from pyro.distributions import Normal, Bernoulli
from pyro.infer import SVI
from pyro.optim import Adam


import os

language = sys.argv[1]
languageCode = sys.argv[2]

myID = random.randint(0,10000000)

posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]

deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp
from random import random, shuffle

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

import os
sys.path.append("..")
from corpusIterator_V import CorpusIterator_V

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
     for sentence in CorpusIterator_V(language,partition).iterator():
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


# "linearization_logprobability"
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   # Loop Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum + sum(line.get("children_decisions_logprobs",[]))

   if "linearization_logprobability" in line:
      allGradients += line["linearization_logprobability"] # the linearization of this element relative to its siblings affects everything starting at the start of the constituent, but nothing to the left of it
   else:
      assert line["coarse_dep"] == "root"


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



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax, distanceWeights):
       childrenLinearized = []
       while len(remainingChildren) > 0:
           logits = torch.cat([distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]].view(1) for x in remainingChildren])
           #print logits
           if reverseSoftmax:
              logits = -logits
           #print (reverseSoftmax, logits)
           softmax = softmax_layer(logits.view(1,-1)).view(-1)
           selected = 0
#           selected = numpy.random.choice(range(0, len(remainingChildren)), p=softmax.data.numpy())
           log_probability = torch.log(softmax[selected])
           assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
           sentence[remainingChildren[selected]-1]["linearization_logprobability"] = log_probability
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
       return childrenLinearized           



def orderSentence(sentence, dhLogits, printThings, dhWeights, distanceWeights):
   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   for line in sentence:
      line["coarse_dep"] = makeCoarse(line["dep"])
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      if line["coarse_dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         continue
      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      probability = 1/(1 + torch.exp(-dhLogit))
      dhSampled = (line["index"] < line["head"])
      #else:
      #   dhSampled = (random() < probability.data.numpy())
#      logProbabilityGradient = (1 if dhSampled else -1) * (1-probability)
#      line["ordering_decision_gradient"] = logProbabilityGradient
      line["ordering_decision_log_probability"] = torch.log(1/(1 + torch.exp(- (1 if dhSampled else -1) * dhLogit)))
#      print ".."
#      print line["ordering_decision_log_probability"]
#      print line["ordering_decision_log_probability_random"]
   
      direction = "DH" if dhSampled else "HD"
#torch.exp(line["ordering_decision_log_probability"]).data.numpy()[0],
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, (str(probability.data.numpy())+"      ")[:8], str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]].data.numpy())+"    ")    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])
      sentence[headIndex]["children_decisions_logprobs"] = (sentence[headIndex].get("children_decisions_logprobs", []) + [line["ordering_decision_log_probability"]])



   for line in sentence:
      if "children_DH" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False, distanceWeights)
         line["children_DH"] = childrenLinearized
      if "children_HD" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True, distanceWeights)
         line["children_HD"] = childrenLinearized

#         shuffle(line["children_HD"])
   
   linearized = []
   recursivelyLinearize(sentence, root, linearized, Variable(torch.FloatTensor([0.0])))
   if printThings or len(linearized) == 0:
     print " ".join(map(lambda x:x["word"], sentence))
     print " ".join(map(lambda x:x["word"], linearized))


   # store new dependency links
   moved = [None] * len(sentence)
   for i, x in enumerate(linearized):
#      print x
      moved[x["index"]-1] = i
 #  print moved
   for i,x in enumerate(linearized):
  #    print x
      if x["head"] == 0: # root
         x["reordered_head"] = 0
      else:
         x["reordered_head"] = 1+moved[x["head"]-1]
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
   

itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

print itos_deps





words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))
#print stoi
#print itos[5]
#print stoi[itos[5]]

assert stoi[itos[5]] == 5

vocab_size = 50000

depLMean = 10.0
depLSquared = 100.0
l2_weight = 0.001

# 0 EOS, 1 UNK, 2 BOS
#word_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = 50).cuda()
#pos_u_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = 10).cuda()
#pos_p_embeddings = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=10).cuda()



#dropout = nn.Dropout(0.5).cuda()

#rnn = nn.LSTM(70, 128, 2).cuda()
#for name, param in rnn.named_parameters():
#  if 'bias' in name:
#     nn.init.constant(param, 0.0)
#  elif 'weight' in name:
#     nn.init.xavier_normal(param)
#
#decoder = nn.Linear(128,vocab_size+3).cuda()
#pos_ptb_decoder = nn.Linear(128,len(posFine)+3).cuda()
#
#components = [word_embeddings, pos_u_embeddings, pos_p_embeddings, rnn, decoder, pos_ptb_decoder]


#def parameters_lm():
# for c in components:
#   for param in c.parameters():
#      yield param


#def parametersRandom():
#  yield dhWeights
#  yield distanceWeights 


#def parameters():
# yield intercept 
# yield depLWeight
# yield surpWWeight
# yield surpPWeight
#for pa in parameters():
#  print pa

initrange = 0.1
#word_embeddings.weight.data.uniform_(-initrange, initrange)
#pos_u_embeddings.weight.data.uniform_(-initrange, initrange)
#pos_p_embeddings.weight.data.uniform_(-initrange, initrange)
#decoder.bias.data.fill_(0)
#decoder.weight.data.uniform_(-initrange, initrange)
#pos_ptb_decoder.bias.data.fill_(0)
#pos_ptb_decoder.weight.data.uniform_(-initrange, initrange)

batchSize = 1

lr_regression = 0.001
lr_lm = 0.1


crossEntropy = 10.0

def encodeWord(w):
   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)



#import torch.cuda
import torch.nn.functional

logsoftmax = torch.nn.LogSoftmax()

def deepCopy(sentence):
  result = []
  for w in sentence:
     entry = {}
     for key, value in w.iteritems():
       entry[key] = value
     result.append(entry)
  return result
dhWeights_Prior = Normal(Variable(torch.FloatTensor([0.0] * len(itos_deps))), Variable(torch.FloatTensor([1.0]* len(itos_deps))))
distanceWeights_Prior = Normal(Variable(torch.FloatTensor([0.0] * len(itos_deps))), Variable(torch.FloatTensor([1.0]* len(itos_deps))))

counter = 0
corpus = CorpusIterator_V(language,"train")

def guide(corpus):
  mu_DH = pyro.param("mu_DH", Variable(torch.FloatTensor([0.0]*len(itos_deps)), requires_grad=True))
  mu_Dist = pyro.param("mu_Dist", Variable(torch.FloatTensor([0.0]*len(itos_deps)), requires_grad=True))

  sigma_DH = pyro.param("sigma_DH", Variable(torch.FloatTensor([1.0]*len(itos_deps)), requires_grad=True))
  sigma_Dist = pyro.param("sigma_Dist", Variable(torch.FloatTensor([1.0]*len(itos_deps)), requires_grad=True))

  dhWeights = pyro.sample("dhWeights", dist.Normal(mu_DH, sigma_DH)) #Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
  distanceWeights = pyro.sample("distanceWeights", dist.Normal(mu_Dist, sigma_Dist)) #Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)

def model(corpus):
  global counter
  dhWeights = pyro.sample("dhWeights", dhWeights_Prior) #Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
  distanceWeights = pyro.sample("distanceWeights", distanceWeights_Prior) #Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
  


  for q in pyro.irange("data_loop", corpus.length(), subsample_size=5, use_cuda=False):
       point = corpus.getSentence(q)
       current = [point]
       counter += 1
       printHere = (counter % 100 == 0)
       batchOrderedLogits = zip(*map(lambda (y,x):orderSentence(x, dhLogits, y % batchSize==0 and printHere, dhWeights, distanceWeights), zip(range(len(current)),current)))
      
       batchOrdered = batchOrderedLogits[0]
       lengths = map(len, current)
       maxLength = lengths[int(0.8*batchSize)]

       assert batchSize == 1
       
       if printHere:
         print "BACKWARD 3 "+__file__+" "+language+" "+str(myID)+" "+str(counter)

       logitCorr = batchOrdered[0][-1]["relevant_logprob_sum"]
       pyro.sample("result_Correct_{}".format(q),  Bernoulli(logits=logitCorr), obs=Variable(torch.FloatTensor([1.0])))
#       pyro.observe("result_Correct_{}".format(q), Bernoulli(logits=logitCorr), Variable(torch.FloatTensor([1.0])))

       


adam_params = {"lr": 0.001, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
from pyro.infer import Trace_ELBO
svi = SVI(model, guide, optimizer, loss=Trace_ELBO()) #, num_particles=7)

n_steps = 400000
# do gradient steps
for step in range(1,n_steps):
    if step % 100 == 1:
      print "DOING A STEP"
      print "......."
      print step

 #   print 
#    quit()

#    for name in pyro.get_param_store().get_all_param_names():
#       print [name, pyro.param(name).data.numpy()]
    svi.step(corpus)

    if step % 2000 == 0:
       print "Saving"
       save_path = "/u/scr/mhahn/deps/"
       #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
       with open(save_path+"/manual_output_ground_coarse/"+language+"_"+__file__+"_model_"+str(myID)+".tsv", "w") as outFile:
          print >> outFile, "\t".join(map(str,["FileName","ModelName","Counter", "AverageLoss","DH_Mean_NoPunct","DH_Sigma_NoPunct", "Distance_Mean_NoPunct", "Distance_Sigma_NoPunct", "Dependency"]))
          dh_numpy = pyro.get_param_store().get_param("mu_DH").data.numpy()
          dh_sigma_numpy = pyro.get_param_store().get_param("sigma_DH").data.numpy()
          dist_numpy = pyro.get_param_store().get_param("mu_Dist").data.numpy()
          dist_sigma_numpy = pyro.get_param_store().get_param("sigma_Dist").data.numpy()

          for i in range(len(itos_deps)):
             key = itos_deps[i]
             dependency = key
             print >> outFile, "\t".join(map(str,[myID, __file__, counter, crossEntropy, dh_numpy[i], dh_sigma_numpy[i], dist_numpy[i], dist_sigma_numpy[i], dependency]))





