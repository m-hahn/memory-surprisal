#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Derived from mis_LDC2012T05__ByRegion_ByConfiguration.py, try to clearly capture the thing coming BEFORE


import random
import sys

objectiveName = "LM"

language = "PTB"
languageCode = "PTB"
dropout_rate = 0.1
emb_dim = 100
rnn_dim = 128
rnn_layers = 1
lr_lm = 0.01
model = sys.argv[1]
assert model in ["RAN", "RN", "NR", "ANR"]
input_dropoutRate = 0.2
batchSize = 2
replaceWordsProbability = 0.0
horizon = 20
prescripedID = None
gpuNumber = "GPU0"
assert gpuNumber.startswith("GPU")
gpuNumber = int(gpuNumber[3:])

assert len(sys.argv) == 2


assert dropout_rate <= 0.5
assert input_dropoutRate <= 0.5

devSurprisalTable = [None] * horizon
if prescripedID is not None:
  myID = int(prescripedID)
else:
  myID = random.randint(0,10000000)

DOING_PARAMETER_SEARCH =False
import sys
print  >> sys.stderr, ("DOING PARAMETER SEARCH?", DOING_PARAMETER_SEARCH)
assert not DOING_PARAMETER_SEARCH

TARGET_DIR = "/u/scr/mhahn/deps/memory-need-neural-wordforms_fullVocab_chinlang/"

#with open("/juicier/scr120/scr/mhahn/deps/LOG"+language+"_"+__file__+"_model_"+str(myID)+".txt", "w") as outFile:
 #   print >> outFile, " ".join(sys.argv)



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
   for partition in ["train", "dev", "test"]:
     for sentence in CorpusIterator(language,partition, storeMorph=True).iterator():
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
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum, region):
   line = sentence[position-1]
   if region[0].startswith("n_") or region[0] == "between":
      line["region"] = region
   if "region" not in line:
       line["region"] = ("NONE",)
   if line["region"][0].startswith("n_") or region[0] == "between":
      region =        line["region"]
   # Loop Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum #+ sum(line.get("children_decisions_logprobs",[]))

#   if "linearization_logprobability" in line:
#      allGradients += line["linearization_logprobability"] # the linearization of this element relative to its siblings affects everything starting at the start of the constituent, but nothing to the left of it
#   else:
#      assert line["dep"] == "root"


   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients, region)
   result.append(line)
#   print ["DECISIONS_PREPARED", line["index"], line["word"], line["dep"], line["head"], allGradients.data.numpy()[0]]
   line["relevant_logprob_sum"] = allGradients
   if "children_HD" in line:
      for child in line["children_HD"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients, region)
   return allGradients

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       global model
#       childrenLinearized = []
#       while len(remainingChildren) > 0:
       x = [x for x in remainingChildren if "postpone" in sentence[x-1]]
#       if len(x) > 0:
 #        print([sentence[x-1]["word"] for x in remainingChildren])
       if True:
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

wordByRegion = {}

afterAndHeadJoint = []
beforeAndHeadJoint = []
beforeAndRelVbJoint = []
afterAndRelVbJoint = []
headAndRelVbJoint = []
headAndHeadJoint = []

headheadAndHeadJoint = {"HD" : [], "DH" : []}
headheadAndRelVbJoint = {"HD" : [], "DH" : []}


def getFirstIndexOfChildren(line):
    return min([line["index"]] + [getFirstIndexOfChildren(child) for child in line.get("children", [])])
def getLastIndexOfChildren(line):
    return max([line["index"]] + [getLastIndexOfChildren(child) for child in line.get("children", [])])

def orderSentence(sentence, dhLogits, printThings):
   global model


   headNouns = []
   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   if model == "REAL_REAL":
      eliminated = []
   for line in sentence:
      if line["dep"] == "root":
          continue
      if line["dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         continue
      head = sentence[line["head"]-1] 
      if "children" not in head:
          head["children"] = []
      head["children"].append(line)

   relevantPieces = []

   for line in sentence:
      if line["dep"] == "root":
          root = line["index"]
          continue
      if line["dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         if model == "REAL_REAL":
            eliminated.append(line)
         continue
      
      key = (sentence[line["head"]-1]["posUni"], line["dep"], line["posUni"])
      if key[0] == "NN" and key[1].startswith("acl"): # == "acl:relcl": # and key[2].startswith("V"): # == ("NN", "acl", "v"):
          line["region"] = "n_att_v"
#          print "===================="
 #         print "\t".join([str(key), sentence[line["head"]-1]["word"], line["word"]])
          #printThings = True
          line["postpone"] = True
      if key[0] == "NN" and key[1] == "amod" and key[2] == "JJ" and model not in ["ANR", "RAN"]:
          line["region"] = "n_att_a"
  #        print "===================="
   #       print "\t".join([str(key), sentence[line["head"]-1]["word"], line["word"]])
          #printThings = True
          line["postpone"] = True

      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
#      probability = 1/(1 + torch.exp(-dhLogit))
      if True:
         if "postpone" in line:
            relation = sentence[line["head"]-1]["dep"]
            headHead = sentence[line["head"]-1]["head"]-1
            if headHead == -1:
               dirWRTHeadHead = "ROOT"
               headHead = "ROOT"
            else:
               dirWRTHeadHead = ("DH" if (line["head"]-1) < headHead else "HD")
               headHead = sentence[headHead]["word"]
            #print(relation+"\t"+headHead+"\t"+dirWRTHeadHead)
            if headHead in ["把", "将"] and relation == "pob":
                relation = "ba"
            #    print("BA CONSTRUCTION")
            typeOfObservation = (dirWRTHeadHead, relation)

            sentence[line["head"]-1]["region"] = ("HeadNoun",) + typeOfObservation
            line["region"] = (line["region"],) + typeOfObservation

            firstWordAfterHead = line["head"]-1+1
            afterHead = "EOS"
            while firstWordAfterHead < len(sentence):
                if sentence[firstWordAfterHead]["posUni"] == "wp":
                     firstWordAfterHead += 1
                else:
                    afterHead = sentence[firstWordAfterHead]["word"]
                    break

            relativeClauseSpan = ( getFirstIndexOfChildren(line)-1, getLastIndexOfChildren(line)-1)
            betweenAndHeadSpan = (line["head"]-1, relativeClauseSpan[0]-1)

            beforeSpan = (0, betweenAndHeadSpan[0]-1)
            afterSpan = (relativeClauseSpan[1]+1, len(sentence)-1)
            relevantPiece = {}
            relevantPiece["before"] = sentence[beforeSpan[0]:beforeSpan[1]+1]
            relevantPiece["relativeClause"] = sentence[relativeClauseSpan[0]:relativeClauseSpan[1]+1]
            relevantPiece["betweenAndHead"] = sentence[betweenAndHeadSpan[0]:betweenAndHeadSpan[1]+1]
            relevantPiece["after"] = sentence[afterSpan[0]:afterSpan[1]+1]
            

            if True or dirWRTHeadHead == "HD":
               if len(relevantPiece["before"]) >= 5:
                  if len(relevantPiece["after"]) >= 5:
                     relevantPiece["before"] = relevantPiece["before"][-5:]
                     relevantPiece["after"] = relevantPiece["after"][:5]
                     relevantPieces.append(relevantPiece)

#                     print("===========")
#                     print(" ".join([x["word"] for x in relevantPiece["before"]]))
#                     print(" ".join([x["word"] for x in relevantPiece["betweenAndHead"]]))
#                     print(" ".join([x["word"] for x in relevantPiece["relativeClause"]]))
#                     print(" ".join([x["word"] for x in relevantPiece["after"]]))


                     assert len(relevantPiece["after"]) == 5
                     assert len(relevantPiece["before"]) == 5

            #print(beforeSpan, relativeClauseSpan, betweenAndHeadSpan, afterSpan)
            firstIndexOfChildren = relativeClauseSpan[0]
            beforeRelCl = "SOS"
            while firstIndexOfChildren > 0:
                 if sentence[firstIndexOfChildren-1]["posUni"] == "wp":
                       firstIndexOfChildren -= 1
                 else:
                       beforeRelCl = sentence[firstIndexOfChildren-1]["word"]
                       break

            headWord = sentence[line["head"]-1]["word"]

            # 4.14301878143
            # 2.80087874316
            # 3.97867212314

            if dirWRTHeadHead == "HD": # and afterHead != "EOS":
                #print(" ".join([headHead, "_", dirWRTHeadHead, "_", relation,  "_", headWord, "_", line["word"], "----", afterHead, "----", beforeRelCl]))
                #print " ".join(map(lambda x:x["word"], sentence))

                beforeAndHeadJoint.append((beforeRelCl, headHead)) # before corresponds to the `head' in this variable name.
                afterAndHeadJoint.append((headWord, afterHead))
                beforeAndRelVbJoint.append((beforeRelCl, line["word"])) # Noun and RelVb

                headAndRelVbJoint.append((headHead, line["word"])) # Noun and RelVb
                afterAndRelVbJoint.append((afterHead, line["word"])) # Noun and RelVb
                headAndHeadJoint.append((headWord, headHead)) # before corresponds to the `head' in this variable name.

            if dirWRTHeadHead != "ROOT":
              headheadAndHeadJoint[dirWRTHeadHead].append((headWord, headHead))
              headheadAndRelVbJoint[dirWRTHeadHead].append((line["word"], headHead))
              
            siblings = sentence[line["head"]-1]["children"]
            siblingsBetweenThisAndNoun = [x for x in siblings if x["index"] < line["head"] and x["index"] > line["index"]]
#            print(" ".join([x["word"] for x in siblingsBetweenThisAndNoun]))
            for x in siblingsBetweenThisAndNoun:
               if "region" not in x:
                  x["region"] = ("between",)+typeOfObservation

            if model in ["RN", "RAN"]:
                dhSampled = True
            elif model in ["NR", "ANR"]:
                dhSampled = False
            else:
                assert False
         else:
            dhSampled = (line["head"] > line["index"]) #(random() < probability.data.numpy()[0])
      else:
         dhSampled = (dhLogit > 0) #(random() < probability.data.numpy())
    
      direction = "DH" if dhSampled else "HD"
#torch.exp(line["ordering_decision_log_probability"]).data.numpy()[0],
#      if True or printThings: 
 #        print "\t".join(map(str,["ORD", line["index"], ((line["word"])+"           ")[:10], ("->".join(list(key)) + "         ")[:22], line["head"], dhLogit, dhSampled, direction]))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])
      #sentence[headIndex]["children_decisions_logprobs"] = (sentence[headIndex].get("children_decisions_logprobs", []) + [line["ordering_decision_log_probability"]])

   return relevantPieces


   if model != "REAL_REAL":
      for line in sentence:
         if "children_DH" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
            line["children_DH"] = childrenLinearized
         if "children_HD" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
            line["children_HD"] = childrenLinearized
   if model == "REAL_REAL":
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
   recursivelyLinearize(sentence, root, linearized, 0, ("NONE",))
   if model == "REAL_REAL":
      linearized = filter(lambda x:"removed" not in x, sentence)
   for i, line in enumerate(linearized):
       if i+1 < len(linearized) and line["region"] == ("NONE",) and linearized[i+1]["region"] != ("NONE",):
           line["region"] = ("before",)+linearized[i+1]["region"][1:]
       if i > 0 and line["region"] == ("NONE",) and linearized[i-1]["region"] != ("NONE",) and not linearized[i-1]["region"][0].startswith("after"):
           line["region"] = ("after",)+linearized[i-1]["region"][1:]
       if i > 1 and line["region"] == ("NONE",) and linearized[i-2]["region"] != ("NONE",) and not linearized[i-2]["region"][0].startswith("after"):
           line["region"] = ("after",)+linearized[i-2]["region"][1:]
       if i > 2 and line["region"] == ("NONE",) and linearized[i-3]["region"] != ("NONE",) and not linearized[i-3]["region"][0].startswith("after"):
           line["region"] = ("after",)+linearized[i-3]["region"][1:]

   #print [x["region"] for x in sentence if "region" in x]
   #print [x["region"] for x in linearized]
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

#if model != "RANDOM_MODEL" and model != "REAL" and model != "RANDOM_BY_TYPE":
#   inpModels_path = "/juicier/scr120/scr/mhahn/deps/"+"/manual_output/"
#   models = os.listdir(inpModels_path)
#   models = filter(lambda x:"_"+model+".tsv" in x, models)
#   if len(models) == 0:
#     assert False, "No model exists"
#   if len(models) > 1:
#     assert False, [models, "Multiple models exist"]
#   
#   with open(inpModels_path+models[0], "r") as inFile:
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
if model == "RANDOM_MODEL":
  for key in range(len(itos_deps)):
     dhWeights[key] = random() - 0.5
     distanceWeights[key] = random()
  originalCounter = "NA"
elif model in ["NR", "RN", "ANR", "RAN"] or model == "REAL_REAL":
  originalCounter = "NA"
elif model == "RANDOM_BY_TYPE":
  dhByType = {}
  distByType = {}
  for dep in itos_pure_deps:
    dhByType[dep.split(":")[0]] = random() - 0.5
    distByType[dep.split(":")[0]] = random()
  for key in range(len(itos_deps)):
     dhWeights[key] = dhByType[itos_deps[key][1].split(":")[0]]
     distanceWeights[key] = distByType[itos_deps[key][1].split(":")[0]]
  originalCounter = "NA"
elif model == "RANDOM_BY_TYPE_CONS":
  distByType = {}
  for dep in itos_pure_deps:
    distByType[dep.split(":")[0]] = random()
  for key in range(len(itos_deps)):
     dhWeights[key] = 1.0
     distanceWeights[key] = distByType[itos_deps[key][1].split(":")[0]]
  originalCounter = "NA"
elif model == "RANDOM_MODEL_CONS":
  for key in range(len(itos_deps)):
     dhWeights[key] = 1.0
     distanceWeights[key] = random()
  originalCounter = "NA"
elif model == "GROUND":
  groundPath = "/u/scr/mhahn/deps/manual_output_ground_coarse/"
  import os
  files = [x for x in os.listdir(groundPath) if x.startswith(language+"_infer")]
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
         print(dependency, dhHere, distHere)
         dhByDependency[dependency] = dhHere
         distByDependency[dependency] = distHere
  for key in range(len(itos_deps)):
     dhWeights[key] = dhByDependency[itos_deps[key][1].split(":")[0]]
     distanceWeights[key] = distByDependency[itos_deps[key][1].split(":")[0]]
  originalCounter = "NA"
else:
  assert False

lemmas = list(vocab_lemmas.iteritems())
lemmas = sorted(lemmas, key = lambda x:x[1], reverse=True)
itos_lemmas = map(lambda x:x[0], lemmas)
stoi_lemmas = dict(zip(itos_lemmas, range(len(itos_lemmas))))


#          state = {"arguments" : sys.argv, "words" : itos, "components" : [c.state_dict() for c in components]}
#          torch.save(state, "/u/scr/mhahn/MODELS/"+language+"_"+__file__+"_code_"+str(myID)+".txt")


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

vocab_size = 50000
vocab_size = min(len(itos),vocab_size)
import sys
print >> sys.stderr, ("VOCAB_SIZE", vocab_size)
#print itos[:vocab_size]
#quit()

# 0 EOS, 1 UNK, 2 BOS


print posUni
#print posFine
print "VOCABULARY "+str(vocab_size+3)
outVocabSize = len(itos_pos_uni)+vocab_size+3
#quit()


itos_total = ["EOS", "EOW", "SOS"] + itos_pos_uni + itos[:vocab_size]
assert len(itos_total) == outVocabSize
# could also provide per-word subcategorization frames from the treebank as input???

#parameterList = list(parameters())



in_rnn_after = nn.LSTM(rnn_dim, rnn_dim, rnn_layers, bidirectional=True).cuda()
in_rnn_before = nn.LSTM(rnn_dim, rnn_dim, rnn_layers, bidirectional=True).cuda()

out_rnn_betwHd = nn.LSTM(rnn_dim, rnn_dim, rnn_layers, bidirectional=True).cuda()
#out_rnn_betweenHead_fromBefore = nn.LSTM(rnn_dim, rnn_dim, rnn_layers, bidirectional=True).cuda()
#out_rnn_betweenHead_fromAfter = nn.LSTM(rnn_dim, rnn_dim, rnn_layers, bidirectional=True).cuda()

rnns = [in_rnn_after, in_rnn_before]
rnns += [out_rnn_betwHd]
#rnns += [out_rnn_betweenHead_fromBefore, out_rnn_betweenHead_fromAfter]

for rnn in rnns:
  for name, param in rnn.named_parameters():
    if 'bias' in name:
       nn.init.constant(param, 0.0)
    elif 'weight' in name:
       nn.init.xavier_normal(param)
  
embeddings = torch.nn.Embedding(num_embeddings = outVocabSize, embedding_dim=rnn_dim).cuda()
decoder = nn.Linear(rnn_dim,outVocabSize).cuda()
#decoder.weight.data = embeddings.weight.data
decoder.bias.data.fill_(0)

components = [decoder, embeddings] + rnns

def parameters():
 for c in components:
   for param in c.parameters():
      yield param

initrange = 0.1






crossEntropy = 10.0

#def encodeWord(w):
#   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)



import torch.nn.functional

inputDropout = torch.nn.Dropout2d(p=input_dropoutRate)


counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 


lossModule = nn.NLLLoss()
lossModuleTest = nn.CrossEntropyLoss(size_average=False, reduce=False, ignore_index=0)


def numerify(sentence):
    input_indices = [] # Start of Segment
    for line in sentence:
         if random() < replaceWordsProbability:
             targetWord = randint(0,vocab_size-1)
         else:
             targetWord = stoi[line["word"]]
         if targetWord >= vocab_size:
            input_indices.append(stoi_pos_uni[line["posUni"]]+3)
         else:
            input_indices.append(targetWord+3+len(itos_pos_uni))
    return [2]+input_indices+[2]


def expandLeft(ls, tg):
   return [0 for _ in range(tg-len(ls))] + ls
   
def expandRight(ls, tg):
   return ls + [0 for _ in range(tg-len(ls))]

def doForwardPass(sentence, ordered, surprisalTable=None, doDropout=True, batchSizeHere=1, printHere=True):
       global counter
       global crossEntropy
       global devLosses


       betwHd = torch.LongTensor(numerify(ordered["betweenAndHead"])).cuda()
       after = torch.LongTensor(numerify(ordered["after"])).cuda()
       before = torch.LongTensor(numerify(ordered["before"])).cuda()


       betwHd_emb = embeddings(betwHd).unsqueeze(1)
       after_emb = embeddings(after).unsqueeze(1)
       before_emb = embeddings(before).unsqueeze(1)

       _, after_enc = in_rnn_after(after_emb)
       _, before_enc = in_rnn_before(before_emb)

       out_betwHd, _ = out_rnn_betwHd(betwHd_emb, None)
       out_betwHd_fromBefore, _ = out_rnn_betwHd(betwHd_emb, before_enc)
       out_betwHd_fromAfter, _ = out_rnn_betwHd(betwHd_emb, after_enc)


       out_betwHd = decoder(out_betwHd.view(out_betwHd.size()[0], 2,rnn_dim))
       out_betwHd_forward = out_betwHd[:,0][:-1]
       out_betwHd_backward = out_betwHd[:,1][1:]
       betwHd_target_forward = betwHd[1:]
       betwHd_target_backward = betwHd[:-1]

       loss_betwHd_forward = lossModuleTest(out_betwHd_forward, betwHd_target_forward)
       loss_betwHd_backward = lossModuleTest(out_betwHd_backward, betwHd_target_backward)



       out_betwHd_fromBefore = decoder(out_betwHd_fromBefore.view(out_betwHd_fromBefore.size()[0], 2,rnn_dim))
       #print(out_betwHd_fromBefore.size())
       out_betwHd_fromBefore_forward = out_betwHd_fromBefore[:,0][:-1]
       out_betwHd_fromBefore_backward = out_betwHd_fromBefore[:,1][1:]
       betwHd_fromBefore_target_forward = betwHd[1:]
       betwHd_fromBefore_target_backward = betwHd[:-1]

       #print(out_betwHd_fromBefore_forward.size(), betwHd_fromBefore_target_forward.size())
       loss_betwHd_fromBefore_forward = lossModuleTest(out_betwHd_fromBefore_forward, betwHd_fromBefore_target_forward)
       loss_betwHd_fromBefore_backward = lossModuleTest(out_betwHd_fromBefore_backward, betwHd_fromBefore_target_backward)


       out_betwHd_fromAfter = decoder(out_betwHd_fromAfter.view(out_betwHd_fromAfter.size()[0], 2,rnn_dim))
       #print(out_betwHd_fromAfter.size())
       out_betwHd_fromAfter_forward = out_betwHd_fromAfter[:,0][:-1]
       out_betwHd_fromAfter_backward = out_betwHd_fromAfter[:,1][1:]
       betwHd_fromAfter_target_forward = betwHd[1:]
       betwHd_fromAfter_target_backward = betwHd[:-1]

       #print(out_betwHd_fromAfter_forward.size(), betwHd_fromAfter_target_forward.size())
       loss_betwHd_fromAfter_forward = lossModuleTest(out_betwHd_fromAfter_forward, betwHd_fromAfter_target_forward)
       loss_betwHd_fromAfter_backward = lossModuleTest(out_betwHd_fromAfter_backward, betwHd_fromAfter_target_backward)










       loss = loss_betwHd_forward.sum() + loss_betwHd_backward.sum()
       loss += loss_betwHd_fromBefore_forward.sum() + loss_betwHd_fromBefore_backward.sum()
       loss += loss_betwHd_fromAfter_forward.sum() + loss_betwHd_fromAfter_backward.sum()

       for c in components:
          c.zero_grad()
       MIBetwHdWithBefore = (float(loss_betwHd_forward.sum()) + float(loss_betwHd_backward.sum()) - float(loss_betwHd_fromBefore_forward.sum()) - float(loss_betwHd_fromBefore_backward.sum()))/2
       MIBetwHdWithAfter = (float(loss_betwHd_forward.sum()) + float(loss_betwHd_backward.sum()) - float(loss_betwHd_fromAfter_forward.sum()) - float(loss_betwHd_fromAfter_backward.sum()))/2

       global ra_MIBetwHdWithBefore 
       global ra_MIBetwHdWithAfter 
       global ra_MIStrongerBefore

       ra_MIBetwHdWithBefore = 0.999 * ra_MIBetwHdWithBefore + (1-0.999) * MIBetwHdWithBefore
       ra_MIBetwHdWithAfter = 0.999 * ra_MIBetwHdWithAfter + (1-0.999) * MIBetwHdWithAfter
       ra_MIStrongerBefore = 0.999 * ra_MIStrongerBefore + (1-0.999) * (1 if MIBetwHdWithBefore>MIBetwHdWithAfter else 0)
       if random() < 0.01:
          print("HEAD")
          print("\t".join([str(x) for x in (float(loss_betwHd_forward.mean()), float(loss_betwHd_backward.mean()))]))
          print("\t".join([str(x) for x in (float(loss_betwHd_fromBefore_forward.mean()), float(loss_betwHd_fromBefore_backward.mean()))]))
          print("\t".join([str(x) for x in (float(loss_betwHd_fromAfter_forward.mean()), float(loss_betwHd_fromAfter_backward.mean()))]))

          print("------- "+str(EPOCH)+" "+str(SENT))
          print("MI BetwHd Before", round(ra_MIBetwHdWithBefore,2), "After", round(ra_MIBetwHdWithAfter,2), "    ", ra_MIStrongerBefore)

       return loss

ra_MIBetwHdWithBefore = 0
ra_MIBetwHdWithAfter = 0
ra_MIBetwHdWithBefore = 0
ra_MIBetwHdWithAfter = 0
ra_MIStrongerBefore = 0.5

def  doBackwardPass(loss):
       global lastDevLoss
       global failedDevRuns
       loss.backward()
#       torch.nn.utils.clip_grad_norm(parameterList, 5.0, norm_type='inf')
       for param in parameters():
         if param.grad is None:
           if random()< 0.001:
               print "WARNING: None gradient"
           continue
         param.data.sub_(lr_lm * param.grad.data)






def computeDevLoss():
   devBatchSize = 512
   global printHere
#   global counter
#   global devSurprisalTable
   global horizon
   devLoss = 0.0
   devWords = 0
#   corpusDev = getNextSentence("dev")
   corpusDev = CorpusIterator(language,"dev", storeMorph=True).iterator(rejectShortSentences = False)
   stream = createStreamContinuous(corpusDev)

   surprisalTable = [0 for _ in range(horizon)]
   devCounter = 0
   devCounterTimesBatchSize = 0
   while True:
#     try:
#        input_indices, wordStartIndices = next(stream)
     try:
        input_indices_list = []
        wordStartIndices_list = []
        for _ in range(devBatchSize):
           input_indices, wordStartIndices = next(stream)
           input_indices_list.append(input_indices)
           wordStartIndices_list.append(wordStartIndices)
     except StopIteration:
        devBatchSize = len(input_indices_list)
#        break
     if devBatchSize == 0:
       break
     devCounter += 1
#     counter += 1
     printHere = (devCounter % 100 == 0)
     _, _, _, newLoss, newWords = doForwardPass(input_indices_list, wordStartIndices_list, surprisalTable = surprisalTable, doDropout=False, batchSizeHere=devBatchSize)
     devLoss += newLoss
     devWords += newWords
     if printHere:
         print "Dev examples "+str(devCounter)
     devCounterTimesBatchSize += devBatchSize
   devSurprisalTableHere = [surp/(devCounterTimesBatchSize) for surp in surprisalTable]
   return devLoss/devWords, devSurprisalTableHere







depLengths = 0
depsNum = 0

for EPOCH in range(30):
 corpusDev = CorpusIterator(language,"train", storeMorph=True).iterator(rejectShortSentences = False)
 SENT = 0
 for sentence in corpusDev:
   ordereds = orderSentence(sentence, dhLogits, False)
   for ordered in ordereds:
      SENT += 1
      #print(list(ordered))
      loss = doForwardPass(sentence, ordered)
      doBackwardPass(loss)
# ['relativeClause', 'betweenAndHead', 'after', 'before']



quit()

DEV_PERIOD = 5000
epochCount = 0
corpusBase = CorpusIterator(language, storeMorph=True)
while failedDevRuns == 0:
  epochCount += 1
  print "Starting new epoch, permuting corpus"
  corpusBase.permute()
#  corpus = getNextSentence("train")
  corpus = corpusBase.iterator(rejectShortSentences = False)
  stream = createStream(corpus)



  if counter > 5:
#       if counter % DEV_PERIOD == 0:
          newDevLoss, devSurprisalTableHere = computeDevLoss()
#             devLosses.append(
          devLosses.append(newDevLoss)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
          if newDevLoss > 15 or len(devLosses) > 99:
              print "Abort, training too slow?"
              devLosses.append(newDevLoss+0.001)

          if lastDevLoss is None or newDevLoss < lastDevLoss:
              devSurprisalTable = devSurprisalTableHere
#          if counter == DEV_PERIOD and model != "REAL_REAL":
#             with open(TARGET_DIR+"/model-"+language+"_"+__file__+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
#                 print >> outFile, "\t".join(["Key", "DH_Weight", "Distance_Weight"])
#                 for i, key in enumerate(itos_deps):
#                   #dhWeight = dhWeights[i]
#                   distanceWeight = distanceWeights[i]
#                   print >> outFile, "\t".join(map(str,[key, dhWeight, distanceWeight]))
          
          with open(TARGET_DIR+"/estimates-"+language+"_"+__file__+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
              print >> outFile, " ".join(sys.argv)
              print >> outFile, " ".join(map(str,devLosses))
              print >> outFile, " ".join(map(str,devSurprisalTable))
              print >> outFile, "PARAMETER_SEARCH" if DOING_PARAMETER_SEARCH else "RUNNING"

          if newDevLoss > 15 or len(devLosses) > 100:
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
             print zip(range(1,horizon+1), devSurprisalTable)


             break


          state = {"arguments" : sys.argv, "words" : itos, "components" : [c.state_dict() for c in components]}
          torch.save(state, "/u/scr/mhahn/MODELS/"+language+"_"+__file__+"_code_"+str(myID)+".txt")



  while True:
       counter += 1
       printHere = (counter % 100 == 0)

       try:
          input_indices_list = []
          wordStartIndices_list = []
          for _ in range(batchSize):
             input_indices, wordStartIndices = next(stream)
             input_indices_list.append(input_indices)
             wordStartIndices_list.append(wordStartIndices)
       except StopIteration:
          break
       loss, baselineLoss, policy_related_loss, _, wordNumInPass = doForwardPass(input_indices_list, wordStartIndices_list, batchSizeHere=batchSize)
       if wordNumInPass > 0:
         doBackwardPass(loss, baselineLoss, policy_related_loss)
       else:
         print "No words, skipped backward"
       if printHere:
          print "Epoch "+str(epochCount)+" "+str(counter)
          print zip(range(1,horizon+1), devSurprisalTable)




import math

def mi(xyp):
   xs = {}
   ys = {}
   xys = {}
   for x, y in xyp:
        xs[x] = xs.get(x,0)+1
        ys[y] = ys.get(y,0)+1
        xys[(x,y)] = xys.get((x,y),0) + 1
   mi = 0
   for x, y in xys:
      joint = math.log(xys[(x,y)]) - math.log(len(xyp))
      xm = math.log(xs[x]) - math.log(len(xyp))
      ym = math.log(ys[y]) - math.log(len(xyp))
      mi += float(xys[(x,y)])/len(xyp) * (joint - xm - ym)
   return mi

#4.14301878143
#2.80087874316
print("In the HD case")
print(mi(beforeAndHeadJoint))  #4.51
print(mi(afterAndHeadJoint))   #2.80087874316
print(mi(beforeAndRelVbJoint)) #3.69
print(mi(headAndRelVbJoint))   #3.97
print(mi(afterAndRelVbJoint))  #2.97
print(mi(headAndHeadJoint))    #4.14


#
#                beforeAndHeadJoint.append((beforeRelCl, headHead)) # before corresponds to the `head' in this variable name.
#                afterAndHeadJoint.append((headWord, afterHead))
#                beforeAndRelVbJoint.append((beforeRelCl, line["word"])) # Noun and RelVb
#
#                headAndRelVbJoint.append((headHead, line["word"])) # Noun and RelVb
#                afterAndRelVbJoint.append((afterHead, line["word"])) # Noun and RelVb
#                headAndHeadJoint.append((headWord, headHead)) # before corresponds to the `head' in this variable name.
#

print("DH")
print(mi(headheadAndHeadJoint["DH"])) # 4.95
print(mi(headheadAndRelVbJoint["DH"]))# 5.10
print("HD")
print(mi(headheadAndHeadJoint["HD"])) # 4.14
print(mi(headheadAndRelVbJoint["HD"])) # 3.98


