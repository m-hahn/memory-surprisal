import random
import sys

objectiveName = "LM"

#language = sys.argv[1]
#languageCode = sys.argv[2]
#dropout_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.33
#emb_dim = int(sys.argv[4]) if len(sys.argv) > 4 else 100
#rnn_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 512
#rnn_layers = int(sys.argv[6]) if len(sys.argv) > 6 else 2
#lr_lm = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
model = "RANDOM_MODEL" #sys.argv[8]
#input_dropoutRate = float(sys.argv[9]) # 0.33
#batchSize = int(sys.argv[10])
#replaceWordsProbability = float(sys.argv[11])
#horizon = int(sys.argv[12]) if len(sys.argv) > 12 else 20
#prescripedID = sys.argv[13] if len(sys.argv)> 13 else None
#gpuNumber = sys.argv[14] if len(sys.argv) > 14 else "GPU0"
#assert gpuNumber.startswith("GPU")
#gpuNumber = int(gpuNumber[3:])
#
##if len(sys.argv) == 13:
##  del sys.argv[12]
#assert len(sys.argv) in [12,13,14, 15]
#
#
#assert dropout_rate <= 0.5
#assert input_dropoutRate <= 0.5
#
#devSurprisalTable = [None] * horizon
#if prescripedID is not None:
#  myID = int(prescripedID)
#else:
#  myID = random.randint(0,10000000)
#
#
#TARGET_DIR = "/u/scr/mhahn/deps/memory-need-neural-wordforms/"


posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]



from math import log, exp
from random import random, shuffle, randint


from corpusIterator_PTB_Deps import CorpusIterator_PTB

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}

import nltk.tree

corpus_cached = {}
corpus_cached["train"] = CorpusIterator_PTB("PTB", "train")
corpus_cached["dev"] = CorpusIterator_PTB("PTB", "dev")


def descendTree(tree, vocab, posFine, depsVocab):
   label = tree.label()
   for child in tree:
      if type(child) == nltk.tree.Tree:
   #     print((label, child.label()), type(tree))
        key = (label, child.label())
        depsVocab.add(key)
        descendTree(child, vocab, posFine, depsVocab)
      else:
        posFine.add(label)
        word = child.lower()
        if "*-" in word:
           continue
        vocab[word] = vocab.get(word, 0) + 1
    #    print(child)
def initializeOrderTable():
   orderTable = {}
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentenceAndTree in corpus_cached[partition].iterator():
      _, sentence = sentenceAndTree
      #descendTree(sentence, vocab, posFine, depsVocab)

      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          posFine.add(line["posFine"])
          depsVocab.add(line["dep"])
   return vocab, depsVocab

#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()




def orderSentenceRec(tree, sentence, printThings, linearized):
   label = tree.label()
   if label[-1] in "1234567890":
        label = label[:label.rfind("-")]
   children = [child for child in tree]
   if type(children[0]) != nltk.tree.Tree:
      assert all([type(x) != nltk.tree.Tree for x in children])
      assert len(list(children)) == 1, list(children)
      for c in children:
        if label in ["'", ":", "``", ",", "''", "#", ".", "-NONE-"] or label[0] == "-" or "*-" in c:
           continue
        word = sentence[tree.start]["word"] #c.lower(), )
        if word != c.lower().replace("\/","/"):
           print(142, word, c.lower())
        return {"word" : word, "category" : label, "children" : None, "dependency" : "NONE"}
   else:
      assert all([type(x) == nltk.tree.Tree for x in children])
      children = [child for child in children if child.start < child.end] # remove children that consist of gaps or otherwise eliminated tokens

      # find which children seem to be dependents of which other children
      if model != "REAL_REAL": 
        childDeps = [None for _ in children]
        for i in range(len(children)):
           incomingFromOutside = [x for x in tree.incoming if x in children[i].incoming]
           if len(incomingFromOutside) > 0:
              childDeps[i] = sentence[incomingFromOutside[-1][1]]["dep"]
              if len(incomingFromOutside) > 1:
                  print("FROM OUTSIDE", [sentence[incomingFromOutside[x][1]]["dep"] for x in range(len(incomingFromOutside))])
           for j in range(len(children)):
              if i == j:
                 continue
              incomingFromJ = [x for x in children[i].incoming if x in children[j].outgoing]
              if len(incomingFromJ) > 0:
                 if len(incomingFromJ) > 1:
                    duplicateDeps = tuple([sentence[incomingFromJ[x][1]]["dep"] for x in range(len(incomingFromJ))])
                    if not (duplicateDeps == ("obj", "xcomp")):
                       print("INCOMING FROM NEIGHBOR", duplicateDeps)
                 childDeps[i] = sentence[incomingFromJ[-1][1]]["dep"]
        assert None not in childDeps, (childDeps, children)
  
        keys = childDeps
  
        logits = [(x, distanceWeights[stoi_deps[key]], key) for x, key in zip(children, keys)]
        logits = sorted(logits, key=lambda x:-x[1])
        childrenLinearized = map(lambda x:x[0], logits)
      else:
        childrenLinearized = children
#      print(logits)
   
      childrenAsTrees = []
      for child, _, dependency in logits:
          childrenAsTrees.append(orderSentenceRec(child, sentence, printThings, linearized))
          if childrenAsTrees[-1] is None: # this will happen for punctuation etc 
              del childrenAsTrees[-1]
          else:
             childrenAsTrees[-1]["dependency"] = dependency
      if len(childrenAsTrees) == 0:
         return None
      else:
         return {"category" : label, "children" : childrenAsTrees, "dependency" : "NONE"}

def numberSpans(tree, start, sentence):
   if type(tree) != nltk.tree.Tree:
      if tree.startswith("*") or tree == "0":
        return start, ([]), ([])
      else:
        outgoing = ([(start, x) for x in sentence[start].get("children", [])])
        return start+1, ([(sentence[start]["head"]-1, start)]), outgoing
   else:
      tree.start = start
      incoming = ([])
      outgoing = ([])
      for child in tree:
        start, incomingC, outgoingC = numberSpans(child, start, sentence)
        incoming +=  incomingC
        outgoing += outgoingC
      tree.end = start
      incoming = ([(hd,dep) for hd, dep in incoming if hd < tree.start or hd>= tree.end])
      outgoing = ([(hd,dep) for hd, dep in outgoing if dep < tree.start or dep>= tree.end])

      tree.incoming = incoming
      tree.outgoing = outgoing
      return start, incoming, outgoing

import copy

def binarize(tree):
   # tree is a single node, i.e. a dict
   if tree["children"] is None:
       return tree
   else:
#       print(tree)
       if len(tree["children"]) == 0:
           assert False
       elif len(tree["children"]) <= 1: # remove unary projections
#          print("Removing Unary: "+tree["category"])
          result = binarize(tree["children"][0]) #{"category" : tree["category"], "dependency" : tree["dependency"], "children" : children}
          result["category"] = tree["category"]
          return result
       else:
          children = [binarize(x) for x in tree["children"]]
          return {"category" : tree["category"], "children" : children, "dependency" : tree["dependency"]}

def orderSentence(tree, printThings):
   global model
   linearized = []
   tree, sentence = tree
   for i in range(len(sentence)):
      line = sentence[i]
      if line["dep"] == "root":
         continue
      head = line["head"] - 1
      if "children" not in sentence[head]:
        sentence[head]["children"] = []
      sentence[head]["children"].append(i)
   end, incoming, outgoing = numberSpans(tree, 0, sentence)
   assert len(incoming) == 1, incoming
   assert len(outgoing) == 0, outgoing
   if (end != len(sentence)):
      print(tree.leaves())
      print([x["word"] for x in sentence])
   return binarize(orderSentenceRec(tree, sentence, printThings, linearized))

vocab, depsVocab = initializeOrderTable()


posFine = list(posFine)
itos_pos_fine = posFine
stoi_pos_fine = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = itos_pure_deps
stoi_deps = stoi_pure_deps

#print itos_deps

#dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)


import os

if model == "RANDOM_MODEL":
  for key in range(len(itos_deps)):
     #dhWeights[key] = random() - 0.5
     distanceWeights[key] = random()
  originalCounter = "NA"
elif model == "REAL" or model == "REAL_REAL":
  originalCounter = "NA"
elif model == "RANDOM_BY_TYPE":
  #dhByType = {}
  distByType = {}
  for dep in itos_pure_deps:
 #   dhByType[dep] = random() - 0.5
    distByType[dep] = random()
  for key in range(len(itos_deps)):
#     dhWeights[key] = dhByType[itos_deps[key]]
     distanceWeights[key] = distByType[itos_deps[key]]
  originalCounter = "NA"

lemmas = list(vocab_lemmas.iteritems())
lemmas = sorted(lemmas, key = lambda x:x[1], reverse=True)

words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))

if len(itos) > 6:
   assert stoi[itos[5]] == 5


vocab_size = 10000
vocab_size = min(len(itos),vocab_size)

print posFine
print morphKeyValuePairs
print itos[:vocab_size]
print "VOCABULARY "+str(len(posFine)+vocab_size+3)
outVocabSize = len(posFine)+vocab_size+3


itos_total = ["EOS", "EOW", "SOS"] + itos_pos_fine + itos[:vocab_size]
assert len(itos_total) == outVocabSize




initrange = 0.1
crossEntropy = 10.0

import torch.nn.functional



counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 


lossModule = nn.NLLLoss()
lossModuleTest = nn.NLLLoss(size_average=False, reduce=False, ignore_index=2)

corpusBase = corpus_cached["train"]
corpus = corpusBase.iterator()



# get the initial grammar

# perform splits on the grammar

# run EM

unary_rules = {}

binary_rules = {}

terminals = {}

def addCounts(tree):
   if tree["children"] is None:
      nonterminal = tree["category"]#+"@"+tree["dependency"]       
      terminal = tree["word"]
      if nonterminal not in terminals:
        terminals[nonterminal] = {}
      if terminal not in terminals[nonterminal]:
          terminals[nonterminal][terminal] = 0
      terminals[nonterminal][terminal] += 1
   else:
      for child in tree["children"]:
         addCounts(child)
      if len(tree["children"]) == 1:
         assert False
         nonterminal = tree["category"]#+"@"+tree["dependency"]       
         child = tree["children"][0]
         nonterminalChild = child["category"]#+"@"+child["dependency"]       
         if nonterminal not in unary_rules:
             unary_rules[nonterminal] = {}
         if nonterminalChild not in unary_rules[nonterminal]:
            unary_rules[nonterminal][nonterminalChild] = 0
         unary_rules[nonterminal][nonterminalChild] += 1
      elif len(tree["children"]) == 2:
         nonterminal = tree["category"]#+"@"+tree["dependency"]       
         childCats = tuple([x["category"] for x in tree["children"]])
#         left  = tree["children"][0]
 #        right = tree["children"][1]
   
  #       nonterminalLeft  = left["category"]#+"@"+left["dependency"]       
   #      nonterminalRight = right["category"]#+"@"+right["dependency"]       
         if nonterminal not in binary_rules:
              binary_rules[nonterminal] = {}
         if childCats not in binary_rules[nonterminal]:
            binary_rules[nonterminal][childCats] = 0
         binary_rules[nonterminal][childCats] += 1


roots = {}


inStackDistribution = {() : 0}

# stack = incomplete constituents that have been started
def updateInStackDistribution(tree, stack):
   if tree["children"] is None:
      return
   else:
     leftSide = tree["category"]
     rightSide = tuple([x["category"] for x in tree["children"]])
     for i, child in enumerate(tree["children"]):
        stack2 = stack + ((leftSide, rightSide[i:])     ,)
        updateInStackDistribution(tree["children"][0], stack2)
        stack3 = stack + ((leftSide, rightSide[i+1:])     ,)
        if i+1 < len(tree["children"]):
           inStackDistribution[stack3] = inStackDistribution.get(stack3, 0) + 1
   #  updateInStackDistribution(tree["children"][1], stack + (tree["category"],))

sentCount = 0
for sentence in corpus:
   sentCount += 1
   ordered = orderSentence(sentence,  sentCount % 50 == 0)
#   print(ordered)
   addCounts(ordered)
   roots[ordered["category"]] = roots.get(ordered["category"], 0) + 1
   print(sentCount, ordered["category"])
   # update inStackDistribution
   inStackDistribution[()] += 1
   updateInStackDistribution(ordered, ())
 #  break

print(list(binary_rules))
print(unary_rules)
#print(terminals)
print(len(binary_rules))
print(sorted(list(binary_rules["S"].items()), key=lambda x:x[1]))
#print(sorted(list(binary_rules["S_BAR"].items()), key=lambda x:x[1]))
print(roots)





# construct count matrices

# construct grammar

# create split

# run EM

# merge symbols

inStackDistribution = sorted(list(inStackDistribution.items()), key=lambda x:x[1])
print(inStackDistribution)
print(len(inStackDistribution))


corpusBase = corpus_cached["dev"]
corpus = corpusBase.iterator()

def linearizeTree2String(tree, sent):
   if tree["children"] is None:
       sent.append(tree["word"])
   else:
      for x in tree["children"]:
          linearizeTree2String(x, sent)

sentCount = 0
for sentence in corpus:
   sentCount += 1
   ordered = orderSentence(sentence,  sentCount % 50 == 0)

   print(ordered)
   linearized = []
   linearizeTree2String(ordered, linearized)
   print(linearized)


   for start in range(10):
      consumed = []
      beam = [(x[0], -log(x[1])) for x in inStackDistribution[-100:]]
      for length in range(5):
         # for each 
         consumed.append(linearized[start+length])
         print(beam) # the beam is a stack of partially (at least 1) satisfied rule expansions
         # get the corresponding preterminals

         # 

         # now update the beam
      print(start, consumed)
   break



