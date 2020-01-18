# 4: tries to get rid of leaking probability mass



# could marginalize out all the words that don't occur

# Based on cky4d5.py
# Fixing NA issue
# Based on Stat9

# Minibatched version of cky_gpu_Stat3.py
# Uses Python3

# Unclear what explains difference between this and Non-"Debug" vesion

import random
import sys

objectiveName = "LM"

language = sys.argv[1]
model = sys.argv[2] #"REAL_REAL" #sys.argv[8]
assert model in ["REAL", "RANDOM_BY_TYPE"]

posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]



from math import log, exp
from random import random, shuffle, randint


from corpusIterator_V import CorpusIterator_V

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}

import nltk.tree

#corpus_cached = {}
#corpus_cached["train"] = CorpusIterator_PTB("PTB", "train")
#corpus_cached["dev"] = CorpusIterator_PTB("PTB", "dev")


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
   keys = set()
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in CorpusIterator_V(language,partition, storeMorph=True).iterator():
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

   leftChildren = []
   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         leftChildren.append(recursivelyLinearize(sentence, child, result, allGradients))
   result.append(line)
   line["relevant_logprob_sum"] = allGradients
   rightChildren = []
   if "children_HD" in line:
      for child in line["children_HD"]:
         rightChildren.append(recursivelyLinearize(sentence, child, result, allGradients))
   
   head = line["word"]
   if vocab[head] < 1000 or random() < 0.2:
      head = "_"

 
   inner = {"word" : line["word"], "category" :line["posUni"]+"_"+head, "children" : None, "line" : line, "coarse_dep" : line["coarse_dep"]}
   while rightChildren:
      sibling = rightChildren.pop(0)
 #     print(sibling)
      inner = {"category" : line["posUni"]+"_"+head+"_"+sibling["coarse_dep"], "children" : [inner, sibling], "line" : line, "coarse_dep" : line["coarse_dep"]}
   while leftChildren:
      sibling = leftChildren.pop(-1)
#      print(sibling)
      inner = {"category" : line["posUni"]+"_"+head+"_"+sibling["coarse_dep"], "children" : [sibling, inner], "line" : line, "coarse_dep" : line["coarse_dep"]}
   return inner

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       global model
#       childrenLinearized = []
#       while len(remainingChildren) > 0:
       if model == "REAL":
          return remainingChildren
       logits = [(x, distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]]) for x in remainingChildren]
       logits = sorted(logits, key=lambda x:x[1], reverse=(not reverseSoftmax))
       childrenLinearized = list(map(lambda x:x[0], logits))
           
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

##def orderSentenceRec(tree, sentence, printThings, linearized):
##   label = tree.label()
##   if "-" in label:
##      label = label[:label.index("-")]
##   children = [child for child in tree]
##   if type(children[0]) != nltk.tree.Tree:
##      assert all([type(x) != nltk.tree.Tree for x in children])
##      assert len(list(children)) == 1, list(children)
##      for c in children:
##        if len(label) == 0 or label in ["'", ":", "``", ",", "''", "#", ".", "-NONE-"] or label[0] == "-" or "*-" in c:
##           continue
##        word = sentence[tree.start]["word"] #c.lower(), )
##        if word != c.lower().replace("\/","/"):
##           print(142, word, c.lower())
##        return {"word" : word, "category" : label, "children" : None, "dependency" : "NONE"}
##   else:
##      assert all([type(x) == nltk.tree.Tree for x in children])
##      children = [child for child in children if child.start < child.end] # remove children that consist of gaps or otherwise eliminated tokens
##
##      # find which children seem to be dependents of which other children
##      if model != "REAL_REAL": 
##        childDeps = [None for _ in children]
##        for i in range(len(children)):
##           incomingFromOutside = [x for x in tree.incoming if x in children[i].incoming]
##           if len(incomingFromOutside) > 0:
##              childDeps[i] = sentence[incomingFromOutside[-1][1]]["dep"]
##              if len(incomingFromOutside) > 1:
##                  print("FROM OUTSIDE", [sentence[incomingFromOutside[x][1]]["dep"] for x in range(len(incomingFromOutside))])
##           for j in range(len(children)):
##              if i == j:
##                 continue
##              incomingFromJ = [x for x in children[i].incoming if x in children[j].outgoing]
##              if len(incomingFromJ) > 0:
##                 if len(incomingFromJ) > 1:
##                    duplicateDeps = tuple([sentence[incomingFromJ[x][1]]["dep"] for x in range(len(incomingFromJ))])
##                    if not (duplicateDeps == ("obj", "xcomp")):
##                       print("INCOMING FROM NEIGHBOR", duplicateDeps)
##                 childDeps[i] = sentence[incomingFromJ[-1][1]]["dep"]
##        assert None not in childDeps, (childDeps, children)
##  
##        keys = childDeps
##  
##        logits = [(x, distanceWeights[stoi_deps[key]], key) for x, key in zip(children, keys)]
##        logits = sorted(logits, key=lambda x:-x[1])
##        childrenLinearized = list(map(lambda x:x[0], logits))
##      else:
##        childrenLinearized = children
###      print(logits)
##   
##      childrenAsTrees = []
##      for child in childrenLinearized:
##          childrenAsTrees.append(orderSentenceRec(child, sentence, printThings, linearized))
##          if childrenAsTrees[-1] is None: # this will happen for punctuation etc 
##              del childrenAsTrees[-1]
##          else:
##             childrenAsTrees[-1]["dependency"] = "Something"
##      if len(childrenAsTrees) == 0:
##         return None
##      else:
##         return {"category" : label, "children" : childrenAsTrees, "dependency" : "NONE"}
##
##
##import copy
##
##def binarize(tree):
##   # tree is a single node, i.e. a dict
##   if tree["children"] is None:
##       return tree
##   else:
###       print(tree)
##       if len(tree["children"]) == 0:
##           assert False
##       if len(tree["children"]) <= 1: # remove unary projections
###          print("Removing Unary: "+tree["category"])
##          result = binarize(tree["children"][0]) #{"category" : tree["category"], "dependency" : tree["dependency"], "children" : children}
###          result["category"] = tree["category"] # TODO not sure why this makes a difference (together with adding/removing _BAR)
##          return result
##       else:
##          children = [binarize(x) for x in tree["children"]]
##          left = children[0]
##          for child in children[1:]:
##             left = {"category" : tree["category"], "children" : [left, child], "dependency" : tree["dependency"]} # +"_BAR"
##          return left
##
###          return {"category" : tree["category"], "children" : children, "dependency" : tree["dependency"]}

def orderSentence(sentence, printThings):
   global model

   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   if model == "REAL_REAL":
      eliminated = []
   for line in sentence:
      if line["dep"] == "root":
          root = line["index"]
          continue
      if line["dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         if model == "REAL_REAL":
            eliminated.append(line)
         continue
      key = (sentence[line["head"]-1]["posUni"], line["dep"], line["posUni"])
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
#      probability = 1/(1 + torch.exp(-dhLogit))
      if model == "REAL":
         dhSampled = (line["head"] > line["index"]) #(random() < probability.data.numpy()[0])
      else:
         dhSampled = (dhLogit > 0) #(random() < probability.data.numpy())
#      logProbabilityGradient = (1 if dhSampled else -1) * (1-probability)
#      line["ordering_decision_gradient"] = logProbabilityGradient
      #line["ordering_decision_log_probability"] = torch.log(1/(1 + torch.exp(- (1 if dhSampled else -1) * dhLogit)))

      
     
      direction = "DH" if dhSampled else "HD"
#torch.exp(line["ordering_decision_log_probability"]).data.numpy()[0],
      if printThings: 
         print("\t".join(list(map(str,["ORD", line["index"], ("|".join(line["morph"])+"           ")[:10], ("->".join(list(key)) + "         ")[:22], line["head"], dhLogit, dhSampled, direction]))))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])
      #sentence[headIndex]["children_decisions_logprobs"] = (sentence[headIndex].get("children_decisions_logprobs", []) + [line["ordering_decision_log_probability"]])


   if model != "REAL_REAL":
      for line in sentence:
         if "children_DH" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
            line["children_DH"] = childrenLinearized
         if "children_HD" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
            line["children_HD"] = childrenLinearized
   elif model == "REAL_REAL":
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
   tree = recursivelyLinearize(sentence, root, linearized, 0)
   if model == "REAL_REAL":
      linearized = list(filter(lambda x:"removed" not in x, sentence))
   if printThings or len(linearized) == 0:
     print(" ".join(list(map(lambda x:x["word"], sentence))))
     print(" ".join(list(map(lambda x:x["word"], linearized))))
   return tree


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


dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)


import os

if model == "RANDOM_MODEL":
  assert False
  for key in range(len(itos_deps)):
     dhWeights[key] = random() - 0.5
     distanceWeights[key] = random()
  originalCounter = "NA"
elif model == "REAL" or model == "REAL_REAL":
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


lemmas = list(vocab_lemmas.items())
lemmas = sorted(lemmas, key = lambda x:x[1], reverse=True)
itos_lemmas = list(map(lambda x:x[0], lemmas))
stoi_lemmas = dict(zip(itos_lemmas, range(len(itos_lemmas))))

words = list(vocab.items())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = list(map(lambda x:x[0], words))
stoi = dict(list(zip(itos, range(len(itos)))))

if len(itos) > 6:
   assert stoi[itos[5]] == 5


vocab_size = 10000
vocab_size = min(len(itos),vocab_size)

outVocabSize = len(posFine)+vocab_size+3





initrange = 0.1
crossEntropy = 10.0

import torch.nn.functional



counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 


lossModule = nn.NLLLoss()
lossModuleTest = nn.NLLLoss(size_average=False, reduce=False, ignore_index=2)

#corpusBase = corpus_cached["train"]
#corpus = corpusBase.iterator()



# get the initial grammar

# perform splits on the grammar

# run EM

unary_rules = {}

binary_rules = {}

terminals = {}
terminalsTotal = {}

wordCounts = {}

def addCounts(tree):
   assert tree["category"] in leftCornerCounts, tree["category"]
   if tree["children"] is None:
      nonterminal = tree["category"] #+"@"+head
      terminal = tree["word"]
      if nonterminal not in terminals:
        terminals[nonterminal] = {}
      if terminal not in terminals[nonterminal]:
          terminals[nonterminal][terminal] = 0
      terminals[nonterminal][terminal] += 1
      wordCounts[terminal] = wordCounts.get(terminal,0)+1
   else:
      for child in tree["children"]:
         addCounts(child)
      if True:
         nonterminal = tree["category"] #+"@"+head #+"@"+tree["dependency"]       
         assert len(tree["children"]) == 2
         childCats = tuple([x["category"] for x in tree["children"]])
         assert len(childCats) == 2
         if nonterminal not in binary_rules:
              binary_rules[nonterminal] = {}
         if childCats not in binary_rules[nonterminal]:
            binary_rules[nonterminal][childCats] = 0
         binary_rules[nonterminal][childCats] += 1


roots = {}
rootsTotal = 0


inStackDistribution = {}

leftCornerCounts = {}
leftCornerCountsTotal = {}

def updateLeftCorner(nonterminal, preterminal):
    if nonterminal not in leftCornerCounts:
       leftCornerCounts[nonterminal] = {}
       leftCornerCountsTotal[nonterminal] = 0
    if preterminal not in leftCornerCounts[nonterminal]:
       leftCornerCounts[nonterminal][preterminal] = 0
    leftCornerCounts[nonterminal][preterminal] += 1
    leftCornerCountsTotal[nonterminal] += 1
  

# stack = incomplete constituents that have been started
def updateInStackDistribution(tree, stack):
   if tree["children"] is None:
      assert len(stack) > 0, tree
#      print(stack)
      assert len(stack[-1][1]) > 0, stack
      inStackDistribution[stack] = inStackDistribution.get(stack, 0) + 1
      updateLeftCorner(tree["category"], tree["category"])
      assert tree["category"] in leftCornerCounts
      return tree["category"]
   else:
     leftSide = tree["category"]
     rightSide = tuple([x["category"] for x in tree["children"]])
     leftCorner = None
     for i, child in enumerate(tree["children"]):
        lc = updateInStackDistribution(child, stack + ((leftSide, rightSide[i:])  ,))
        if leftCorner is None:
           leftCorner = lc
     updateLeftCorner(tree["category"], leftCorner)
     assert tree["category"] in leftCornerCounts
     return leftCorner


def linearizeTree2String(tree, sent):
   if tree["children"] is None:
       sent.append(tree["word"])
   else:
      for x in tree["children"]:
          linearizeTree2String(x, sent)


sentCount = 0

print("Collecting counts from training corpus")
for sentence in CorpusIterator_V(language,"train").iterator():
   sentCount += 1
   ordered = orderSentence(sentence,  sentCount % 50 == 0)


   linearized = []
   linearizeTree2String(ordered, linearized)
#   if len(linearized) > 10:
 #     continue

#   print(ordered)
   roots[ordered["category"]] = roots.get(ordered["category"], 0) + 1
   rootsTotal = rootsTotal + 1

   if sentCount % 100 == 0:
      print(sentCount, ordered["category"])
   # update inStackDistribution
   leftCorner = updateInStackDistribution(ordered, (("root", (ordered["category"],),),))
   updateLeftCorner("root", leftCorner)
   addCounts(ordered)


OOV_THRESHOLD_TRAINING = 2

print(terminals)
print(wordCounts)
assert "_OOV_" not in wordCounts
wordCounts["_OOV_"] = 0
for preterminal in terminals:
  assert "_OOV_" not in terminals[preterminal]
  terminals[preterminal]["_OOV_"] = 0
  for word in list(terminals[preterminal]):
    if wordCounts[word] < OOV_THRESHOLD_TRAINING:
      terminals[preterminal]["_OOV_"] += terminals[preterminal][word]
      wordCounts["_OOV_"] += terminals[preterminal][word]
      wordCounts[word] -= terminals[preterminal][word]
      del terminals[preterminal][word]

#quit()

# Nontermainl ROOT
binary_rules["_SENTENCES_"] = {("ROOT", "_SENTENCES_") : 100000}
terminals["_EOS_"] = {"_eos_" : 1000000}

assert "__TOTAL__" not in roots

binary_rules["ROOT"] = {(left, "_EOS_") : count for left, count in roots.items() if left != "__TOTAL__"}
wordCounts["_eos_"] = 1000000

#assert "_eos_" not in itos
#assert "_eos_" not in stoi

#itos.append("_eos_")
#stoi["_eos_"] = len(itos)-1

#print(itos)
#quit()

   # Only first sentence
 #  if sentCount > 100:
  #   break

#binary_rules["root"] = {}
#for r in roots:
#   binary_rules["root"][(r,)] = roots[r]

 #  break

#print(list(binary_rules))
#print(unary_rules)
#print(terminals)
#print(len(binary_rules))
#print(sorted(list(binary_rules["S"].items()), key=lambda x:x[1]))
#print(sorted(list(binary_rules["S_BAR"].items()), key=lambda x:x[1]))
#print(roots)


#print(leftCornerCounts)


# construct count matrices

# construct grammar

# create split

# run EM

# merge symbols

inStackDistribution = sorted(list(inStackDistribution.items()), key=lambda x:x[1])
#print(inStackDistribution)
#print(len(inStackDistribution))
#print(inStackDistribution[-1])
#print(inStackDistribution[-200:])
#quit()


inStackDistributionSum = sum([x[1] for x in inStackDistribution])

nonAndPreterminals = {}

for preterminal in terminals:
    nonAndPreterminals[preterminal] = sum([y for x, y in terminals[preterminal].items()])
    terminalsTotal[preterminal] = nonAndPreterminals[preterminal]



for nonterminal in binary_rules:
    if nonterminal not in nonAndPreterminals:
       nonAndPreterminals[nonterminal]=0
    nonAndPreterminals[nonterminal] += sum([y for x, y in binary_rules[nonterminal].items()])

print(sorted(list(terminals)))
print(sorted(list(binary_rules)))
#quit()


# construct the reachability heuristic

# all nonAndPreterminals
# all words


#assert "PRN" in leftCornerCounts, nonAndPreterminals["PRN"]
#assert "MD" in leftCornerCounts, nonAndPreterminals["PRN"]




# Future version: this is simply done with a neural net, not a cached distribution

#corpusBase = corpus_cached["dev"]
#corpus = corpusBase.iterator()



leftCornerCached = {}


def logSumExpList(x):
   x = torch.stack(x, dim=0)
   constant = torch.max(x)
   if constant == float("-inf"):
     return x[0]
   result = constant + torch.log(torch.exp(x-constant).sum(dim=0))
   return result

# Alternative (fast?) implementation of logSumExp
def logSumExp(x,y):
   constantX = torch.max(x)
   if constantX == float("-Inf"):
     return y
   constantY = torch.max(y)
   if constantY == float("-Inf"):
     return x
   resultInner = torch.exp(x-(constantX+constantY)) + torch.exp(y-(constantX+constantY))
   result = (constantX + constantY) + torch.log(resultInner)
   return result


itos_setOfNonterminals = sorted(list(set(list(binary_rules) + list(terminals))))
stoi_setOfNonterminals = dict(list(zip(itos_setOfNonterminals, range(len(itos_setOfNonterminals)))))
print(itos_setOfNonterminals)



matrixLeft = torch.cuda.FloatTensor([[0 for _ in itos_setOfNonterminals] for _ in itos_setOfNonterminals]) # traces the LEFT edge
matrixRight = torch.cuda.FloatTensor([[0 for _ in itos_setOfNonterminals] for _ in itos_setOfNonterminals]) # traces the RIGHT edge

OOV_THRESHOLD = 3
OOV_COUNT= 0
OTHER_WORDS_SMOOTHING = 0.0001

preterminalsSet = set(terminals) # terminals is a dict
print(preterminalsSet)
relevantWordCount = 1+len([x for x in wordCounts if wordCounts[x] >= OOV_THRESHOLD_TRAINING])

for parent in binary_rules:
   for (left, right), ruleCount in binary_rules[parent].items():
      smoothing = OOV_COUNT + OTHER_WORDS_SMOOTHING*relevantWordCount if parent in preterminalsSet else 0
      ruleProb = exp(log(ruleCount) - log(nonAndPreterminals[parent]+ smoothing))
      matrixLeft[stoi_setOfNonterminals[parent]][stoi_setOfNonterminals[left]] -= ruleProb
      matrixRight[stoi_setOfNonterminals[parent]][stoi_setOfNonterminals[right]] -= ruleProb
      assert ruleProb > 0, ruleCount



nonAndPreterminals_numeric = {}
for nonterminal in nonAndPreterminals:
   nonAndPreterminals_numeric[stoi_setOfNonterminals[nonterminal]] = nonAndPreterminals[nonterminal]



binary_rules_matrix = torch.cuda.FloatTensor([[[0 for _ in range(len(itos_setOfNonterminals))]  for _ in range(len(itos_setOfNonterminals))] for _ in range(len(itos_setOfNonterminals))])


binary_rules_numeric = {}
for parent in binary_rules:
   parenti = stoi_setOfNonterminals[parent]
   binary_rules_numeric[parenti] = {}
#   print(len(binary_rules[parent]))
   for (left, right), ruleCount in binary_rules[parent].items():
       lefti = stoi_setOfNonterminals[left]
       righti = stoi_setOfNonterminals[right]
       binary_rules_numeric[parenti][(lefti, righti)] = ruleCount
       smoothing = OOV_COUNT + OTHER_WORDS_SMOOTHING*relevantWordCount if parent in preterminalsSet else 0
       binary_rules_matrix[parenti][lefti][righti] = exp(log(ruleCount) - log(nonAndPreterminals_numeric[parenti] + smoothing))
   totalProbabilityMass = binary_rules_matrix[parenti].sum()
   print("BINARY RULES", parent, totalProbabilityMass)
   assert float(totalProbabilityMass) <= 1.01

#print(len(binary_rules_numeric))
#quit()

print("Constructing prefix matrix")
for i in range(len(itos_setOfNonterminals)):
    matrixLeft[i][i] += 1
    matrixRight[i][i] += 1
print(matrixLeft)
print(matrixLeft.sum(dim=1))
invertedLeft = torch.inverse(matrixLeft)
#log_invertedLeft = torch.log(invertedLeft)

#invertedRight = torch.inverse(matrixRight).tolist()


def plus(x,y):
   if x is None or y is None:
      return None
   return x+y

MAX_BOUNDARY = 10
# It seems greater MAX_BOUNDARY values result in NAs. Maybe have to stabilise by batch?
surprisalTableSums = [0 for _ in range(MAX_BOUNDARY)]
surprisalTableCounts = [0 for _ in range(MAX_BOUNDARY)]


LEFT_CONTEXT = 5

BATCHSIZE = 3000 #200

sentCount = 0
def iterator_dense(corpus):
  chunk = []
  global sentCount
  sentCount = 0
  for sentence in corpus:
     sentCount += 1
     ordered = orderSentence(sentence,  sentCount % 50 == 0)
     linearized0 = []
     linearizeTree2String(ordered, linearized0)
     chunk += linearized0 + ["_eos_"]
     while len(chunk) > MAX_BOUNDARY:
        yield chunk[:MAX_BOUNDARY]
        chunk = chunk[1:]


def runOnCorpus():
  global BATCHSIZE
  global chart
  chart = [[torch.cuda.FloatTensor([[float("-Inf") for _ in itos_setOfNonterminals] for _ in range(BATCHSIZE)]) for _ in range(MAX_BOUNDARY)] for _ in range(MAX_BOUNDARY)]

  iterator = iterator_dense(CorpusIterator_V(language,"dev").iterator())
  chunk = []
  surprisals = [0 for _ in range(MAX_BOUNDARY)]
  while True:
     linearized = []
     try:
       for _ in range(BATCHSIZE):
          linearized.append(next(iterator))
     except StopIteration:
       if len(linearized) == 0:
          break
       BATCHSIZE = len(linearized) 
       chart = [[torch.cuda.FloatTensor([[float("-Inf") for _ in itos_setOfNonterminals] for _ in range(BATCHSIZE)]) for _ in range(MAX_BOUNDARY)] for _ in range(MAX_BOUNDARY)]

     print(sentCount, [surprisals[i+1] - surprisals[i] for i in range(MAX_BOUNDARY-1)]) # [surprisalTableSums[0]/surprisalTableCounts[-1]] + [(surprisalTableSums[i+1]-surprisalTableSums[i])/surprisalTableCounts[-1] for i in range(MAX_BOUNDARY-1)]) 

     computeSurprisals(linearized)
     surprisals = [surprisalTableSums[i]/(surprisalTableCounts[i]+1e-9) for i in range(MAX_BOUNDARY)]
     print(sentCount, [surprisals[i+1] - surprisals[i] for i in range(MAX_BOUNDARY-1)]) # [surprisalTableSums[0]/surprisalTableCounts[-1]] + [(surprisalTableSums[i+1]-surprisalTableSums[i])/surprisalTableCounts[-1] for i in range(MAX_BOUNDARY-1)]) 



itos = set(["_OOV_", "_EMPTY_"])
assert "_OOV_" in itos

for preterminal in terminals:
  for word in terminals[preterminal]:
    itos.add(word)
assert "_OOV_" in itos
itos = sorted(list(itos))
assert "_OOV_" in itos

stoi = dict(list(zip(itos, list(range(len(itos))))))
assert "_OOV_" in itos

#print(stoi)
assert "_OOV_" in stoi
assert "_eos_" in stoi

print("Constructing lexical probabilities")
lexicalProbabilities_matrix = torch.FloatTensor([[float("-inf") for _ in itos] for _ in stoi_setOfNonterminals])



for preterminal in terminals:
#  lexicalProbabilities_matrix[stoi_setOfNonterminals[preterminal]][stoi["_OOV_"]] = (log(OOV_COUNT) - log(nonAndPreterminals[preterminal]+ OOV_COUNT + OTHER_WORDS_SMOOTHING*len(wordCounts)))
  lexicalProbabilities_matrix[stoi_setOfNonterminals[preterminal]][stoi["_EMPTY_"]] = 0 # this is intended for the context words

  for word in stoi:
    if word == "_EMPTY_":
         print(preterminal, word, terminals[preterminal].get(word, 0))
         continue
#    if word == "__TOTAL__":
 #     continue
  #  assert word in wordCounts, (word, terminals[preterminal].get(word, 0))
    count = terminals[preterminal].get(word, 0) + OTHER_WORDS_SMOOTHING
    lexicalProbabilities_matrix[stoi_setOfNonterminals[preterminal]][stoi[word]] = (log(count) - log(nonAndPreterminals[preterminal]+ OOV_COUNT + OTHER_WORDS_SMOOTHING*len(stoi)))
#  print(itos[:10])
 # print(lexicalProbabilities_matrix[stoi_setOfNonterminals[preterminal]].exp())
  print("TERMINAL EXPANSION", preterminal, lexicalProbabilities_matrix[stoi_setOfNonterminals[preterminal]].exp().sum())
#for nonterminal in binary_rules:
#  print(nonterminal, lexicalProbabilities_matrix[stoi_setOfNonterminals[nonterminal]])
print(len(stoi), len(wordCounts))
#quit()
#quit()

for i in range(len(lexicalProbabilities_matrix)):
   for j in range(len(lexicalProbabilities_matrix[i])):
       if float(lexicalProbabilities_matrix[i][j]) == float("-inf"):
         assert itos_setOfNonterminals[i] not in terminals, itos_setOfNonterminals[i]
lexicalProbabilities_matrix = lexicalProbabilities_matrix.cuda().t()
#print(lexicalProbabilities_matrix) # (nonterminals, words)
# TODO why are there some -inf's?

def computeSurprisals(linearized):
      assert len(linearized[0]) == MAX_BOUNDARY
      assert len(linearized) == BATCHSIZE

      # Presumably unnecessary
      for x in chart:     
          for y in x:
               y.fill_(float("-Inf"))

      for length in range(1, MAX_BOUNDARY+1): # the NUMBER of words spanned. start+length is the first word OUTSIDE the constituent
         for start in range(MAX_BOUNDARY): # the index of the first word taking part in the thing
            if start+length-1 >= MAX_BOUNDARY:
               continue
            if length == 1: 
               if start < LEFT_CONTEXT:
                 for preterminal in terminals:
                    chart[start][start][:,stoi_setOfNonterminals[preterminal]].fill_(0)
               else:
                 lexical_tensor = torch.LongTensor([0 for _ in range(BATCHSIZE)])
             
                 for batch in range(BATCHSIZE): 
                    if wordCounts.get(linearized[batch][start],0) < OOV_THRESHOLD: # OOV
                       lexical_tensor[batch] = stoi["_OOV_"]
                    else:
                       lexical_tensor[batch] = stoi[linearized[batch][start]]
                 lexical_tensor = lexical_tensor.cuda()
                 chart[start][start] = torch.nn.functional.embedding(input=lexical_tensor, weight=lexicalProbabilities_matrix)
                 assert start == start+length-1
            else:
                entries = []
                for start2 in range(start+1, MAX_BOUNDARY):
                  left = chart[start][start2-1]
                  right = chart[start2][start+length-1]
                  maxLeft = torch.max(left) #, dim=1, keepdim=True)[0]
                  maxRight = torch.max(right) #, dim=1, keepdim=True)[0]
                  if float(maxLeft) == float("-inf") or float(maxRight) == float("-inf"): # everything will be 0
                     continue
                  resultLeft = torch.tensordot(torch.exp(left-maxLeft), binary_rules_matrix, dims=([1], [1]))
                  resultTotal = torch.bmm(resultLeft, torch.exp(right-maxRight).view(BATCHSIZE, -1, 1)).squeeze(2)
                  resultTotal = torch.nn.functional.relu(resultTotal) # because some values end up being slightly negative in result
                  resultTotalLog = torch.log(resultTotal)+(maxLeft+maxRight)
                  entries.append(resultTotalLog)
                chart[start][start+length-1] = logSumExpList(entries)
      #############################
      # Now consider different endpoints
      valuesPerBoundary = [0]
      for BOUNDARY in range(1, MAX_BOUNDARY+1):
         chartFromStart = [torch.cuda.FloatTensor([[float("-Inf") for _ in itos_setOfNonterminals] for _ in range(BATCHSIZE)]) for _ in range(BOUNDARY)]

         if True:      
             right = chart[BOUNDARY-1][BOUNDARY-1]
             right_max = torch.max(right)
             result = torch.tensordot(torch.exp(right-right_max), invertedLeft, dims=([1], [1]))
             resultLog = (torch.log(result) + right_max)
             chartFromStart[BOUNDARY-1] = resultLog
      
         for start in range(BOUNDARY-1)[::-1]: # now construct potential constituents that start at `start', but end outside of the portion
               entries = []
               for start2 in range(start+1, BOUNDARY):
                  left = chart[start][start2-1]
                  right = chartFromStart[start2]
                  maxLeft = torch.max(left)
                  maxRight = torch.max(right)
                  if float(maxLeft) == float("-inf") or float(maxRight) == float("-inf"): # everything will be 0
                     continue
                  resultLeft = torch.tensordot(torch.exp(left-maxLeft), binary_rules_matrix, dims=([1], [1]))
                  resultTotal = torch.bmm(resultLeft, torch.exp(right-maxRight).view(BATCHSIZE, -1, 1)).squeeze(2)
                  result = torch.tensordot(resultTotal, invertedLeft, dims=([1], [1]))
                  result = torch.nn.functional.relu(result) # because some values end up being slightly negative in result
                  resultLog = (torch.log(result) + (maxLeft+maxRight))
                  entries.append(resultLog)
               chartFromStart[start] = logSumExpList(entries)
         prefixProb = float(chartFromStart[0][:,stoi_setOfNonterminals["_SENTENCES_"]].sum()) #log(sum([exp(float(x[0])) if x[0] is not None else 0 for x in chartFromStart[0]])) # log P(S|root) -- the full mass comprising all possible trees (including spurious ambiguities arising from the PCFG conversion)

         surprisalTableSums[BOUNDARY-1] += prefixProb
         surprisalTableCounts[BOUNDARY-1] += BATCHSIZE
         valuesPerBoundary.append(prefixProb)
         print(BOUNDARY, prefixProb/BATCHSIZE, linearized[0])
         assert prefixProb/BATCHSIZE - 0.01 < valuesPerBoundary[-2]/BATCHSIZE, ("bug or numerical problem?", (prefixProb/BATCHSIZE, valuesPerBoundary[-2]/BATCHSIZE))
print("Reading data")

runOnCorpus() 
