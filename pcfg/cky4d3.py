#Like cky3.py, but computes prefix AND suffix probabilities
# TODO use tests to verify this one
# Attempt at vectorization of cky4c5.py

##############
# Other (approximate) option for infix probs: add a few `empty words' around the string, without any penalties for per-word production

import random
import sys

objectiveName = "LM"

model = "REAL_REAL" #sys.argv[8]


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
      for child in childrenLinearized:
          childrenAsTrees.append(orderSentenceRec(child, sentence, printThings, linearized))
          if childrenAsTrees[-1] is None: # this will happen for punctuation etc 
              del childrenAsTrees[-1]
          else:
             childrenAsTrees[-1]["dependency"] = "Something"
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
       if len(tree["children"]) <= 1: # remove unary projections
#          print("Removing Unary: "+tree["category"])
          result = binarize(tree["children"][0]) #{"category" : tree["category"], "dependency" : tree["dependency"], "children" : children}
          result["category"] = tree["category"]
          return result
       else:
          children = [binarize(x) for x in tree["children"]]
          left = children[0]
          for child in children[1:]:
             left = {"category" : tree["category"]+"_BAR", "children" : [left, child], "dependency" : tree["dependency"]}
          return left

#          return {"category" : tree["category"], "children" : children, "dependency" : tree["dependency"]}

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
  generateGrammar = random.Random(5)
  for dep in itos_pure_deps:
 #   dhByType[dep] = random() - 0.5
    distByType[dep] = generateGrammar.random()
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
wordCounts = {}

def addCounts(tree):
   assert tree["category"] in leftCornerCounts, tree["category"]
   if tree["children"] is None:
      nonterminal = tree["category"]#+"@"+tree["dependency"]       
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
         nonterminal = tree["category"]#+"@"+tree["dependency"]       
         assert len(tree["children"]) == 2
         childCats = tuple([x["category"] for x in tree["children"]])
         assert len(childCats) == 2
         if nonterminal not in binary_rules:
              binary_rules[nonterminal] = {}
         if childCats not in binary_rules[nonterminal]:
            binary_rules[nonterminal][childCats] = 0
         binary_rules[nonterminal][childCats] += 1


roots = {}


inStackDistribution = {}

leftCornerCounts = {}

def updateLeftCorner(nonterminal, preterminal):
    if nonterminal not in leftCornerCounts:
       leftCornerCounts[nonterminal] = {}
       leftCornerCounts[nonterminal]["__TOTAL__"] = 0
    if preterminal not in leftCornerCounts[nonterminal]:
       leftCornerCounts[nonterminal][preterminal] = 0
    leftCornerCounts[nonterminal][preterminal] += 1
    leftCornerCounts[nonterminal]["__TOTAL__"] += 1
  

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
for sentence in corpus:
   sentCount += 1
   ordered = orderSentence(sentence,  sentCount % 50 == 0)


   linearized = []
   linearizeTree2String(ordered, linearized)
#   if len(linearized) > 10:
 #     continue

#   print(ordered)
   roots[ordered["category"]] = roots.get(ordered["category"], 0) + 1
   roots["__TOTAL__"] = roots.get("__TOTAL__", 0) + 1

   if sentCount % 100 == 0:
      print(sentCount, ordered["category"])
   # update inStackDistribution
   leftCorner = updateInStackDistribution(ordered, (("root", (ordered["category"],),),))
   updateLeftCorner("root", leftCorner)
   addCounts(ordered)

   # Only first sentence
 #  if sentCount > 100:
  #   break

#binary_rules["root"] = {}
#for r in roots:
#   binary_rules["root"][(r,)] = roots[r]

 #  break

print(list(binary_rules))
print(unary_rules)
#print(terminals)
print(len(binary_rules))
#print(sorted(list(binary_rules["S"].items()), key=lambda x:x[1]))
#print(sorted(list(binary_rules["S_BAR"].items()), key=lambda x:x[1]))
print(roots)


print(leftCornerCounts)


# construct count matrices

# construct grammar

# create split

# run EM

# merge symbols

inStackDistribution = sorted(list(inStackDistribution.items()), key=lambda x:x[1])
#print(inStackDistribution)
print(len(inStackDistribution))
print(inStackDistribution[-1])
print(inStackDistribution[-200:])
#quit()


inStackDistributionSum = sum([x[1] for x in inStackDistribution])

nonAndPreterminals = {}

for preterminal in terminals:
    nonAndPreterminals[preterminal] = sum([y for x, y in terminals[preterminal].iteritems()])
    terminals[preterminal]["__TOTAL__"] = nonAndPreterminals[preterminal]



for nonterminal in binary_rules:
    if nonterminal not in nonAndPreterminals:
       nonAndPreterminals[nonterminal]=0
    nonAndPreterminals[nonterminal] += sum([y for x, y in binary_rules[nonterminal].iteritems()])



# construct the reachability heuristic

# all nonAndPreterminals
# all words


#assert "PRN" in leftCornerCounts, nonAndPreterminals["PRN"]
#assert "MD" in leftCornerCounts, nonAndPreterminals["PRN"]




# Future version: this is simply done with a neural net, not a cached distribution

corpusBase = corpus_cached["dev"]
corpus = corpusBase.iterator()



leftCornerCached = {}




def logSumExp(x,y):
   constant = torch.max(x,y)
   resultInner = torch.exp(x-constant) + torch.exp(y-constant)
   result = constant + torch.log(resultInner)
   nothingX = (x == float("-Inf"))
   nothingY = (y == float("-Inf"))
   result[nothingX] = y[nothingX]
   result[nothingY] = x[nothingY]
   return result


itos_setOfNonterminals = sorted(list(set(list(binary_rules) + list(terminals))))
stoi_setOfNonterminals = dict(zip(itos_setOfNonterminals, range(len(itos_setOfNonterminals))))
print(itos_setOfNonterminals)



matrixLeft = torch.FloatTensor([[0 for _ in itos_setOfNonterminals] for _ in itos_setOfNonterminals]) # traces the LEFT edge
matrixRight = torch.FloatTensor([[0 for _ in itos_setOfNonterminals] for _ in itos_setOfNonterminals]) # traces the RIGHT edge

# One thing to keep in mind is that a bit of probability mass is wasted, namely that of training words ending up OOV
OOV_THRESHOLD = 3
OOV_COUNT= 10
OTHER_WORDS_SMOOTHING = 0.1


for parent in binary_rules:
   for (left, right), ruleCount in binary_rules[parent].iteritems():
      ruleProb = exp(log(ruleCount) - log(nonAndPreterminals[parent]+ OOV_COUNT + OTHER_WORDS_SMOOTHING*len(wordCounts)))
      matrixLeft[stoi_setOfNonterminals[parent]][stoi_setOfNonterminals[left]] -= ruleProb
      matrixRight[stoi_setOfNonterminals[parent]][stoi_setOfNonterminals[right]] -= ruleProb
      assert ruleProb > 0, ruleCount

binary_rules_numeric = {}
for parent in binary_rules:
   parenti = stoi_setOfNonterminals[parent]
   binary_rules_numeric[parenti] = {}
#   print(len(binary_rules[parent]))
   for (left, right), ruleCount in binary_rules[parent].iteritems():
       lefti = stoi_setOfNonterminals[left]
       righti = stoi_setOfNonterminals[right]
       binary_rules_numeric[parenti][(lefti, righti)] = ruleCount
#print(len(binary_rules_numeric))
#quit()

nonAndPreterminals_numeric = {}
for nonterminal in nonAndPreterminals:
   nonAndPreterminals_numeric[stoi_setOfNonterminals[nonterminal]] = nonAndPreterminals[nonterminal]


for i in range(len(itos_setOfNonterminals)):
    matrixLeft[i][i] += 1
    matrixRight[i][i] += 1
print(matrixLeft)
print(matrixLeft.sum(dim=1))
invertedLeft = torch.inverse(matrixLeft).tolist()
invertedRight = torch.inverse(matrixRight).tolist()
#print(invertedLeft)
#print(invertedLeft.size())
#for i in range(len(itos_setOfNonterminals)):
#   invertedLeft[i][i] -= 1
#print(invertedLeft)
#print(invertedLeft.sum())
#quit()


# Smoothing method: Words occurring <3 times in the training set are declared OOV
# Assign fixed count of 10 to OOV for each preterminal

# Second: Each Non-OOV nonterminal also has small mass for any preterminal


def plus(x,y):
   if x is None or y is None:
      return None
   return x+y

MAX_BOUNDARY = 5
surprisalTableSums = [0 for _ in range(MAX_BOUNDARY)]
surprisalTableCounts = [0 for _ in range(MAX_BOUNDARY)]


LEFT_CONTEXT = 2

sentCount = 0
for sentence in corpus:
   sentCount += 1
   ordered = orderSentence(sentence,  sentCount % 50 == 0)

#   print(ordered)
   linearized0 = []
   linearizeTree2String(ordered, linearized0)
   BATCHSIZE = 2
   for END in [MAX_BOUNDARY]:
      linearized = linearized0[0:END]
      if len(linearized) < END:
         continue
      chart = [[[torch.FloatTensor([float("-Inf") for _ in range(BATCHSIZE)]) for _ in itos_setOfNonterminals] for _ in linearized] for _ in linearized]
     
      for length in range(1, len(linearized)+1): # the NUMBER of words spanned. start+length is the first word OUTSIDE the constituent
         for start in range(len(linearized)): # the index of the first word taking part in the thing
            if start+length-1 >= len(linearized):
               continue
            if length == 1: # TODO for words at the boundary, immediately add prefix and suffix counts
               if start < LEFT_CONTEXT:
                 for preterminal in terminals:
                    chart[start][start][stoi_setOfNonterminals[preterminal]].fill_(0)
               else:
                 if wordCounts.get(linearized[start],0) < OOV_THRESHOLD: # OOV
                    for preterminal in terminals:
                        chart[start][start][stoi_setOfNonterminals[preterminal]].fill_(log(OOV_COUNT) - log(nonAndPreterminals[preterminal]+OOV_COUNT + OTHER_WORDS_SMOOTHING*len(wordCounts)))
                 else:
                    for preterminal in terminals:
                        count = terminals[preterminal].get(linearized[start], 0) + OTHER_WORDS_SMOOTHING
                        chart[start][start][stoi_setOfNonterminals[preterminal]].fill_(log(count) - log(nonAndPreterminals[preterminal]+ OOV_COUNT + OTHER_WORDS_SMOOTHING*len(wordCounts)))
                 assert start == start+length-1
            else:
                for start2 in range(start+1, len(linearized)):
                  for nonterminal, rules in binary_rules_numeric.iteritems():
                    for rule in rules.iteritems():
                        assert len(rule[0]) == 2
   
                        (leftCat, rightCat), ruleCount = rule
                        left = chart[start][start2-1][leftCat]
                        right = chart[start2][start+length-1][rightCat]
   
                        ruleProb = log(ruleCount) - log(nonAndPreterminals_numeric[nonterminal]+ OOV_COUNT + OTHER_WORDS_SMOOTHING*len(wordCounts))
   
                        new = left + right + ruleProb
                        entry = chart[start][start+length-1][nonterminal]
                        chart[start][start+length-1][nonterminal] = logSumExp(new, entry)
                        
#                        assert new <= 0
 #                       assert entry <= 0
      #############################
      # Now consider different endpoints
      valuesPerBoundary = [0]
      for BOUNDARY in range(LEFT_CONTEXT+1, len(linearized)+1):
         chartFromStart = [[torch.FloatTensor([float("-Inf") for _ in range(BATCHSIZE)]) for _ in itos_setOfNonterminals] for _ in range(BOUNDARY)]
      
         for start in range(BOUNDARY): # the index of the first word taking part in the thing
            if start+1-1 >= BOUNDARY:
               continue
            if 1 == 1: # TODO for words at the boundary, immediately add prefix and suffix counts
                 assert start == start+1-1
                 if start == BOUNDARY-1:
                   for preterminal in terminals:
                      preterminalID = stoi_setOfNonterminals[preterminal]
                      for nonterminalID in range(len(itos_setOfNonterminals)):
                        if invertedLeft[nonterminalID][preterminalID] > 0:
                          chartFromStart[start][nonterminalID] = logSumExp(chartFromStart[start][nonterminalID], log(invertedLeft[nonterminalID][preterminalID]) + chart[start][start][preterminalID])
      
         for start in range(BOUNDARY)[::-1]: # now construct potential constituents that start at `start', but end outside of the portion
               for start2 in range(start+1, BOUNDARY):
                  for nonterminal, rules in binary_rules_numeric.iteritems():
                    newAccumulated = torch.FloatTensor([float("-Inf") for _ in range(BATCHSIZE)])
                    for rule in rules.iteritems():
                        assert len(rule[0]) == 2
      
                        (leftCat, rightCat), ruleCount = rule
                        left = chart[start][start2-1][leftCat]
                        right = chartFromStart[start2][rightCat]
      
                        ruleProb = log(ruleCount) - log(nonAndPreterminals_numeric[nonterminal]+ OOV_COUNT + OTHER_WORDS_SMOOTHING*len(wordCounts))
      
                        assert ruleProb <= 0, (ruleCount, nonAndPreterminals_numeric[nonterminal]+ OOV_COUNT + OTHER_WORDS_SMOOTHING*len(wordCounts))
                        new = left + right + ruleProb
                        newAccumulated = logSumExp(newAccumulated, new)
                    nonterminalID = nonterminal
                    for nonterminalUpper in xrange(len(itos_setOfNonterminals)):
                        if invertedLeft[nonterminalUpper][nonterminalID] > 0:
                          logFactorForEnvironments = log(invertedLeft[nonterminalUpper][nonterminalID])
                          entry = chartFromStart[start][nonterminalUpper]
                          chartFromStart[start][nonterminalUpper] = logSumExp(newAccumulated + logFactorForEnvironments, entry)
      
  
  
         for root in itos_setOfNonterminals:
             count = roots.get(root, 0)
             iroot = stoi_setOfNonterminals[root]
             if chartFromStart[0][iroot] is not None:
                if count == 0:
                   chartFromStart[0][iroot] = torch.FloatTensor([float("-Inf") for _ in range(BATCHSIZE)])
                else:
                  chartFromStart[0][iroot] += log(count) - log(roots["__TOTAL__"])
  

         prefixProb = log(sum([exp(float(x[0])) if x[0] is not None else 0 for x in chartFromStart[0]])) # log P(S|root) -- the full mass comprising all possible trees (including spurious ambiguities arising from the PCFG conversion)
#         print("Prefix surprisal", prefixProb/(len(linearized)))
   #      quit()

         surprisalTableSums[BOUNDARY-1] += prefixProb
         surprisalTableCounts[BOUNDARY-1] += 1
         valuesPerBoundary.append(prefixProb)
         print(BOUNDARY, prefixProb, linearized)
         assert prefixProb  < valuesPerBoundary[-2], "bug or numerical problem?"
      print(sentCount, [surprisalTableSums[0]/surprisalTableCounts[-1]] + [(surprisalTableSums[i+1]-surprisalTableSums[i])/surprisalTableCounts[-1] for i in range(END-1)]) 
  
