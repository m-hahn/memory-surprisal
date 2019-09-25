# Based on cky4d5.py
# Fixing NA issue
# Based on Stat9

# Minibatched version of cky_gpu_Stat3.py
# Uses Python3


import random
import sys

objectiveName = "LM"

model = sys.argv[1] #"REAL_REAL" #sys.argv[8]
assert model in ["REAL_REAL", "RANDOM_BY_TYPE"]

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
        childrenLinearized = list(map(lambda x:x[0], logits))
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
stoi_pos_fine = dict(list(zip(posFine, range(len(posFine)))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(list(zip(itos_pure_deps, range(len(itos_pure_deps)))))
   

itos_deps = itos_pure_deps
stoi_deps = stoi_pure_deps


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

lemmas = list(vocab_lemmas.items())
lemmas = sorted(lemmas, key = lambda x:x[1], reverse=True)

words = list(vocab.items())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = list(map(lambda x:x[0], words))
stoi = dict(list(zip(itos, range(len(itos)))))

if len(itos) > 6:
   assert stoi[itos[5]] == 5


vocab_size = 10000
vocab_size = min(len(itos),vocab_size)

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


# Nontermainl ROOT
binary_rules["_SENTENCES_"] = {("ROOT", "_SENTENCES_") : 100000}
terminals["_EOS_"] = {"_eos_" : 1000000}

binary_rules["ROOT"] = {(left, "_EOS_") : count for left, count in roots.items() if left != "__TOTAL__"}
wordCounts["_eos_"] = 1000000





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
    nonAndPreterminals[preterminal] = sum([y for x, y in terminals[preterminal].items()])
    terminals[preterminal]["__TOTAL__"] = nonAndPreterminals[preterminal]



for nonterminal in binary_rules:
    if nonterminal not in nonAndPreterminals:
       nonAndPreterminals[nonterminal]=0
    nonAndPreterminals[nonterminal] += sum([y for x, y in binary_rules[nonterminal].items()])



# construct the reachability heuristic

# all nonAndPreterminals
# all words


#assert "PRN" in leftCornerCounts, nonAndPreterminals["PRN"]
#assert "MD" in leftCornerCounts, nonAndPreterminals["PRN"]




# Future version: this is simply done with a neural net, not a cached distribution

corpusBase = corpus_cached["dev"]
corpus = corpusBase.iterator()



leftCornerCached = {}



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

# One thing to keep in mind is that a bit of probability mass is wasted, namely that of training words ending up OOV
OOV_THRESHOLD = 3
OOV_COUNT= 10
OTHER_WORDS_SMOOTHING = 0.1


for parent in binary_rules:
   for (left, right), ruleCount in binary_rules[parent].items():
      ruleProb = exp(log(ruleCount) - log(nonAndPreterminals[parent]+ OOV_COUNT + OTHER_WORDS_SMOOTHING*len(wordCounts)))
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
       binary_rules_matrix[parenti][lefti][righti] = exp(log(ruleCount) - log(nonAndPreterminals_numeric[parenti]+ OOV_COUNT + OTHER_WORDS_SMOOTHING*len(wordCounts)))


#print(len(binary_rules_numeric))
#quit()

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

MAX_BOUNDARY = 20
surprisalTableSums = [0 for _ in range(MAX_BOUNDARY)]
surprisalTableCounts = [0 for _ in range(MAX_BOUNDARY)]


LEFT_CONTEXT = 10

BATCHSIZE = 1000 #200

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
  iterator = iterator_dense(corpus)
  chunk = []
  while True:
     linearized = [next(iterator) for _ in range(BATCHSIZE)]
     computeSurprisals(linearized)
     surprisals = [surprisalTableSums[i]/(surprisalTableCounts[i]+1e-9) for i in range(MAX_BOUNDARY)]
     print(sentCount, [surprisals[i+1] - surprisals[i] for i in range(MAX_BOUNDARY-1)]) # [surprisalTableSums[0]/surprisalTableCounts[-1]] + [(surprisalTableSums[i+1]-surprisalTableSums[i])/surprisalTableCounts[-1] for i in range(MAX_BOUNDARY-1)]) 

chart = [[torch.cuda.FloatTensor([[float("-Inf") for _ in itos_setOfNonterminals] for _ in range(BATCHSIZE)]) for _ in range(MAX_BOUNDARY)] for _ in range(MAX_BOUNDARY)]


itos = set(["_OOV_", "_EMPTY_"])
assert "_OOV_" in itos

for preterminal in terminals:
  for word in terminals[preterminal]:
    itos.add(word)
assert "_OOV_" in itos
itos = list(itos)
assert "_OOV_" in itos

stoi = dict(list(zip(itos, list(range(len(itos))))))
assert "_OOV_" in itos

print(stoi)
assert "_OOV_" in stoi

lexicalProbabilities_matrix = torch.FloatTensor([[float("-inf") for _ in itos] for _ in stoi_setOfNonterminals])

for preterminal in terminals:
  lexicalProbabilities_matrix[stoi_setOfNonterminals[preterminal]][stoi["_OOV_"]] = (log(OOV_COUNT) - log(nonAndPreterminals[preterminal]+ OOV_COUNT + OTHER_WORDS_SMOOTHING*len(wordCounts)))
  lexicalProbabilities_matrix[stoi_setOfNonterminals[preterminal]][stoi["_EMPTY_"]] = 0 # this is intended for the context words

  for word in stoi:
    if word == "_OOV_" or word == "_EMPTY_":
         continue
    count = terminals[preterminal].get(word, 0) + OTHER_WORDS_SMOOTHING
    lexicalProbabilities_matrix[stoi_setOfNonterminals[preterminal]][stoi[word]] = (log(count) - log(nonAndPreterminals[preterminal]+ OOV_COUNT + OTHER_WORDS_SMOOTHING*len(wordCounts)))

lexicalProbabilities_matrix = lexicalProbabilities_matrix.cuda().t()
print(lexicalProbabilities_matrix) # (nonterminals, words)

def computeSurprisals(linearized):
      assert len(linearized[0]) == MAX_BOUNDARY
      assert len(linearized) == BATCHSIZE

      for x in chart:     
          for y in x:
               y.fill_(float("-Inf"))

      for length in range(1, MAX_BOUNDARY+1): # the NUMBER of words spanned. start+length is the first word OUTSIDE the constituent
         for start in range(MAX_BOUNDARY): # the index of the first word taking part in the thing
            if start+length-1 >= MAX_BOUNDARY:
               continue
            if length == 1: # TODO for words at the boundary, immediately add prefix and suffix counts
               if start < LEFT_CONTEXT:
                 for preterminal in terminals:
                    chart[start][start][:,stoi_setOfNonterminals[preterminal]].fill_(0)
               else:
                 lexical_tensor = torch.LongTensor([0 for _ in range(BATCHSIZE)])
             
                 for batch in range(BATCHSIZE): # TODO speed this up as a single matrix multiplication
                    if wordCounts.get(linearized[batch][start],0) < OOV_THRESHOLD: # OOV
                       lexical_tensor[batch] = stoi["_OOV_"]
                    else:
                       lexical_tensor[batch] = stoi[linearized[batch][start]]
                 lexical_tensor = lexical_tensor.cuda()
                 chart[start][start] = torch.nn.functional.embedding(input=lexical_tensor, weight=lexicalProbabilities_matrix)
                 assert start == start+length-1
            else:
                for start2 in range(start+1, MAX_BOUNDARY):
  #                print(chart[start][start2-1].size(), chart[start2][start+length-1].size(), binary_rules_matrix.size())
                  left = chart[start][start2-1]
                  right = chart[start2][start+length-1]
                  maxLeft = torch.max(left) #, dim=1, keepdim=True)[0]
                  maxRight = torch.max(right) #, dim=1, keepdim=True)[0]
                  if float(maxLeft) == float("-inf") or float(maxRight) == float("-inf"): # everything will be 0
                     continue
                  resultLeft = torch.tensordot(torch.exp(left-maxLeft), binary_rules_matrix, dims=([1], [1]))
                  resultTotal = torch.bmm(resultLeft, torch.exp(right-maxRight).view(BATCHSIZE, -1, 1)).squeeze(2)

                  if True:
                     resultTotal = torch.nn.functional.relu(resultTotal) # because some values end up being slightly negative in result

                  resultTotalLog = torch.log(resultTotal)+(maxLeft+maxRight)
                  if False:
                     resultTotalLog[resultTotal <= 0].fill_(float("-inf"))
#                  assert "nan" not in str(resultTotalLog.max())

                  entry = chart[start][start+length-1]
                  chart[start][start+length-1] = logSumExp(resultTotalLog, entry)
      #############################
      # Now consider different endpoints
      valuesPerBoundary = [0]
      for BOUNDARY in range(LEFT_CONTEXT+1, MAX_BOUNDARY+1):
         chartFromStart = [torch.cuda.FloatTensor([[float("-Inf") for _ in itos_setOfNonterminals] for _ in range(BATCHSIZE)]) for _ in range(BOUNDARY)]

         if True:      
             right = chart[BOUNDARY-1][BOUNDARY-1]
             right_max = torch.max(right)
             result = torch.tensordot(torch.exp(right-right_max), invertedLeft, dims=([1], [1]))
             resultLog = (torch.log(result) + right_max)
             chartFromStart[BOUNDARY-1] = resultLog
      
         for start in range(BOUNDARY)[::-1]: # now construct potential constituents that start at `start', but end outside of the portion
               for start2 in range(start+1, BOUNDARY):

                  left = chart[start][start2-1]
                  right = chartFromStart[start2]
                  maxLeft = torch.max(left)
                  maxRight = torch.max(right)
                  if float(maxLeft) == float("-inf") or float(maxRight) == float("-inf"): # everything will be 0
                     continue

                  resultLeft = torch.tensordot(torch.exp(left-maxLeft), binary_rules_matrix, dims=([1], [1]))
                  resultTotal = torch.bmm(resultLeft, torch.exp(right-maxRight).view(BATCHSIZE, -1, 1)).squeeze(2)

#                  resultLeft = torch.tensordot(torch.exp(left-maxLeft), binary_rules_matrix, dims=([0], [1]))
 #                 resultTotal = torch.tensordot(resultLeft, torch.exp(right-maxRight), dims=([1], [0]))
#                  resultTotalLog = torch.log(resultTotal)+maxLeft+maxRight
 #                 resultTotalLog[resultTotal <= 0].fill_(float("-inf"))

 #                 resultTotalLog_max = torch.max(resultTotalLog)

                  result = torch.tensordot(resultTotal, invertedLeft, dims=([1], [1]))

                  result = torch.nn.functional.relu(result) # because some values end up being slightly negative in result

                  resultLog = (torch.log(result) + (maxLeft+maxRight))
                  if False: # On the GPU, it seems log(0) = -inf
                     resultLog[result <= 0].fill_(float("-inf"))
                  chartFromStart[start] = logSumExp(chartFromStart[start], resultLog)
#         for root in itos_setOfNonterminals:
#             count = roots.get(root, 0)
#             iroot = stoi_setOfNonterminals[root]
#             if chartFromStart[0][iroot] is not None:
#                if count == 0:
#                   chartFromStart[0][iroot] = torch.cuda.FloatTensor([float("-Inf") for _ in range(BATCHSIZE)])
#                else:
#                  chartFromStart[0][iroot] += log(count) - log(roots["__TOTAL__"])
#  

         prefixProb = float(chartFromStart[0][:,stoi_setOfNonterminals["_SENTENCES_"]].sum()) #log(sum([exp(float(x[0])) if x[0] is not None else 0 for x in chartFromStart[0]])) # log P(S|root) -- the full mass comprising all possible trees (including spurious ambiguities arising from the PCFG conversion)

         surprisalTableSums[BOUNDARY-1] += prefixProb
         surprisalTableCounts[BOUNDARY-1] += BATCHSIZE
         valuesPerBoundary.append(prefixProb)
         print(BOUNDARY, prefixProb/BATCHSIZE, linearized[0])
         assert prefixProb  < valuesPerBoundary[-2], "bug or numerical problem?"
#         print(len(nonAndPreterminals_numeric))  
runOnCorpus() 
