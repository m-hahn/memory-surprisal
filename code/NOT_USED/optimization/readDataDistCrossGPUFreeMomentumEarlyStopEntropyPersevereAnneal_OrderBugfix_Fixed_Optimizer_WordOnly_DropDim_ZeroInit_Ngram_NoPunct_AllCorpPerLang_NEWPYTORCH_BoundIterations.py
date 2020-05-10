# Based on readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_Optimizer_WordOnly_DropDim_ZeroInit_Ngram_NoPunct_AllCorpPerLang_NEWPYTORCH_BoundIterations.py


# TODO also try other optimizers

import random
import sys

objectiveName = "LM"

language = sys.argv[1]
languageCode = sys.argv[2]
entropy_weight = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
lr_policy = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01 # optimizer vs original: have to multiply by lr_lm, therefore here 0.01 instead of 0.1
#entropy_weight  = 0.0001 #00000001
momentum = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9 #float(sys.argv[5]) if len(sys.argv) > 5 else 0.9
optimizer_name = sys.argv[6] if len(sys.argv) > 6 else "SGD"

#dropout_rate = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
#emb_dim = int(sys.argv[7]) if len(sys.argv) > 7 else 50
#rnn_dim = int(sys.argv[9]) if len(sys.argv) > 9 else 128
#rnn_layers = int(sys.argv[10]) if len(sys.argv) > 10 else 2
assert len(sys.argv) == 7

myID = random.randint(0,10000000)


__file__ = __file__.split("/")[-1]

#with open("/juicier/scr120/scr/mhahn/deps/LOG"+language+"_"+__file__+"_model_"+str(myID)+".txt", "w") as outFile:
with open("/juicier/scr120/scr/mhahn/deps/LOG"+language+"_"+__file__+"_model_"+str(myID)+".txt", "w") as outFile:
    print >> outFile, " ".join(sys.argv)



posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]

deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp
from random import random, shuffle
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
     for sentence in CorpusIterator(language,partition).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
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
#           softmax = torch.distributions.Categorical(logits=logits)
#           selected = softmax.sample()
#           print selected
#           quit()
#           softmax = torch.cat(logits)



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
   

itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

print itos_deps

dhWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
distanceWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
for i, key in enumerate(itos_deps):

   # take from treebank, or randomize
   dhLogits[key] = 0.0 #2*(random()-0.5)
   dhWeights.data[i] = dhLogits[key]

   originalDistanceWeights[key] = 0.0 #random()  
   distanceWeights.data[i] = originalDistanceWeights[key]

from torch import optim
if optimizer_name == "SGD":
  policy_optimizer = optim.SGD([dhWeights, distanceWeights], lr=lr_policy, momentum=momentum)
elif optimizer_name == "Adagrad":
  policy_optimizer = optim.Adagrad([dhWeights, distanceWeights], lr=lr_policy)
elif optimizer_name == "Adam":
  policy_optimizer = optim.Adam([dhWeights, distanceWeights], lr=lr_policy)
else:
  assert False

words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))
#print stoi
#print itos[5]
#print stoi[itos[5]]

assert stoi[itos[5]] == 5

#print dhLogits

#for sentence in getNextSentence():
#   print orderSentence(sentence, dhLogits)

vocab_size = 50000


# 0 EOS, 1 UNK, 2 BOS
#word_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = emb_dim).cuda()
#pos_u_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = 10).cuda()
#pos_p_embeddings = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=10).cuda()
#
#
#baseline = nn.Linear(emb_dim, 1).cuda()
#
#dropout = nn.Dropout(dropout_rate).cuda()
#
#rnn = nn.LSTM(emb_dim+20, rnn_dim, rnn_layers).cuda()
#for name, param in rnn.named_parameters():
#  if 'bias' in name:
#     nn.init.constant(param, 0.0)
#  elif 'weight' in name:
#     nn.init.xavier_normal(param)
#
#decoder = nn.Linear(rnn_dim,vocab_size+3).cuda()
#pos_ptb_decoder = nn.Linear(rnn_dim,len(posFine)+3).cuda()
#
#components = [word_embeddings, pos_u_embeddings, pos_p_embeddings, rnn, decoder, pos_ptb_decoder, baseline]
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
#word_embeddings.weight.data.uniform_(-initrange, initrange)
#pos_u_embeddings.weight.data.uniform_(-initrange, initrange)
#pos_p_embeddings.weight.data.uniform_(-initrange, initrange)
#decoder.bias.data.fill_(0)
#decoder.weight.data.uniform_(-initrange, initrange)
#pos_ptb_decoder.bias.data.fill_(0)
#pos_ptb_decoder.weight.data.uniform_(-initrange, initrange)
#baseline.bias.data.fill_(0)
#baseline.weight.data.uniform_(-initrange, initrange)

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
       #for c in components:
       #   c.zero_grad()

       momentum = 0.0
       assert momentum == 0.0 # here taking offline estimate, so no momentum
       for p in  [dhWeights, distanceWeights]:
          if p.grad is not None:
             p.grad.data = p.grad.data.mul(momentum)


       totalQuality = 0.0

# (Variable(weight.new(2, bsz, self.nhid).zero_()),Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
#       for i in range(maxLength+1):
       if True:
           # TODO word dropout could also be added: randomly sprinkle `1' (UNK) in the LongTensor (not into input_words -- this will also be used for the softmax!)
#           words_layer = word_embeddings(Variable(torch.LongTensor(input_words)).cuda())
#           #pos_u_layer = pos_u_embeddings(Variable(torch.LongTensor(input_pos_u)).cuda())
#           #pos_p_layer = pos_p_embeddings(Variable(torch.LongTensor(input_pos_p)).cuda())
#           inputEmbeddings = dropout(words_layer) #torch.cat([words_layer, pos_u_layer, pos_p_layer], dim=2))
##           print hidden
#           #output, hidden = rnn(inputEmbeddings, hidden)
##           print maxLength
#           
##           print inputEmbeddings
##           print output
#           baseline_predictions = baseline(words_layer)

           # word logits
           #word_logits = decoder(dropout(output))
           #word_logits = word_logits.view(-1, vocab_size+3)
           #word_softmax = logsoftmax(word_logits)
           #word_softmax = word_softmax.view(-1, batchSize, vocab_size+3)

           # pos logits
#           pos_logits = pos_ptb_decoder(dropout(output))
#           pos_logits = pos_logits.view(-1, len(posFine)+3)
#           pos_softmax = logsoftmax(pos_logits)
 #          pos_softmax = pos_softmax.view(-1, batchSize, len(posFine)+3)

        
 
#           print word_logits
#           print predictions
           lossesWord = [[None]*batchSize for i in range(maxLength+1)]
#           lossesPOS = [[None]*batchSize for i in range(maxLength+1)]

#           print word_logits
#           for i in range(maxLength+1):
           for i in range(1,len(input_words)-1): #range(1,maxLength+1): # don't include i==0
              for j in range(batchSize):
#                 if input_words[i+1][j] != 0:
                 if i+1 < len(input_words) and input_words[i][j] != 0:
#                    print word_logits
#                    predictions = logsoftmax(word_logits[i][j]) # TODO PyTorch version doesn't support dimension-wise logsoftmax
                    left = input_words[i][j]
                    right = input_words[i+1][j]

                    if left in lm_counts and right in lm_counts:
                        delta = 0.5
                        prob = max(lm_counts[left].get(right,0)-delta, 0) / lm_counts[left]["_TOTAL_"] + ((lm_counts[right]["_TOTAL_"] + 0.0) / lm_counts["_TOTAL_"]) * delta * (len(lm_counts[left]) - 1.0) / lm_counts[left]["_TOTAL_"]
#                        print (lm_counts[left].get(right,0), lm_counts[left]["_TOTAL_"], lm_counts[right]["_TOTAL_"], (len(lm_counts[left]) - 1.0))
                        assert prob <= 1.0, prob
                        lossesWord[i][j] = - log(prob)
#                        baselineLoss += torch.nn.functional.mse_loss(baseline_predictions[i+1][j], Variable(torch.cuda.FloatTensor([lossesWord[i][j]]) ))

                        loss += lossesWord[i][j] # + lossesPOS[i][j]
                        lossWords += lossesWord[i][j]
#                        print (len(batchOrdered[j]), len(input_words), (i if i < len(batchOrdered[j]) else i-1))
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
#           losses = loss(predictions, input_words[i+1]) 
#           print losses
#    for i, sentence in enumerate(batchOrderLogits):
#       embeddingsLayer
         print lossWords/wordNum
         print ["CROSS ENTROPY", crossEntropy, exp(crossEntropy)]
       crossEntropy = 0.99 * crossEntropy + 0.01 * (lossWords/wordNum) #.data.cpu().numpy()[0]
       totalQuality = loss #.data.cpu().numpy()[0] # consists of lossesWord + lossesPOS
       numberOfWords = wordNum
       probabilities = torch.sigmoid(dhWeights)
#       print ["MEAN PROBABILITIES", torch.mean(probabilities)]
       #print ["PG", policyGradientLoss]

       neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))

       policy_related_loss = entropy_weight * neg_entropy + policyGradientLoss # lives on CPU
       return loss, baselineLoss, policy_related_loss, totalQuality, numberOfWords


def  doBackwardPass(loss, baselineLoss, policy_related_loss):
       global lastDevLoss
       global failedDevRuns
       if printHere:
         print "BACKWARD 1"
       policy_related_loss.backward()
       if printHere:
         print "BACKWARD 2"

#       loss += entropy_weight * neg_entropy
#       loss += lr_policy * policyGradientLoss

#       loss += baselineLoss # lives on GPU
#       try:
#         loss.backward()
#         torch.nn.utils.clip_grad_norm(parameters(), 5.0, norm_type='inf')
#         lm_optimizer.step()
#       except AttributeError:
#         print "Apparently no gradients for baseline"

       if printHere:
         print "BACKWARD 3 "+__file__+" "+language+" "+str(myID)+" "+str(counter)+" "+str(lastDevLoss)+" "+str(failedDevRuns)+"  "+(" ".join(map(str,["ENTROPY", entropy_weight, "LR_POLICY", lr_policy, "MOMENTUM", momentum, policy_optimizer ])))
         print devLosses
       torch.nn.utils.clip_grad_norm([dhWeights, distanceWeights], 5.0, norm_type='inf')

#       torch.nn.utils.clip_grad_norm(parameters(), 5.0, norm_type='inf')
#       for param in parameters():
#         #print "UPDATING"
#         if param.grad is None:
#           print "WARNING: None gradient"
#           continue
#         param.data.sub_(lr_lm * param.grad.data)
       policy_optimizer.step()

def computeDevLoss():
   global printHere
   global counter
   devLoss = 0.0
   devWords = 0
#   corpusDev = getNextSentence("dev")
   corpusDev = CorpusIterator(language,"dev").iterator(rejectShortSentences = True)

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
  corpus = CorpusIterator(language).iterator(rejectShortSentences = True)


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
          save_path = "/juicier/scr120/scr/mhahn/deps/"
          #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
          with open(save_path+"/manual_output/"+language+"_"+__file__+"_model_"+str(myID)+".tsv", "w") as outFile:
             print >> outFile, "\t".join(map(str,["FileName","ModelName","Counter", "AverageLoss","Head","DH_Weight","Dependency","Dependent","DistanceWeight", "EntropyWeight", "ObjectiveName"]))
             for i in range(len(itos_deps)):
                key = itos_deps[i]
                dhWeight = dhWeights[i].data.numpy()
                distanceWeight = distanceWeights[i].data.numpy()
                head, dependency, dependent = key
                print >> outFile, "\t".join(map(str,[myID, __file__, counter, crossEntropy, head, dhWeight, dependency, dependent, distanceWeight, entropy_weight, objectiveName]))
       if counter >= 1000000:
          print "Stop after "+str(1000000)+" iterations"
          quit()

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
#
#
#
