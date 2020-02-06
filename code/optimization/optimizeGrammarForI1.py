# optimizeGrammarForI1.py
# yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast_SaveLast_NoFinePOS_OnlyWordForms_FullVocab.py
# readDataDistCrossGPUFreeAllTwoEqual_NoClip_ByCoarseOnly_FixObj_OnlyLangmod_Replication_Best.py


#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"



# TODO also try other optimizers

import random
import sys
import math

objectiveName = "LM"


import argparse

parser = argparse.ArgumentParser()

# 


parser.add_argument('--language', type=str, dest="language")
parser.add_argument('--entropy_weight', type=float, default=0.001, dest="entropy_weight")
parser.add_argument('--lr_policy', type=float, default=0.001, dest="lr_policy")
parser.add_argument('--momentum_policy', type=float, default=0.9, dest="momentum_policy")
parser.add_argument('--lr_baseline', type=float, default=1.0, dest="lr_baseline")
parser.add_argument('--dropout_prob', type=float, default=0.5, dest="dropout_prob")
parser.add_argument('--lr', type=float, default=0.1, dest="lr")
parser.add_argument('--batchSize', type=int, default=1, dest="batchSize")
parser.add_argument('--dropout_rate', type=float, default=1, dest="dropout_rate")
parser.add_argument('--emb_dim', type=int, default=10, dest="emb_dim")
parser.add_argument('--rnn_dim', type=int, default=5, dest="rnn_dim")
parser.add_argument('--rnn_layers', type=int, default=1, dest="rnn_layers")
parser.add_argument('--input_dropoutRate', type=float, default=0.0, dest="input_dropoutRate")
parser.add_argument('--replaceWordsProbability', type=float, default=0.0, dest="replaceWordsProbability")
parser.add_argument('--prescribedID', type=int, default=random.randint(0,10000000000), dest="prescribedID")


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
       while len(remainingChildren) > 0:
           logits = torch.cat([distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]].view(1) for x in remainingChildren])
           #print logits
#           if reverseSoftmax:
 #             logits = -logits
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
      dhSampled = (random() < probability.data.numpy())
#      logProbabilityGradient = (1 if dhSampled else -1) * (1-probability)
#      line["ordering_decision_gradient"] = logProbabilityGradient
      line["ordering_decision_log_probability"] = torch.log(1/(1 + torch.exp(- (1 if dhSampled else -1) * dhLogit)))

      
     
      direction = "DH" if dhSampled else "HD"
#torch.exp(line["ordering_decision_log_probability"]).data.numpy(),
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, (str(probability.data.numpy())+"      ")[:8], str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]].data.numpy())+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])
      sentence[headIndex]["children_decisions_logprobs"] = (sentence[headIndex].get("children_decisions_logprobs", []) + [line["ordering_decision_log_probability"]])



   for line in sentence:
      if "children_DH" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:])
         line["children_DH"] = childrenLinearized
      if "children_HD" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:])[::-1]
         line["children_HD"] = childrenLinearized

#         shuffle(line["children_HD"])
   
   linearized = []
   logprob_sum = recursivelyLinearize(sentence, root, linearized, Variable(torch.FloatTensor([0.0])))
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



#          save_path = 
#          #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
#          with open(save_path+"/manual_output_funchead_two_coarse_lambda09_best/"+args.language+"_"+__file__+"_model_"+str(myID)+".tsv", "w") as outFile:
# 
relevantPath = "/u/scr/mhahn/deps/optimized_i1/"

import os
#files = [x for x in os.listdir(relevantPath) if x.startswith(args.language+"_")]
#posCount = 0
#negCount = 0
#for name in files:
#  with open(relevantPath+name, "r") as inFile:
#    for line in inFile:
#        line = line.split("\t")
#        if line[5] == "obj":
#          dhWeight = float(line[4])
#          if dhWeight < 0:
#             negCount += 1
#          elif dhWeight > 0:
#             posCount += 1
#          break
#
#print(["Neg count", negCount, "Pos count", posCount])
#
#if posCount >= 4 and negCount >= 4:
#   print("Enough models!")
#   quit()

#with open(relevantPath+"/LOG"+args.language+"_"+__file__+"_model_"+str(myID)+".txt", "w") as outFile:
#    print >> outFile, " ".join(sys.argv)

dhWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
distanceWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
for i, key in enumerate(itos_deps):

   # take from treebank, or randomize
   dhLogits[key] = 0.1*(random()-0.5)
#   if key == "obj": # this being the reference point, we fix this
 #      dhLogits[key] = (10.0 if posCount < negCount else -10.0)

   dhWeights.data[i] = dhLogits[key]

   originalDistanceWeights[key] = 0.0 #0.1*(random()-0.5)
   distanceWeights.data[i] = originalDistanceWeights[key]



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




baseline = torch.nn.Embedding(num_embeddings = len(itos_pos_uni)+vocab_size+3, embedding_dim=1).cuda()
word_pos_morph_embeddings = torch.nn.Embedding(num_embeddings = len(itos_pos_uni)+vocab_size+3, embedding_dim=args.emb_dim).cuda()
print posUni
#print posFine
print "VOCABULARY "+str(vocab_size+3)
outVocabSize = len(itos_pos_uni)+vocab_size+3
#quit()


itos_total = ["EOS", "EOW", "SOS"] + itos_pos_uni + itos[:vocab_size]
assert len(itos_total) == outVocabSize



from torch import optim




#################################################################################3
# Language Model
#################################################################################3



dropout = nn.Dropout(args.dropout_rate).cuda()

rnn = nn.LSTM(args.emb_dim, args.rnn_dim, args.rnn_layers).cuda()
for name, param in rnn.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)

decoder = nn.Linear(args.rnn_dim,outVocabSize).cuda()
#pos_ptb_decoder = nn.Linear(128,len(posFine)+3).cuda()

components = [rnn, decoder, word_pos_morph_embeddings, baseline]
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
#word_embeddings.weight.data.uniform_(-initrange, initrange)
#pos_u_embeddings.weight.data.uniform_(-initrange, initrange)
#pos_p_embeddings.weight.data.uniform_(-initrange, initrange)
#morph_embeddings.weight.data.uniform_(-initrange, initrange)
word_pos_morph_embeddings.weight.data.uniform_(-initrange, initrange)

decoder.bias.data.fill_(0)
decoder.weight.data.uniform_(-initrange, initrange)
#pos_ptb_decoder.bias.data.fill_(0)
#pos_ptb_decoder.weight.data.uniform_(-initrange, initrange)
#baseline.bias.data.fill_(0)
baseline.weight.data.fill_(math.log(vocab_size)) #uniform_(-initrange, initrange)




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


#def encodeWord(w):
#   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)



import torch.cuda
import torch.nn.functional

inputDropout = torch.nn.Dropout2d(p=args.input_dropoutRate)


counter = 0


lastDevLoss = None

failedDevRuns = 0

devLosses = [] 

#loss_op = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index = 0)
lossModuleTest = nn.NLLLoss(size_average=False, reduce=False, ignore_index=2)


baselineAverageLoss = 1.0

def doForwardPass(input_indices, wordStartIndices, surprisalTable=None, doDropout=True, batchSizeHere=1, relevant_logprob_sum=None):
       global counter
       global crossEntropy
       global printHere
       global devLosses
       global baselineAverageLoss

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

       sequenceLength = max(map(len, input_indices))
       for i in range(batchSizeHere):
          input_indices[i] = input_indices[i][:]
          while len(input_indices[i]) < sequenceLength:
             input_indices[i].append(2)

       inputTensor = Variable(torch.LongTensor(input_indices).transpose(0,1).contiguous()).cuda() # so it will be sequence_length x batchSizeHere
       inputTensorIn = inputTensor[:-1].view(1, -1)
       inputTensorOut = inputTensor[1:].view(1, -1)

       inputEmbeddings = word_pos_morph_embeddings(inputTensorIn)
       #print("inputEmbeddings", inputEmbeddings.size())
       if doDropout:
          inputEmbeddings = inputDropout(inputEmbeddings)
          if args.dropout_rate > 0:
             inputEmbeddings = dropout(inputEmbeddings)
       output, hidden = rnn(inputEmbeddings, hidden)

       if doDropout:
          output = dropout(output)
       word_logits = decoder(output)
       word_logits = word_logits.view((sequenceLength-1)*batchSizeHere, outVocabSize)
       word_softmax = logsoftmax(word_logits)




       lossesWord = lossModuleTest(word_softmax, inputTensorOut.view((sequenceLength-1)*batchSizeHere))
       loss = lossesWord.sum()
       wordNum = (len(wordStartIndices[0]) - 1)*batchSizeHere
       
       #print("lossesWord", lossesWord.size())

       # TODO here

#       lossesWord_tensor = loss_op(word_softmax.view(-1, outVocabSize), inputTensorOut.view(-1)).view(-1, batchSizeHere)

       reward = (lossesWord).detach()

       baseline_predictions =  baseline(inputTensorOut)


       baseline_shifted = baseline_predictions


     #  print(zip(baseline_shifted.view(-1), reward.view(-1)))
       baselineLoss = torch.nn.functional.mse_loss(baseline_shifted.view(-1, batchSizeHere), reward.view(-1, batchSizeHere), size_average=False, reduce=False)

       baselineAverageLoss = 0.99 * baselineAverageLoss + (1-0.99) * baselineLoss.cpu().data.mean().numpy()
       if printHere:
          print(baselineLoss)
          print(["Baseline loss", sqrt(baselineAverageLoss)])

       rewardMinusBaseline = (reward.view(-1, batchSizeHere) - baseline_shifted.view(-1, batchSizeHere)).detach().cpu()

#       for i in range(0,len(input_words)-1): #range(1,maxLength+1): # don't include i==0 -- but why?? changed this June 29, 2018
       if relevant_logprob_sum is not None:
         for j in range(batchSizeHere):
 #            if input_words[i+1][j] != 0:
          policyGradientLoss += (rewardMinusBaseline[:,j].sum() * relevant_logprob_sum[j])
          #      if input_words[i+1] > 2 and j == 0 and printHere:
           #        print [itos[input_words[i+1][j]-3], itos_pos_ptb[input_pos_p[i+1][j]-3], lossesWord_tensor[i][j].data.cpu().numpy(), lossesPOS_tensor[i][j].data.cpu().numpy(), baseline_predictions[i+1][j].data.cpu().numpy()]
           #     wordNum += 1

       lossWords = lossesWord.sum()
       loss += lossWords





           
       if wordNum == 0:
         print input_words
         print batchOrdered
         return 0,0,0,0,0
       if printHere:
         print loss/wordNum
         print lossWords/wordNum
         print ["CROSS ENTROPY LM", crossEntropy, exp(crossEntropy)]
         print baselineAverageLoss
       crossEntropy = 0.99 * crossEntropy + 0.01 * (lossWords/wordNum).data.cpu().numpy()
       totalQuality = loss.data.cpu().numpy() # consists of lossesWord + lossesPOS
       numberOfWords = wordNum

       policy_related_loss =  policyGradientLoss # lives on CPU
       return (loss/batchSizeHere, baselineLoss/batchSizeHere, policy_related_loss/batchSizeHere, totalQuality, numberOfWords)


def  doBackwardPass(loss, baselineLoss, policy_related_loss):
       if printHere:
         print "BACKWARD 1"
       policy_related_loss = policy_related_loss

       global dhWeights
       probabilities = torch.sigmoid(dhWeights)
       neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))
       policy_related_loss += args.entropy_weight * neg_entropy # lives on CPU


       policy_related_loss.backward()
       if printHere:
         print "BACKWARD 2"

       totalLoss = loss
       totalLoss += args.lr_baseline * baselineLoss.sum()
       totalLoss.backward() # lives on GPU



       if printHere:
         print args 
         print "BACKWARD 3 "+__file__+" "+args.language+" "+str(myID)+" "+str(counter)+" "+str(lastDevLoss)+" "+str(failedDevRuns)+"  "+(" ".join(map(str,["ENTROPY", args.entropy_weight, "LR_POLICY", args.lr_policy, "MOMENTUM", args.momentum_policy])))
         print "dev losses LM"
         print devLosses
         print crossEntropy

#       print("Before")
   #    if printHere:
    #      print([counter, "Distance grad", torch.abs(distanceWeights.grad).sum()])
       if random() > 0.99:
          print("INFINITY NORM OF GRADIENTS", max(p.grad.data.abs().max() for p in parameters_policy_cached))
#       if len([x for x in parameters_policy_cached if x.grad is not None]) > 0:
#          torch.nn.utils.clip_grad_norm(parameters_policy_cached, 5.0, norm_type='inf')
#       else:
#         print "Skipping gradient clipping for policy"
       torch.nn.utils.clip_grad_norm(parameters_cached, 5.0, norm_type='inf')
       counterHere = 0


       for param in parameters_cached:
         counterHere += 1
         if param.grad is None:
           assert False
#           print counterHere
#           print "WARNING: None gradient"
#           continue
         param.data.sub_(args.lr * param.grad.data)





       for param in parameters_policy_cached:
         counterHere += 1
         if counter < 200 and (param is distanceWeights or param is dhWeights): # allow baseline to warum up
             continue
         if param.grad is None:
           print counterHere
           print "WARNING: None gradient"
           continue
         param.data.sub_(args.lr_policy * param.grad.data)



def computeDevLoss():
   devBatchSize = 1
   global printHere
#   global counter
#   global devSurprisalTable
   devLoss = 0.0
   devWords = 0
#   corpusDev = getNextSentence("dev")
   corpusDev = CorpusIterator(args.language,"dev").iterator(rejectShortSentences = False)
   stream = createStream(corpusDev, training=False)

   surprisalTable = [0 for _ in range(2)]
   devCounter = 0
   devCounterTimesBatchSize = 0
   while True:
#     try:
#        input_indices, wordStartIndices = next(stream)
     try:
        input_indices_list = []
        wordStartIndices_list = []
        for _ in range(devBatchSize):
           input_indices, wordStartIndices, _ = next(stream)
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
     _, _, _, newLoss, newWords = doForwardPass(input_indices_list, wordStartIndices_list, surprisalTable = surprisalTable, doDropout=False, batchSizeHere=devBatchSize,relevant_logprob_sum=None )
     devLoss += newLoss
     devWords += newWords
     if printHere:
         print "Dev examples "+str(devCounter)
     devCounterTimesBatchSize += devBatchSize
   #devSurprisalTableHere = [surp/(devCounterTimesBatchSize) for surp in surprisalTable]
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
       if sentCount % 100 == 0:
          print("\t".join([str(sentCount), "Sentences"]))
       ordered, _, relevant_logprob_sum = orderSentence(sentence, dhLogits, printHere)
       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
            if training and random() < args.replaceWordsProbability:
                targetWord = randint(0,vocab_size-1)
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
while failedDevRuns == 0:
  epochCount += 1
  print "Starting new epoch, permuting corpus"
  corpusBase.permute()
#  corpus = getNextSentence("train")
  corpus = corpusBase.iterator(rejectShortSentences = False)
  stream = createStream(corpus)



  if counter > 5:
#       if counter % DEV_PERIOD == 0:
          newDevLoss, _ = computeDevLoss()
#             devLosses.append(
          devLosses.append(newDevLoss)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
          if newDevLoss > 15 or len(devLosses) > 99:
              print "Abort, training too slow?"
              devLosses.append(newDevLoss+0.001)

#          if lastDevLoss is None or newDevLoss < lastDevLoss:
              #devSurprisalTable = devSurprisalTableHere
        #  with open(TARGET_DIR+"/estimates-"+language+"_"+__file__+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
        #      print >> outFile, " ".join(sys.argv)
        #      print >> outFile, " ".join(map(str,devLosses))
        #      print >> outFile, " ".join(map(str,devSurprisalTable))
        #      print >> outFile, "PARAMETER_SEARCH" if DOING_PARAMETER_SEARCH else "RUNNING"

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
#             print zip(range(1,horizon+1), devSurprisalTable)
#             print devSurprisalTable


             break

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
       loss, baselineLoss, policy_related_loss, _, wordNumInPass = doForwardPass(input_indices_list, wordStartIndices_list, batchSizeHere=devBatchSize, relevant_logprob_sum=relevant_logprob_sum)
       if wordNumInPass > 0:
         doBackwardPass(loss, baselineLoss, policy_related_loss)
       else:
         print "No words, skipped backward"
       if printHere:
          print "Epoch "+str(epochCount)+" "+str(counter)
 #         print zip(range(1,horizon+1), devSurprisalTable)
#          print devSurprisalTable

#while True:
##  corpus = getNextSentence("train")
#  corpus = CorpusIteratorFuncHead(args.language).iterator(rejectShortSentences = True)
#
#
#  while True:
#    try:
#       batch = map(lambda x:next(corpus), 10*range(args.batchSize))
#    except StopIteration:
#       break
#    batch = sorted(batch, key=len)
#    partitions = range(10)
#    shuffle(partitions)
#    for partition in partitions:
#       counter += 1
#       printHere = (counter % 100 == 0)
#       current = batch[partition*args.batchSize:(partition+1)*args.batchSize]
#
#       fromLM = doForwardPass(current)
#       loss, baselineLoss, policy_related_loss, _, wordNumInPass = fromLM
##       print(fromLM, fromParser, fromDepL)
#
#       if wordNumInPass > 0:
#         doBackwardPass(loss, baselineLoss, policy_related_loss)
#       else:
#         print "No words, skipped backward"
#       if counter % 50000 == 0:
#          newDevLoss = computeDevLoss()
#          devLosses.append(newDevLoss)
#          print "New dev loss LM     "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
#
#          if lastDevLoss is None or newDevLoss < lastDevLoss:
#             lastDevLoss = newDevLoss
#             failedDevRuns = 0
#          else:
#             failedDevRuns += 1
#             print "Skip LM, hoping for better model"
#             print devLosses
#
#
#
#          if True:
#             print("DEMO, NOT SAVING SO FAR")
#             assert False
#             continue
#          print "Saving"
#          #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
#          with open(relevantPath"/"+args.language+"_"+__file__+"_model_"+str(myID)+".tsv", "w") as outFile:
#             print >> outFile, "\t".join(map(str,["FileName","ModelName","Counter", "AverageLoss_LM", "DH_Weight","CoarseDependency","DistanceWeight", "EntropyWeight", "ObjectiveName"]))
#             for i in range(len(itos_deps)):
#                key = itos_deps[i]
#                dhWeight = dhWeights[i].data.numpy()
#                distanceWeight = distanceWeights[i].data.numpy()
#                dependency = key
#                print >> outFile, "\t".join(map(str,[myID, __file__, counter, devLosses[-1], dhWeight, dependency, distanceWeight, args.entropy_weight, objectiveName]))
#
#          if failedDevRuns >0:
#              quit()
#
#
#
