#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"



# TODO also try other optimizers

import random
import sys

objectiveName = "LM"

language = sys.argv[1]
languageCode = sys.argv[2]
dropout_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.33
emb_dim = int(sys.argv[4]) if len(sys.argv) > 4 else 100
rnn_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 512
rnn_layers = int(sys.argv[6]) if len(sys.argv) > 6 else 2
lr_lm = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
model = sys.argv[8]
input_dropoutRate = float(sys.argv[9]) # 0.33
batchSize = int(sys.argv[10])
replaceWordsProbability = float(sys.argv[11])
horizon = int(sys.argv[12]) if len(sys.argv) > 12 else 20
prescripedID = sys.argv[13] if len(sys.argv)> 13 else None
gpuNumber = sys.argv[14] if len(sys.argv) > 14 else "GPU0"
assert gpuNumber.startswith("GPU")
gpuNumber = int(gpuNumber[3:])

#if len(sys.argv) == 13:
#  del sys.argv[12]
assert len(sys.argv) in [12,13,14, 15]


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

TARGET_DIR = "/u/scr/mhahn/deps/memory-need-neural-wordforms/"

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
   for partition in ["train", "dev"]:
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
       global model
#       childrenLinearized = []
#       while len(remainingChildren) > 0:
       if model == "REAL":
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
         print "\t".join(map(str,["ORD", line["index"], ("|".join(line["morph"])+"           ")[:10], ("->".join(list(key)) + "         ")[:22], line["head"], dhLogit, dhSampled, direction]))

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
   recursivelyLinearize(sentence, root, linearized, 0)
   if model == "REAL_REAL":
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

vocab_size = 10000
vocab_size = min(len(itos),vocab_size)
#print itos[:vocab_size]
#quit()

# 0 EOS, 1 UNK, 2 BOS
#word_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = emb_dim).cuda()
#pos_u_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = 10).cuda()
#pos_p_embeddings = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=10).cuda()
#morph_embeddings = torch.nn.Embedding(num_embeddings = len(morphKeyValuePairs)+3, embedding_dim=100).cuda()



torch.cuda.set_device(gpuNumber)

word_pos_morph_embeddings = torch.nn.Embedding(num_embeddings = len(itos_pos_uni)+vocab_size+3, embedding_dim=emb_dim).cuda()
print posUni
#print posFine
print "VOCABULARY "+str(vocab_size+3)
outVocabSize = len(itos_pos_uni)+vocab_size+3
#quit()


itos_total = ["EOS", "EOW", "SOS"] + itos_pos_uni + itos[:vocab_size]
assert len(itos_total) == outVocabSize
# could also provide per-word subcategorization frames from the treebank as input???


#baseline = nn.Linear(emb_dim, 1).cuda()

dropout = nn.Dropout(dropout_rate).cuda()

rnn = nn.LSTM(emb_dim, rnn_dim, rnn_layers).cuda()
for name, param in rnn.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)

decoder = nn.Linear(rnn_dim,outVocabSize).cuda()
#pos_ptb_decoder = nn.Linear(128,len(posFine)+3).cuda()

components = [rnn, decoder, word_pos_morph_embeddings]
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
#baseline.weight.data.uniform_(-initrange, initrange)




crossEntropy = 10.0

#def encodeWord(w):
#   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)



import torch.cuda
import torch.nn.functional

inputDropout = torch.nn.Dropout2d(p=input_dropoutRate)


counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 


lossModule = nn.NLLLoss()
lossModuleTest = nn.NLLLoss(size_average=False, reduce=False, ignore_index=2)

def doForwardPass(input_indices, wordStartIndices, surprisalTable=None, doDropout=True, batchSizeHere=1):
       global counter
       global crossEntropy
       global printHere
       global devLosses
       if printHere:
           print "wordStartIndices"
           print wordStartIndices

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSizeHere, 128).zero_()), Variable(torch.FloatTensor().new(2, batchSizeHere, 128).zero_()))
       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0
       for c in components:
          c.zero_grad()

       totalQuality = 0.0

       if True:
           
           sequenceLength = max(map(len, input_indices))
           for i in range(batchSizeHere):
              input_indices[i] = input_indices[i][:]
              while len(input_indices[i]) < sequenceLength:
                 input_indices[i].append(2)

           inputTensor = Variable(torch.LongTensor(input_indices).transpose(0,1).contiguous()).cuda() # so it will be sequence_length x batchSizeHere
#           print inputTensor
#           quit()

           inputTensorIn = inputTensor[:-1]
           inputTensorOut = inputTensor[1:]

#           print(inputTensorIn)
 #          print(inputTensorIn.size())
           inputEmbeddings = word_pos_morph_embeddings(inputTensorIn.view(sequenceLength-1, batchSizeHere))
           if doDropout:
              inputEmbeddings = inputDropout(inputEmbeddings)
              if dropout_rate > 0:
                 inputEmbeddings = dropout(inputEmbeddings)
  #         print(inputEmbeddings)
   #        print(inputEmbeddings.size())
           output, hidden = rnn(inputEmbeddings, hidden)

    #       print(output)
           # word logits
           if doDropout:
              output = dropout(output)
     #      print(output)
      #     print(output.size())
           word_logits = decoder(output)
       #    print(word_logits)
        #   print(word_logits.size())
         #  print(outVocabSize)
           word_logits = word_logits.view((sequenceLength-1)*batchSizeHere, outVocabSize)
           #print(word_logits)
           word_softmax = logsoftmax(word_logits)
          # print(word_softmax)
#           word_softmax = word_softmax.view(-1, batchSizeHere, outVocabSize)

#           lossesWord = [[0]*batchSizeHere for i in range(len(input_indices))]

#           print inputTensorOut.view((sequenceLength-1)*batchSizeHere)
#           quit()
           lossesWord = lossModuleTest(word_softmax, inputTensorOut.view((sequenceLength-1)*batchSizeHere))
#           try:
           loss = lossesWord.sum()
#           except RuntimeError:
#               print(word_softmax.size())
#               print(inputTensorOut.size())
#               print((sequenceLength-1), batchSizeHere)
#               print(word_logits)
#               print(word_softmax)
#               print(inputTensorOut.view(-1))
#               print(lossesWord)
#               print(lossesWord.size())
#               quit()
#           print lossesWord

           if surprisalTable is not None or printHere:           
             lossesCPU = lossesWord.data.cpu().view((sequenceLength-1), batchSizeHere).numpy()
             if printHere:
                for i in range(0,len(input_indices[0])-1): #range(1,maxLength+1): # don't include i==0
#                   for j in range(batchSizeHere):
                         j = 0
                         print (i, itos_total[input_indices[j][i+1]], lossesCPU[i][j])

             if surprisalTable is not None: 
                if printHere:
                   print surprisalTable
                for j in range(batchSizeHere):
                  for r in range(horizon):
                    assert wordStartIndices[j][r]< wordStartIndices[j][r+1]
                    assert wordStartIndices[j][r] < len(lossesWord)+1, (wordStartIndices[j][r],wordStartIndices[j][r+1], len(lossesWord))
                    assert input_indices[j][wordStartIndices[j][r+1]-1] != 2
                    if r == horizon-1:
                      assert wordStartIndices[j][r+1] == len(input_indices[j]) or input_indices[j][wordStartIndices[j][r+1]] == 2
#                    print lossesCPU[wordStartIndices[j][r]:wordStartIndices[j][r+1],j]
 #                   surprisalTable[r] += sum([x.mean() for x in lossesCPU[wordStartIndices[j][r]:wordStartIndices[j][r+1],j]]) #.data.cpu().numpy()[0]
                    surprisalTable[r] += sum(lossesCPU[wordStartIndices[j][r]-1:wordStartIndices[j][r+1]-1,j]) #.data.cpu().numpy()[0]
                   

           wordNum = (len(wordStartIndices[0]) - 1)*batchSizeHere
           assert len(wordStartIndices[0]) == horizon+1, map(len, wordStartIndices)
                    
       if wordNum == 0:
         print input_words
         print batchOrdered
         return 0,0,0,0,0
       if printHere:
         print loss/wordNum
         print lossWords/wordNum
         print ["CROSS ENTROPY", crossEntropy, exp(crossEntropy)]
       crossEntropy = 0.99 * crossEntropy + 0.01 * (loss/wordNum).data.cpu().numpy()
       totalQuality = loss.data.cpu().numpy() # consists of lossesWord + lossesPOS
       numberOfWords = wordNum
#       probabilities = torch.sigmoid(dhWeights)
#       neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))

#       policy_related_loss = lr_policy * (entropy_weight * neg_entropy + policyGradientLoss) # lives on CPU
       return loss, None, None, totalQuality, numberOfWords


parameterList = list(parameters())

def  doBackwardPass(loss, baselineLoss, policy_related_loss):
       global lastDevLoss
       global failedDevRuns
       loss.backward()
       if printHere:
         print "BACKWARD 3 "+__file__+" "+language+" "+str(myID)+" "+str(counter)+" "+str(lastDevLoss)+" "+str(failedDevRuns)+"  "+(" ".join(map(str,["Dropout (real)", dropout_rate, "Emb_dim", emb_dim, "rnn_dim", rnn_dim, "rnn_layers", rnn_layers, "MODEL", model])))
         print devLosses
       torch.nn.utils.clip_grad_norm(parameterList, 5.0, norm_type='inf')
       for param in parameters():
         if param.grad is None:
           print "WARNING: None gradient"
           continue
         param.data.sub_(lr_lm * param.grad.data)



def createStream(corpus):
#    global counter
    global crossEntropy
    global printHere
    global devLosses

    input_indices = [2] # Start of Segment
    wordStartIndices = []
#    sentenceStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       #printHere = (sentCount % 10 == 0)
       ordered, _ = orderSentence(sentence, dhLogits, printHere)

#       sentenceStartIndices.append(len(input_indices))
       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
            if random() < replaceWordsProbability:
                targetWord = randint(0,vocab_size-1)
            else:
                targetWord = stoi[line["word"]]
            if targetWord >= vocab_size:

               input_indices.append(stoi_pos_uni[line["posUni"]]+3)
            else:
               input_indices.append(targetWord+3+len(itos_pos_uni))
          if len(wordStartIndices) == horizon:
             yield input_indices, wordStartIndices+[len(input_indices)]
             input_indices = [2] # Start of Segment (makes sure that first word can be predicted from this token)
             wordStartIndices = []



def createStreamContinuous(corpus):
#    global counter
    global crossEntropy
    global printHere
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
       ordered, _ = orderSentence(sentence, dhLogits, printHere)

#       sentenceStartIndices.append(len(input_indices))
       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
#            input_indices.append(stoi_pos_uni[line["posUni"]]+3)
#            if len(itos_pos_ptb) > 1:
#               input_indices.append(stoi_pos_ptb[line["posFine"]]+3+len(stoi_pos_uni))

            targetWord = stoi[line["word"]]
            if targetWord >= vocab_size:
               input_indices.append(stoi_pos_uni[line["posUni"]]+3)
            else:
               input_indices.append(targetWord+3+len(itos_pos_uni))
          if len(wordStartIndices) == horizon:
#             print input_indices
#             print wordStartIndices+[len(input_indices)]
             yield input_indices, wordStartIndices+[len(input_indices)]
             if DOING_PARAMETER_SEARCH:
               input_indices = [2] # Start of Segment (makes sure that first word can be predicted from this token)
               wordStartIndices = []
             else:
               input_indices = [2]+input_indices[wordStartIndices[1]:] # Start of Segment (makes sure that first word can be predicted from this token)
               wordStartIndices = [x-wordStartIndices[1]+1 for x in wordStartIndices[1:]]
               assert wordStartIndices[0] == 1




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

