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
DOING_PARAMETER_SEARCH = (sys.argv[15] == "True") if len(sys.argv) > 15 else False

trainingSize = int(sys.argv[16]) # 500, 5000
devSize = 2000

#########################################

assert len(sys.argv) == 17
assert dropout_rate <= 0.5
assert input_dropoutRate <= 0.5

#########################################

devSurprisalTable = [None] * horizon
if prescripedID is not None and prescripedID != "None":
  myID = int(prescripedID)
else:
  myID = random.randint(0,10000000)

import sys
print  >> sys.stderr, ("DOING PARAMETER SEARCH?", DOING_PARAMETER_SEARCH)
#assert not DOING_PARAMETER_SEARCH

TARGET_DIR = "/u/scr/mhahn/deps/memory-need-neural-wordforms-fractions/"

posUni = set() 


from math import log, exp
from random import random, shuffle, randint
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
     for sentence in CorpusIterator(language,partition, storeMorph=True, trainSize=trainingSize, devSize=devSize).iterator():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1

          depsVocab.add(line["dep"])
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
   dhLogits = {}
   for key in keys:
      hd = orderTable.get((key[0], key[1], key[2], "HD"), 0) + 1.0
      dh = orderTable.get((key[0], key[1], key[2], "DH"), 0) + 1.0
      dhLogit = log(dh) - log(hd)
      dhLogits[key] = dhLogit
      originalDistanceWeights[key] = (distanceSum[key] / distanceCounts[key])
   return dhLogits, vocab, keys, depsVocab

import torch.nn as nn
import torch
from torch.autograd import Variable


def recursivelyLinearize(sentence, position, result):
   line = sentence[position-1]

   if "children_DH" in line:
      for child in line["children_DH"]:
         recursivelyLinearize(sentence, child, result)
   result.append(line)
   if "children_HD" in line:
      for child in line["children_HD"]:
         recursivelyLinearize(sentence, child, result)

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       global model
       if model == "REAL":
          return remainingChildren
       logits = [(x, distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]]) for x in remainingChildren]
       logits = sorted(logits, key=lambda x:x[1], reverse=(not reverseSoftmax))
       childrenLinearized = map(lambda x:x[0], logits)
       return childrenLinearized           



def orderSentence(sentence, dhLogits, printThings):
   global model

   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0

   if model == "REAL_REAL":
       # Collect tokens to be removed (i.e., punctuation)
      eliminated = []
   for line in sentence:
      if line["dep"] == "root":
          root = line["index"]
          continue
      # Exclude Punctuation
      if line["dep"].startswith("punct"): 
         if model == "REAL_REAL":
            eliminated.append(line)
         continue
      # Determine ordering relative to head
      key = (sentence[line["head"]-1]["posUni"], line["dep"], line["posUni"])
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      if model == "REAL":
         dhSampled = (line["head"] > line["index"])
      else:
         dhSampled = (dhLogit > 0) 
     
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], ("->".join(list(key)) + "         ")[:22], line["head"], dhLogit, dhSampled, direction]))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])


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

   
   linearized = []
   recursivelyLinearize(sentence, root, linearized)
   if model == "REAL_REAL":
      linearized = filter(lambda x:"removed" not in x, sentence)
   if printThings or len(linearized) == 0:
     print " ".join(map(lambda x:x["word"], sentence))
     print " ".join(map(lambda x:x["word"], linearized))
   return linearized, logits


dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()



posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   
itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)


import os

if model == "REAL" or model == "REAL_REAL":
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
  assert False, args.model



words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))

if len(itos) > 6:
   assert stoi[itos[5]] == 5


vocab_size = 10000
vocab_size = min(len(itos),vocab_size)




torch.cuda.set_device(gpuNumber)


###########################################
# Initialize neural language model

word_pos_morph_embeddings = torch.nn.Embedding(num_embeddings = len(itos_pos_uni)+vocab_size+3, embedding_dim=emb_dim).cuda()
print posUni
print "VOCABULARY "+str(vocab_size+3)
outVocabSize = len(itos_pos_uni)+vocab_size+3


itos_total = ["EOS", "EOW", "SOS"] + itos_pos_uni + itos[:vocab_size]
assert len(itos_total) == outVocabSize

dropout = nn.Dropout(dropout_rate).cuda()

rnn = nn.LSTM(emb_dim, rnn_dim, rnn_layers).cuda()
for name, param in rnn.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)

decoder = nn.Linear(rnn_dim,outVocabSize).cuda()

components = [rnn, decoder, word_pos_morph_embeddings]

def parameters():
 for c in components:
   for param in c.parameters():
      yield param

initrange = 0.1
word_pos_morph_embeddings.weight.data.uniform_(-initrange, initrange)

decoder.bias.data.fill_(0)
decoder.weight.data.uniform_(-initrange, initrange)

import torch.cuda
import torch.nn.functional

inputDropout = torch.nn.Dropout2d(p=input_dropoutRate)

lossModule = nn.NLLLoss()
lossModuleTest = nn.NLLLoss(size_average=False, reduce=False, ignore_index=2)


#####################################################3


crossEntropy = 10.0
counter = 0
lastDevLoss = None
failedDevRuns = 0
devLosses = [] 



def doForwardPass(input_indices, wordStartIndices, surprisalTable=None, doDropout=True, batchSizeHere=1):
       global counter
       global crossEntropy
       global printHere
       global devLosses

       hidden = None
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

           inputTensorIn = inputTensor[:-1]
           inputTensorOut = inputTensor[1:]

           inputEmbeddings = word_pos_morph_embeddings(inputTensorIn.view(sequenceLength-1, batchSizeHere))
           if doDropout:
              inputEmbeddings = inputDropout(inputEmbeddings)
              if dropout_rate > 0:
                 inputEmbeddings = dropout(inputEmbeddings)
           output, hidden = rnn(inputEmbeddings, hidden)

           if doDropout:
              output = dropout(output)
           word_logits = decoder(output)
           word_logits = word_logits.view((sequenceLength-1)*batchSizeHere, outVocabSize)
           word_softmax = logsoftmax(word_logits)
           lossesWord = lossModuleTest(word_softmax, inputTensorOut.view((sequenceLength-1)*batchSizeHere))
           loss = lossesWord.sum()

           if surprisalTable is not None or printHere:           
             lossesCPU = lossesWord.data.cpu().view((sequenceLength-1), batchSizeHere).numpy()
             if printHere:
                for i in range(0,len(input_indices[0])-1):
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
       return loss, None, None, totalQuality, numberOfWords


parameterList = list(parameters())

def  doBackwardPass(loss, baselineLoss, policy_related_loss):
       global lastDevLoss
       global failedDevRuns
       loss.backward()
       if printHere:
         print "BACKWARD "+__file__+" "+language+" "+str(myID)+" "+str(counter)+" "+str(lastDevLoss)+" "+str(failedDevRuns)+"  "+(" ".join(map(str,["Dropout (real)", dropout_rate, "Emb_dim", emb_dim, "rnn_dim", rnn_dim, "rnn_layers", rnn_layers, "MODEL", model])))
         print devLosses
       torch.nn.utils.clip_grad_norm(parameterList, 5.0, norm_type='inf')
       for param in parameters():
         if param.grad is None:
           print "WARNING: None gradient"
           continue
         param.data.sub_(lr_lm * param.grad.data)



def createStream(corpus):
    global crossEntropy
    global printHere
    global devLosses

    input_indices = [2] # Start of Segment
    wordStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       ordered, _ = orderSentence(sentence, dhLogits, printHere)

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
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       if sentCount % 10 == 0:
         print ["DEV SENTENCES", sentCount]

       ordered, _ = orderSentence(sentence, dhLogits, printHere)

       for line in ordered+["EOS"]:
          wordStartIndices.append(len(input_indices))
          if line == "EOS":
            input_indices.append(0)
          else:
            targetWord = stoi[line["word"]]
            if targetWord >= vocab_size:
               input_indices.append(stoi_pos_uni[line["posUni"]]+3)
            else:
               input_indices.append(targetWord+3+len(itos_pos_uni))
          if len(wordStartIndices) == horizon:
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
   global horizon
   devLoss = 0.0
   devWords = 0
   corpusDev = CorpusIterator(language,"dev", storeMorph=True, trainSize=trainingSize, devSize=devSize).iterator(rejectShortSentences = False)
   stream = createStreamContinuous(corpusDev)

   surprisalTable = [0 for _ in range(horizon)]
   devCounter = 0
   devCounterTimesBatchSize = 0
   while True:
     try:
        input_indices_list = []
        wordStartIndices_list = []
        for _ in range(devBatchSize):
           input_indices, wordStartIndices = next(stream)
           input_indices_list.append(input_indices)
           wordStartIndices_list.append(wordStartIndices)
     except StopIteration:
        devBatchSize = len(input_indices_list)
     if devBatchSize == 0:
       break
     devCounter += 1
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
corpusBase = CorpusIterator(language, storeMorph=True, trainSize=trainingSize, devSize=devSize)
while failedDevRuns == 0:
  epochCount += 1
  print "Starting new epoch, permuting corpus"
  corpus = corpusBase.iterator(rejectShortSentences = False)
  stream = createStream(corpus)



  if counter > 5:
          newDevLoss, devSurprisalTableHere = computeDevLoss()
          devLosses.append(newDevLoss)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
          if newDevLoss > 15 or len(devLosses) > 99:
              print "Abort, training too slow?"
              devLosses.append(newDevLoss+0.001)

          if lastDevLoss is None or newDevLoss < lastDevLoss:
              devSurprisalTable = devSurprisalTableHere
          with open(TARGET_DIR+"/estimates-"+language+"_"+__file__+"_model_"+str(trainingSize)+"_"+str(myID)+"_"+model+".txt", "w") as outFile:
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

