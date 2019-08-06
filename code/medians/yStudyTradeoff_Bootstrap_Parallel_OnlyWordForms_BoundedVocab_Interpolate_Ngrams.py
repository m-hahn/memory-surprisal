# Better than yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_BinomialTest_Single_MaxControl.py by modeling the median of REAL

import sys

language = sys.argv[1]

def readTSV(x):
    header = next(x).strip().split("\t")
    header = dict(zip(header, range(len(header))))
    data = [y.strip().split("\t") for y in x]
    if len(data) < 2:
       return (header, [])
    for column in range(len(header)):
      try:
        vals=  [int(y[column]) for y in data]
      except ValueError:
        try:
          vals=  [float(y[column]) for y in data]
        except ValueError:
          vals=  [y[column] for y in data]
      for i in range(len(data)):
          data[i][column] = vals[i]
    return (header, data)
try:
  with open("../../results/raw/ngrams/"+language+"_ngrams_decay_after_tuning.tsv", "r") as inFile:
     data = readTSV(inFile)
except IOError:
  print >> sys.stderr, ("ERROR nothing for this language? "+language)
  quit()

#print(len(data))



#data = data %>% group_by(ModelID) %>% mutate(CumulativeMemory = cumsum(Distance*ConditionalMI), CumulativeMI = cumsum(ConditionalMI))

import torch

def g(frame, name, i):
    return frame[1][i][frame[0][name]]

matrix = [[g(data, "ModelID", i), g(data, "Distance", i), g(data, "ConditionalMI", i), g(data, "UnigramCE", i)] for i in range(len(data[1]))]

matrixByType = {}
misByType = {}
#unigramCEByType = {}
for i in range(len(data[1])):
    typ = g(data, "Type", i)
#    print(i,typ, len(data[1]))
    if typ not in matrixByType:
        matrixByType[typ] = []
    matrixByType[typ].append(matrix[i])
MAX_DISTANCE = -1
for typ in matrixByType:
   tensorized = torch.FloatTensor(matrixByType[typ])
   MAX_DISTANCE = int(tensorized[:,1].max())
   matrixByType[typ] = tensorized.view(-1, MAX_DISTANCE, 4)
   #print(matrixByType[typ][0])
   misByType[typ] = matrixByType[typ][:,:,2]
#print(misByType["RANDOM_BY_TYPE"])
if MAX_DISTANCE == -1:
  print >> sys.stderr, ("ERROR nothing for this language? "+language)
  quit()
#data = data %>% group_by(ModelID) %>% mutate(CumulativeMemory = cumsum(Distance*ConditionalMI), CumulativeMI = cumsum(ConditionalMI))
distance = torch.FloatTensor(range(1,1+MAX_DISTANCE))

cumMIs = {}
cumMems = {}
cumInterpolated = {}
maximalMemory = 0
for typ, mis in misByType.iteritems():
  mask = torch.FloatTensor([[1 if j <= i else 0 for j in range(MAX_DISTANCE)] for i in range(MAX_DISTANCE)])
  cumulativeMI = torch.matmul(mis, mask.t())
  #print("MIs", mis[0])
  #print("Cum MI", cumulativeMI[0])
  cumulativeMemory = torch.matmul(distance*mis, mask.t())
  #print("Cum Mem", cumulativeMemory[0])
  cumMIs[typ] = cumulativeMI
  cumMems[typ] = cumulativeMemory
  maximalMemory = max(maximalMemory, float(torch.max(cumulativeMemory)))

#print("MAXIMAL MEMORY", maximalMemory)


for typ, mis in misByType.iteritems():  
  cumMIs[typ] = torch.cat([0*cumMIs[typ][:,-1].unsqueeze(1), cumMIs[typ], cumMIs[typ][:,-1].unsqueeze(1)], dim=1)
  cumMems[typ] = torch.cat([0*(cumMems[typ][:,-1].unsqueeze(1)), cumMems[typ], maximalMemory + 0*(cumMems[typ][:,-1].unsqueeze(1))], dim=1)

  #print(cumMIs[typ][0])
import math

xPoints = torch.FloatTensor([maximalMemory*x/40.0 for x in range(1,40)])

interpolatedByTypes = {}

for typ, mis in misByType.iteritems():  
  cumulativeMI = cumMIs[typ]
  cumulativeMemory = cumMems[typ]
 # print(cumulativeMemory[0])
#  print(cumulativeMI[0])
#0.0 to 2.0
  #print(cumMIs[typ].size())
  #print("X POINTS", xPoints)
  xBigger = (cumMems[typ].unsqueeze(2) > xPoints.unsqueeze(0).unsqueeze(0))
  xSmaller = (cumMems[typ].unsqueeze(2) <= xPoints.unsqueeze(0).unsqueeze(0))

  #print(torch.all(xSmaller + xBigger == 1))

  #print(xBigger.size())
  interpolated = torch.zeros(cumulativeMemory.size()[0], len(xPoints))
  foundValues = torch.zeros(cumulativeMemory.size()[0], len(xPoints))
  for j in range(0,len(xPoints)):
     condition = (xBigger[:,1:,j] * xSmaller[:,:-1,j]).float()
     #print(j, condition[0])
     memoryDifference = ((cumulativeMemory[:,1:] - cumulativeMemory[:,:-1]))
     slope = (xPoints[j] - cumulativeMemory[:,:-1]) / memoryDifference
     slope[memoryDifference == 0] = 1/2
     interpolation = (cumulativeMI[:,:-1] + slope * (cumulativeMI[:,1:] - cumulativeMI[:,:-1]))
     interpolated[:,j] = torch.sum(condition  * interpolation, dim=1)
     foundValues[:,j] = torch.sum(condition, dim=1)
  interpolatedByTypes[typ] = interpolated
  for i in range(0, interpolated.size()[0]):
      for j in range(0, len(xPoints)):
        print "\t".join(map(str, [language, typ, int(matrixByType[typ][i][0][0]), j, float(xPoints[j]), float(interpolated[i,j])]))
#  print(interpolated.size())
 # print(matrixByType[typ].size())
#       print "\t".join(map(str,[language, typ, i, float(xPoints[i]), float(interpolated[:,i].median()), bestCI[0], bestCI[1], bestCI[2]]))


