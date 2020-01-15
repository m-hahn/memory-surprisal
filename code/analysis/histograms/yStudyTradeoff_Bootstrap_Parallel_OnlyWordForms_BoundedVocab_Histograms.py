# A histogram at one surprisal point


import sys

language = sys.argv[1]
#random = sys.argv[2] if len(sys.argv) > 2 else "RANDOM_BY_TYPE"
#real = sys.argv[3] if len(sys.argv) > 3 else "REAL_REAL"

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
with open("/u/scr/mhahn/"+language+"_decay_after_tuning_onlyWordForms_boundedVocab.tsv", "r") as inFile:
     data = readTSV(inFile)
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
for typ in matrixByType:
   matrixByType[typ] = torch.FloatTensor(matrixByType[typ]).view(-1, 19, 4)
   #print(matrixByType[typ][0])
   misByType[typ] = matrixByType[typ][:,:,2]
#print(misByType[random])

#data = data %>% group_by(ModelID) %>% mutate(CumulativeMemory = cumsum(Distance*ConditionalMI), CumulativeMI = cumsum(ConditionalMI))
distance = torch.FloatTensor(range(1,20))

cumMIs = {}
cumMems = {}
cumInterpolated = {}
maximalMemory = 0
maximalMI = 0
for typ, mis in misByType.iteritems():
  mask = torch.FloatTensor([[1 if j <= i else 0 for j in range(19)] for i in range(19)])
  cumulativeMI = torch.matmul(mis, mask.t())
  #print("MIs", mis[0])
  #print("Cum MI", cumulativeMI[0])
  cumulativeMemory = torch.matmul(distance*mis, mask.t())
  #print("Cum Mem", cumulativeMemory[0])
  cumMIs[typ] = cumulativeMI
  cumMems[typ] = cumulativeMemory
  maximalMemory = max(maximalMemory, float(torch.max(cumulativeMemory)))
  maximalMI = max(maximalMI, float(torch.max(cumulativeMI)))

#print("MAXIMAL MEMORY", maximalMemory)


for typ, mis in misByType.iteritems():  
  cumMIs[typ] = torch.cat([0*cumMIs[typ][:,-1].unsqueeze(1), cumMIs[typ], cumMIs[typ][:,-1].unsqueeze(1)], dim=1)
  cumMems[typ] = torch.cat([0*(cumMems[typ][:,-1].unsqueeze(1)), cumMems[typ], maximalMemory + 0*(cumMems[typ][:,-1].unsqueeze(1))], dim=1)

  #print(cumMIs[typ][0])
import math

xPoints = torch.FloatTensor([maximalMemory*x/40.0 for x in range(1,40)])

xPointsMI = torch.FloatTensor([maximalMI*x/40.0 for x in range(1,40)])

for typ, mis in misByType.iteritems():  
  cumulativeMI = cumMIs[typ]
  cumulativeMemory = cumMems[typ]
 # print(cumulativeMemory[0])
  xBigger = (cumMems[typ].unsqueeze(2) > xPoints.unsqueeze(0).unsqueeze(0))
  xSmaller = (cumMems[typ].unsqueeze(2) <= xPoints.unsqueeze(0).unsqueeze(0))

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
  if typ in ["REAL_REAL", "GROUND"]:
     interpolated = interpolated.mean(dim=0).view(1,-1)
  cumInterpolated[typ] = interpolated

  # for each MI value, find the adjacent values
  
maximalVariance = 0
xPoint_MI_MaximalVariance = 0
resulting_MaximalVariance = None
for j in range(0, len(xPointsMI)):
   MIPoint = float(xPointsMI[j])
   resulting = {}
   resultingFull = torch.FloatTensor([])
   for typ, interpolated in cumInterpolated.items():
      isBigger = (MIPoint < interpolated)
      isSmaller = (MIPoint >= interpolated)
      matches = (isBigger[:,1:] * isSmaller[:,:-1])
#      resulting[typ] = torch.masked_select(interpolated[:,:-1], matches)
      resulting[typ] = torch.masked_select(xPoints[:-1], matches)

      resultingFull = torch.cat([resultingFull, resulting[typ]], dim=0)
   if len(resulting["REAL_REAL"]) == 0 or len(resulting["RANDOM_BY_TYPE"]) == 0:
      continue
 #  print("========"+str(j))
#   print(resultingFull)
#   print(resulting)
   variance = float(torch.mean(torch.pow(resultingFull, 2)) - torch.pow(torch.mean(resultingFull), 2))
   if variance > maximalVariance:
      maximalVariance = variance
      xPoint_MI_MaximalVariance = j
      resulting_MaximalVariance = resulting
#print(resulting_MaximalVariance)
#print(xPoint_MI_MaximalVariance, maximalVariance, xPointsMI[xPoint_MI_MaximalVariance])
for typ, memories in resulting_MaximalVariance.items():
  for mem in memories:
   print("\t".join(map(str, [language, typ, float(xPointsMI[xPoint_MI_MaximalVariance]), float(mem)])))
 #  print(torch.pow(resultingFull, 2))
  # print(torch.pow(torch.mean(resultingFull), 2))
#   print(variance)

