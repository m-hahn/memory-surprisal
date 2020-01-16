import sys

language = sys.argv[1]

def readTSV(x):
    header = next(x).strip().split("\t")
    header = dict(zip(header, range(len(header))))
    data = [y.strip().split("\t") for y in x]
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
with open("../results/raw/"+language+"_decay_after_tuning_onlyWordForms_boundedVocab.tsv", "r") as inFile:
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
#print(misByType["RANDOM_BY_TYPE"])

#data = data %>% group_by(ModelID) %>% mutate(CumulativeMemory = cumsum(Distance*ConditionalMI), CumulativeMI = cumsum(ConditionalMI))
distance = torch.FloatTensor(range(1,20))

cumMIs = {}
cumMems = {}
cumInterpolated = {}
for typ, mis in misByType.iteritems():
  mask = torch.FloatTensor([[1 if j <= i else 0 for j in range(19)] for i in range(19)])
  cumulativeMI = torch.matmul(mis, mask.t())
  #print("MIs", mis[0])
  #print("Cum MI", cumulativeMI[0])
  cumulativeMemory = torch.matmul(distance*mis, mask.t())
  #print("Cum Mem", cumulativeMemory[0])
  cumMIs[typ] = cumulativeMI
  cumMems[typ] = cumulativeMemory
  
#0.0 to 2.0
  xPoints = torch.FloatTensor([x/10.0 for x in range(7,20)])
  #print(cumMIs[typ].size())
  #print(xPoints.size())
  xBigger = (cumMems[typ].unsqueeze(2) > xPoints.unsqueeze(0).unsqueeze(0))
  xSmaller = (cumMems[typ].unsqueeze(2) <= xPoints.unsqueeze(0).unsqueeze(0))

  #print(torch.all(xSmaller + xBigger == 1))

  #print(xBigger.size())
  interpolated = torch.zeros(cumulativeMI.size()[0], len(xPoints))
  foundValues = torch.zeros(cumulativeMI.size()[0], len(xPoints))
  for j in range(0,len(xPoints)):
     condition = (xBigger[:,1:,j] * xSmaller[:,:-1,j]).float()
   #  print(condition[0])
     memoryDifference = ((cumulativeMemory[:,1:] - cumulativeMemory[:,:-1]))
     slope = (xPoints[j] - cumulativeMemory[:,:-1]) / (cumulativeMemory[:,1:] - cumulativeMemory[:,:-1])
     slope[memoryDifference == 0] = 1/2
     interpolation = (cumulativeMI[:,:-1] + slope * (cumulativeMI[:,1:] - cumulativeMI[:,:-1]))
     interpolated[:,j] = torch.sum(condition  * interpolation, dim=1)
     foundValues[:,j] = torch.sum(condition, dim=1)
#  print(foundValues[0])
 # print("X", xPoints)
  #print("Inter", interpolated[0])
#  print("Mem", cumulativeMemory[0])
 # print("MI", cumulativeMI[0])
  cumInterpolated[typ] = (interpolated, foundValues)
#quit()

if "REAL_REAL" in cumInterpolated:
  typ1 = "REAL_REAL"
elif "REAL" in cumInterpolated:
  typ1 = "REAL"
else:
  print("\t".join([str(x) for x in [language, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0]]))
  print(1.0)

  quit()
typ2 = "RANDOM_BY_TYPE"

import random


result1 = []
result2 = []

samplesNumber = 800

for u in range(samplesNumber):
#   print(u)
   indices1 = [random.randint(0, len(cumInterpolated[typ1][0])-1) for _ in range(len(cumInterpolated[typ1][0]))]
   indices2 = [random.randint(0, len(cumInterpolated[typ2][0])-1) for _ in range(len(cumInterpolated[typ2][0]))]


   interpolated1 = cumInterpolated[typ1][0][indices1]
   interpolated2 = cumInterpolated[typ2][0][indices2]
   
   
   foundValues1 = cumInterpolated[typ1][1][indices1]
   foundValues2 = cumInterpolated[typ2][1][indices2]
   
   
   strictlyBiggerCounts = []
   strictlySmallerCounts = []
   biggerAverages = []
   smallerAverages = []

   comparison = (interpolated1.unsqueeze(1) > interpolated2.unsqueeze(0)).float()
 #  print(comparison.size())
   bothAreMeaningful = foundValues1.unsqueeze(1) * foundValues2.unsqueeze(0)
   comparableRange = (bothAreMeaningful.sum(2).unsqueeze(2)).sum(2) # length of the comparable range in each case
   comparableRange[comparableRange==0] = 1
  # print(comparableRange.size())
   bigger = (comparison * bothAreMeaningful)
   smaller = ((1- comparison) * bothAreMeaningful)
   bigger = bigger.sum(2) / comparableRange
   smaller = smaller.sum(2) / comparableRange
   strictlyBiggerCounts = (  torch.sum(bigger == 1.0, dim=1))
   strictlySmallerCounts = ( torch.sum(smaller == 1.0, dim=1))
   biggerAverages = ( torch.sum(bigger, dim=1))
   smallerAverages = ( torch.sum(smaller, dim=1))
 #  print(bigger.size())
#   quit()
#   print(strictlyBiggerCounts.size())
   bigger = (torch.sum(strictlyBiggerCounts).numpy() / len(strictlyBiggerCounts))
   smaller = (torch.sum(strictlySmallerCounts).numpy() / len(strictlySmallerCounts))
   biggerAvg = (torch.sum(biggerAverages).numpy() / len(biggerAverages)) / len(interpolated2)
#   print("\t".join([str(x) for x in [language, bigger/(bigger+smaller+0.00000001), biggerAvg]]))
   result1.append(bigger/(bigger+smaller+0.00000001))
   result2.append(biggerAvg)

result1 = sorted(result1)
result2 = sorted(result2)

result1Mean = sum(result1)/samplesNumber
result2Mean = sum(result2)/samplesNumber

result1Low = result1[int(0.01 * samplesNumber)]
result1High = result1[int(0.99 * samplesNumber)]
result2Low = result2[int(0.01 * samplesNumber)]
result2High = result2[int(0.99 * samplesNumber)]

assert result2Mean <= result2High

print("\t".join([str(x) for x in [language, result1Mean, result2Mean, result1Low, result1High, result2Low, result2High]]))
print(result1High-result1Low)
# 
#library(tidyr)
#library(dplyr)
#library(ggplot2)
#data = data %>% rename(Entropy_Rate=Residual)
#plot = ggplot(data, aes(x=Memory, y=Entropy_Rate, group=Type, fill=Type, color=Type)) +
#    geom_point()
#ggsave(plot, file="North_Sami-entropy-memory.pdf")
#
#


