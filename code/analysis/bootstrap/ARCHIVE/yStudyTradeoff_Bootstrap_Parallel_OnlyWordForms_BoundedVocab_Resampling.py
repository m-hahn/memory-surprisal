import sys

language = sys.argv[1]
random = sys.argv[2] if len(sys.argv) > 2 else "RANDOM_BY_TYPE"
real = sys.argv[3] if len(sys.argv) > 3 else "REAL_REAL"

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

#print("MAXIMAL MEMORY", maximalMemory)


for typ, mis in misByType.iteritems():  
  cumMIs[typ] = torch.cat([0*cumMIs[typ][:,-1].unsqueeze(1), cumMIs[typ], cumMIs[typ][:,-1].unsqueeze(1)], dim=1)
  cumMems[typ] = torch.cat([0*(cumMems[typ][:,-1].unsqueeze(1)), cumMems[typ], maximalMemory + 0*(cumMems[typ][:,-1].unsqueeze(1))], dim=1)

  #print(cumMIs[typ][0])


xPoints = torch.FloatTensor([maximalMemory*x/40.0 for x in range(1,40)])


for typ, mis in misByType.iteritems():  
  if typ not in [real, random]:
     continue
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
     memoryDifference = ((cumulativeMI[:,1:] - cumulativeMI[:,:-1]))
     slope = (xPoints[j] - cumulativeMemory[:,:-1]) / (cumulativeMemory[:,1:] - cumulativeMemory[:,:-1])
     slope[memoryDifference == 0] = 1/2
     interpolation = (cumulativeMI[:,:-1] + slope * (cumulativeMI[:,1:] - cumulativeMI[:,:-1]))
     interpolated[:,j] = torch.sum(condition  * interpolation, dim=1)
     foundValues[:,j] = torch.sum(condition, dim=1)
  #print(foundValues[0])
#  print(foundValues[0])
 # print("X", xPoints)
  #print("Inter", interpolated[0])
#  print("Mem", cumulativeMemory[0])
 # print("MI", cumulativeMI[0])
  cumInterpolated[typ] = (interpolated, foundValues)
#quit()


if real in cumInterpolated and random in cumInterpolated:
  typ1 = real
#elif "REAL" in cumInterpolated:
#  typ1 = "REAL"
else:
  print("\t".join([str(x) for x in [language, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0]]))
  print(1.0)

  quit()
typ2 = random

import random




if True:
   interpolated1 = cumInterpolated[typ1][0]
   interpolated2 = cumInterpolated[typ2][0]
   

   foundValues1 = cumInterpolated[typ1][1]
   foundValues2 = cumInterpolated[typ2][1]
 

   strictlyBiggerCounts = []
   strictlySmallerCounts = []
   biggerAverages = []
   smallerAverages = []

   comparisonFavor1 = (interpolated1.unsqueeze(1) >= interpolated2.unsqueeze(0)).float()
   comparisonFavor2 = (interpolated2.unsqueeze(0) >= interpolated1.unsqueeze(1)).float()
 #  print(comparison.mean())
   bothAreMeaningful = foundValues1.unsqueeze(1) * foundValues2.unsqueeze(0)
#   print("BOTH MEANINGFUL", bothAreMeaningful.mean())
   comparableRange = (bothAreMeaningful.sum(2).unsqueeze(2)).sum(2) # length of the comparable range in each case
   comparableRange[comparableRange==0] = 1
  # print(comparableRange.size())
   bigger = (comparisonFavor1 * bothAreMeaningful)
   smaller = (comparisonFavor2 * bothAreMeaningful)
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
   result1_real = (bigger/(bigger+smaller+0.00000001))
   result2_real = (biggerAvg)








result1 = []
result2 = []

samplesNumber = 10000

#print(cumInterpolated[typ1][0].size())
totalInterpolated = torch.cat([cumInterpolated[typ1][0], cumInterpolated[typ2][0]], dim=0)

for u in range(samplesNumber):
   perm = torch.randperm(totalInterpolated.size()[0])
   
   interpolated = totalInterpolated[perm]
#   print(interpolated.size())
   
   interpolated1 = interpolated[:cumInterpolated[typ1][0].size()[0]]
   interpolated2 = interpolated[cumInterpolated[typ1][0].size()[0]:]
 #  print(interpolated1.size())
  # print(interpolated2.size())
   

   foundValues1 = cumInterpolated[typ1][1]
   foundValues2 = cumInterpolated[typ2][1]
 

   strictlyBiggerCounts = []
   strictlySmallerCounts = []
   biggerAverages = []
   smallerAverages = []

   comparisonFavor1 = (interpolated1.unsqueeze(1) >= interpolated2.unsqueeze(0)).float()
   comparisonFavor2 = (interpolated2.unsqueeze(0) >= interpolated1.unsqueeze(1)).float()
 #  print(comparison.mean())
   bothAreMeaningful = foundValues1.unsqueeze(1) * foundValues2.unsqueeze(0)
#   print("BOTH MEANINGFUL", bothAreMeaningful.mean())
   comparableRange = (bothAreMeaningful.sum(2).unsqueeze(2)).sum(2) # length of the comparable range in each case
   comparableRange[comparableRange==0] = 1
  # print(comparableRange.size())
   bigger = (comparisonFavor1 * bothAreMeaningful)
   smaller = (comparisonFavor2 * bothAreMeaningful)
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
#print(sorted(result1))
print("\t".join([str(x) for x in [language, result1_real, result2_real, len([x for x in result1 if x >= result1_real])/float(samplesNumber),  len([x for x in result2 if x >= result2_real])/float(samplesNumber)  ]]))


