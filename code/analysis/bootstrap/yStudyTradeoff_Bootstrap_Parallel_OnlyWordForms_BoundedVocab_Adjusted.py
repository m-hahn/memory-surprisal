import sys

language = sys.argv[1]
random = sys.argv[2] if len(sys.argv) > 2 else "RANDOM_BY_TYPE"
real = sys.argv[3] if len(sys.argv) > 3 else "REAL_REAL"
PRECISION = sys.argv[4] if len(sys.argv) > 4 else 0.01 

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
with open("../../results/raw/word-level/"+language+"_decay_after_tuning_onlyWordForms_boundedVocab.tsv", "r") as inFile:
     data = readTSV(inFile)




import torch

def g(frame, name, i):
    return frame[1][i][frame[0][name]]

matrix = [[g(data, "ModelID", i), g(data, "Distance", i), g(data, "ConditionalMI", i), g(data, "UnigramCE", i)] for i in range(len(data[1]))]

matrixByType = {}
misByType = {}
for i in range(len(data[1])):
    typ = g(data, "Type", i)
    if typ not in matrixByType:
        matrixByType[typ] = []
    matrixByType[typ].append(matrix[i])
for typ in matrixByType:
   matrixByType[typ] = torch.FloatTensor(matrixByType[typ]).view(-1, 19, 4)
   misByType[typ] = matrixByType[typ][:,:,2]

distance = torch.FloatTensor(range(1,20))

cumMIs = {}
cumMems = {}
cumInterpolated = {}
maximalMemory = 0
for typ, mis in misByType.iteritems():
  mask = torch.FloatTensor([[1 if j <= i else 0 for j in range(19)] for i in range(19)])
  cumulativeMI = torch.matmul(mis, mask.t())
  cumulativeMemory = torch.matmul(distance*mis, mask.t())
  cumMIs[typ] = cumulativeMI
  cumMems[typ] = cumulativeMemory
  maximalMemory = max(maximalMemory, float(torch.max(cumulativeMemory)))



for typ, mis in misByType.iteritems():  
  cumMIs[typ] = torch.cat([0*cumMIs[typ][:,-1].unsqueeze(1), cumMIs[typ], cumMIs[typ][:,-1].unsqueeze(1)], dim=1)
  cumMems[typ] = torch.cat([0*(cumMems[typ][:,-1].unsqueeze(1)), cumMems[typ], maximalMemory + 0*(cumMems[typ][:,-1].unsqueeze(1))], dim=1)



xPoints = torch.FloatTensor([maximalMemory*x/20.0 for x in range(1,20)])


for typ, mis in misByType.iteritems():  
  if typ not in [real, random]:
     continue
  cumulativeMI = cumMIs[typ]
  cumulativeMemory = cumMems[typ]
  xBigger = (cumMems[typ].unsqueeze(2) > xPoints.unsqueeze(0).unsqueeze(0))
  xSmaller = (cumMems[typ].unsqueeze(2) <= xPoints.unsqueeze(0).unsqueeze(0))


  interpolated = torch.zeros(cumulativeMemory.size()[0], len(xPoints))
  foundValues = torch.zeros(cumulativeMemory.size()[0], len(xPoints))
  for j in range(0,len(xPoints)):
     condition = (xBigger[:,1:,j] * xSmaller[:,:-1,j]).float()
     memoryDifference = ((cumulativeMI[:,1:] - cumulativeMI[:,:-1]))
     slope = (xPoints[j] - cumulativeMemory[:,:-1]) / (cumulativeMemory[:,1:] - cumulativeMemory[:,:-1])
     slope[memoryDifference == 0] = 1/2
     interpolation = (cumulativeMI[:,:-1] + slope * (cumulativeMI[:,1:] - cumulativeMI[:,:-1]))
     interpolated[:,j] = torch.sum(condition  * interpolation, dim=1)
     foundValues[:,j] = torch.sum(condition, dim=1)
  cumInterpolated[typ] = (interpolated, foundValues)


if real in cumInterpolated and random in cumInterpolated:
  typ1 = real
else:
  print("\t".join([str(x) for x in [language, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0]]))
  print(1.0)

  quit()
typ2 = random

import random


result1 = []
result2 = []

samplesNumber = 1000

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

   comparisonFavor1 = (interpolated1.unsqueeze(1) >= interpolated2.unsqueeze(0)).float()
   comparisonFavor2 = (interpolated2.unsqueeze(0) >= interpolated1.unsqueeze(1)).float()
   bothAreMeaningful = foundValues1.unsqueeze(1) * foundValues2.unsqueeze(0)
   comparableRange = (bothAreMeaningful.sum(2).unsqueeze(2)).sum(2) # length of the comparable range in each case
   comparableRange[comparableRange==0] = 1
   bigger = (comparisonFavor1 * bothAreMeaningful)
   smaller = (comparisonFavor2 * bothAreMeaningful)
   bigger = bigger.sum(2) / comparableRange
   smaller = smaller.sum(2) / comparableRange
   strictlyBiggerCounts = (  torch.sum(bigger == 1.0, dim=1))
   strictlySmallerCounts = ( torch.sum(smaller == 1.0, dim=1))
   biggerAverages = ( torch.sum(bigger, dim=1))
   smallerAverages = ( torch.sum(smaller, dim=1))
   bigger = (torch.sum(strictlyBiggerCounts).numpy() / len(strictlyBiggerCounts))
   smaller = (torch.sum(strictlySmallerCounts).numpy() / len(strictlySmallerCounts))
   biggerAvg = (torch.sum(biggerAverages).numpy() / len(biggerAverages)) / len(interpolated2)
   result1.append(bigger/(bigger+smaller+0.00000001))
   result2.append(biggerAvg)

result1 = sorted(result1)
result2 = sorted(result2)

result1Mean = sum(result1)/samplesNumber
result2Mean = sum(result2)/samplesNumber

result1Low = result1[int(PRECISION * samplesNumber)]
result1High = result1[int((1-PRECISION) * samplesNumber)]
result2Low = result2[int(PRECISION * samplesNumber)]
result2High = result2[int((1-PRECISION) * samplesNumber)]

assert result2Mean <= result2High

print("\t".join([str(x) for x in [language, result1Mean, result2Mean, result1Low, result1High, result2Low, result2High]]))
print(result1High-result1Low)

