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
with open("../../results/raw/word-level/"+language+"_decay_after_tuning_onlyWordForms_boundedVocab.tsv", "r") as inFile:
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

import scipy.stats
import math
import statsmodels.stats.proportion

#binom = 


# Assuming unimodality of RANDOM distribution and that REAL median has been estimated very precisely, get confidence bound on the quantile of REAL in the RANDOM distribution

medians = {}

for real in ["REAL_REAL", "GROUND"]:
  medians[real] = interpolatedByTypes[real].median(dim=0)[0]
#  print(medians[real])
 # print(medians[real].size())
  
  for i in range(39):
     hereRandom = torch.sort(interpolatedByTypes["RANDOM_BY_TYPE"][:,i])[0]

     worseRandomCount = float((hereRandom < medians[real][i]).sum())
     sameRandomCount = float((hereRandom == medians[real][i]).sum())
     betterRandomCount = float((hereRandom > medians[real][i]).sum())
#     print(worseRandomCount, sameRandomCount, betterRandomCount)

     # The goal here is to provide a confidence lower bound on worseRandomCount
     if worseRandomCount == 0:
         print "\t".join(map(str,[language, real, i, 0.0, 0.0, float(xPoints[i])]))
#                              ["Language", "Type", "Position", "LowerConfidenceBound", "Level", "Memory"])

         bound = 0
     else:
   #     print(hereRandom.size())
        largestPossibleValue = xPoints[i]
        #print(i, largestPossibleValue, medians[real][i], hereRandom)
        # if the median is at most the best sample
        #unaccountedForTarget = 1-math.pow(targetLevel, 1/(float(hereRandom.size()[0])))
        randomBestWorse = float(hereRandom[int(worseRandomCount)-1])
        assert randomBestWorse < float(medians[real][i])
#        print((largestPossibleValue, medians[real][i] , randomBestWorse,   (largestPossibleValue - medians[real][i] + 0.0001) / (largestPossibleValue - randomBestWorse + 0.0001)))

        for unaccountedFor in [float(x)/200 for x in range(1,200)]:
#          print scipy.stats.binom_test(x=1, n=10, p=0.5, alternative="greater")
          assert unaccountedFor > 0
          assert unaccountedFor < 1
          
          percentile = unaccountedFor 
          if sameRandomCount+betterRandomCount == 0:
             p1 = math.pow(1-unaccountedFor, worseRandomCount)
          else:
             p1 = 1-(scipy.stats.binom_test(x=sameRandomCount+betterRandomCount, n=worseRandomCount+sameRandomCount+betterRandomCount, p=unaccountedFor, alternative="greater"))
          if p1 < 0.05:
             print "\t".join(map(str,[language, real, i, 1-float(percentile), p1, float(xPoints[i])]))
#             print("PERCENTILE", worseRandomCount / (worseRandomCount + sameRandomCount + betterRandomCount))
             assert 1-float(percentile) <= worseRandomCount / (worseRandomCount + sameRandomCount + betterRandomCount)
             break


