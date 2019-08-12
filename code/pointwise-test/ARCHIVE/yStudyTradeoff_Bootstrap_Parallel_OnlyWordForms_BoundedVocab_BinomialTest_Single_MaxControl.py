# TODO what is this doing?

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
with open("../results/raw/word-level/"+language+"_decay_after_tuning_onlyWordForms_boundedVocab.tsv", "r") as inFile:
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

for real in ["REAL_REAL", "GROUND"]:
    interpolated = interpolatedByTypes[real]
   # print(interpolated.size())
    

#    print(comparisonMean)
    for i in range(39):
       
       hereReal = torch.sort(interpolated[:,i])[0]      
#       print("Real", hereReal.size(), "Random", interpolatedByTypes["RANDOM_BY_TYPE"][:,i].size())

       hereRandom = interpolatedByTypes["RANDOM_BY_TYPE"][:,i]
       comparison = hereRandom.unsqueeze(0) < hereReal.unsqueeze(1)
#       comparisonReverse = hereRandom.unsqueeze(0) > hereReal.unsqueeze(1)

       # for each REAL model, find how many baselines it beats
       realCount = hereReal.size()[0]
       randomCount = hereRandom.size()[0]

       comparisonMeanPerREAL = comparison.sum(dim=1)
#       print(comparisonMeanPerREAL.size())
       # prob(median is in the cell BELOW (worse than) this value)
       totalPValue = 0
       totalMean = 0
       #totalLower = 0
       #totalUpper = 0
       cis = [None for _ in range(realCount)]
       probMedians = [None for _ in range(realCount)]
       for j in range(realCount):
          comparisonMeanForThisREAL = float(comparisonMeanPerREAL[j])
          p = (scipy.stats.binom_test(x=comparisonMeanForThisREAL, n=randomCount, alternative="greater")) 
          # prob that (j-1) are worse than median, realCount-j+1 are >= than the median
          probMedian = scipy.stats.binom.pmf(j, realCount, 0.5)
          probMedians[j] = probMedian
          totalPValue += probMedian * p
          totalMean += probMedian * comparisonMeanForThisREAL/randomCount
          cis[j] = (statsmodels.stats.proportion.proportion_confint(count=comparisonMeanForThisREAL,nobs=randomCount, alpha=0.001, method="jeffreys"))

       bestCI = (0, 1, 1.0)
       for j in range(realCount):
        for k in range(j+1, realCount):
           coverage = sum(probMedians[j:k+1]) # TODO check the precise definition
           lower = min([x[0] for x in cis[j:k+1]])
           upper = max([x[1] for x in cis[j:k+1]])
           if coverage > 1-0.05 and (upper-lower) < bestCI[1] - bestCI[0]:
              bestCI = (lower, upper, coverage*(1-0.001))

       # TODO don't trust the CI
       print "\t".join(map(str,[language, real, i, totalMean, totalPValue, bestCI[0], bestCI[1], bestCI[2]]))





       #quit()

#       comparison = interpolatedByTypes["RANDOM_BY_TYPE"] < minReal.unsqueeze(0)
#       comparisonReverse = interpolatedByTypes["RANDOM_BY_TYPE"] > minReal.unsqueeze(0)
#    
#     #  print(comparison.size())
#       comparisonMean = comparison.float().sum(dim=0)
#       comparisonReverseMean = comparisonReverse.float().sum(dim=0)
#    
#
#
#       p1 = (scipy.stats.binom_test(x=comparisonMean[i], n=comparison.size()[0])) + math.pow(2, -interpolated.size()[0])
#       p2 = (scipy.stats.binom_test(x=comparisonReverseMean[i], n=comparison.size()[0])) + math.pow(2, -interpolated.size()[0])
#



#  mis = list(interpolated[:,-5].numpy())
#  for i in range(len(mis)):
#     print("\t".join(map(str,[language, typ, float(xPoints[-5]), mis[i]])))
#
