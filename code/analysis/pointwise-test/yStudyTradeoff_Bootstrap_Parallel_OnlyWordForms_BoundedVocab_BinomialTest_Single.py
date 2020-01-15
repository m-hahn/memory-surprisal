# This one is mentioned in the paper source

import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--base_directory", dest="base_directory", type=str)
parser.add_argument("--suffix", dest="suffix", type=str, default="")

args=parser.parse_args()
print(args)

assert args.base_directory in ["word-level", "ngrams"]
#assert args.suffix == {"word-level" : "_onlyWordForms_boundedVocab", "ngrams" : ""}[args.base_directory]


language = args.language

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
with open("../../results/raw/"+args.base_directory+"/"+language+args.suffix+".tsv", "r") as inFile:
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

import math

xPoints = torch.FloatTensor([maximalMemory*x/40.0 for x in range(1,40)])

###############
# Interpolating

interpolatedByTypes = {}

for typ, mis in misByType.iteritems():  
  cumulativeMI = cumMIs[typ]
  cumulativeMemory = cumMems[typ]
  xBigger = (cumMems[typ].unsqueeze(2) > xPoints.unsqueeze(0).unsqueeze(0))
  xSmaller = (cumMems[typ].unsqueeze(2) <= xPoints.unsqueeze(0).unsqueeze(0))

  interpolated = torch.zeros(cumulativeMemory.size()[0], len(xPoints))
  foundValues = torch.zeros(cumulativeMemory.size()[0], len(xPoints))
  for j in range(0,len(xPoints)):
     condition = (xBigger[:,1:,j] * xSmaller[:,:-1,j]).float()
     memoryDifference = ((cumulativeMemory[:,1:] - cumulativeMemory[:,:-1]))
     slope = (xPoints[j] - cumulativeMemory[:,:-1]) / memoryDifference
     slope[memoryDifference == 0] = 1/2
     interpolation = (cumulativeMI[:,:-1] + slope * (cumulativeMI[:,1:] - cumulativeMI[:,:-1]))
     interpolated[:,j] = torch.sum(condition  * interpolation, dim=1)
     foundValues[:,j] = torch.sum(condition, dim=1)
  interpolatedByTypes[typ] = interpolated

import scipy.stats

for real in ["REAL_REAL", "GROUND"]:
    if real not in interpolatedByTypes:
         continue
    interpolated = interpolatedByTypes[real]
    median = interpolated.median(dim=0)[0]
    
    comparison = interpolatedByTypes["RANDOM_BY_TYPE"] < median.unsqueeze(0)
    comparisonReverse = interpolatedByTypes["RANDOM_BY_TYPE"] > median.unsqueeze(0)

    comparisonMean = comparison.float().sum(dim=0)
    comparisonReverseMean = comparisonReverse.float().sum(dim=0)

    for i in range(39):
       p1 = (scipy.stats.binom_test(x=comparisonMean[i], n=comparison.size()[0], alternative="greater"))
       print "\t".join(map(str,[language, real, i, float(xPoints[i]), float(comparisonMean[i]/comparison.size()[0]), p1]))



