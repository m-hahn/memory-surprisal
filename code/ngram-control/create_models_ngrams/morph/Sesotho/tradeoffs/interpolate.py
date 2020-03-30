# Better than yStudyTradeoff_Bootstrap_Parallel_OnlyWordForms_BoundedVocab_BinomialTest_Single_MaxControl.py by modeling the median of REAL

import sys





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
          print("NOT NUMERIC", column)
          vals=  [y[column] for y in data]
          print(vals)
      for i in range(len(data)):
          data[i][column] = vals[i]
    return (header, data)

script = "forWords_Japanese_RandomOrder_Normalized_FullData_Heldout.py"


try:
  with open("results.tsv", "r") as inFile:
     data_ = readTSV(inFile)
except IOError:
  print >> sys.stderr, ("ERROR nothing for this language? "+script)
  quit()

import torch

def g(frame, name, i):
    return frame[1][i][frame[0][name]]
   


scripts = set([g(data_, "Script", i) for i in range(len(data_[1]))])
with open("results_interpolated.tsv", "w") as outFile:
 print >> outFile, "\t".join(map(str, ["Script", "Type", "Run", "InterpPoint", "Memory", "MI"]))

 for script in scripts:
   data = (data_[0], [x for x in data_[1] if x[data_[0]["Script"]] == script])
   
   matrix = [[g(data, "Run", i), g(data, "Distance", i), g(data, "MI", i), g(data, "UnigramCE", i)] for i in range(len(data[1]))]
   
   matrixByType = {}
   misByType = {}
   for i in range(len(data[1])):
       typ = g(data, "Type", i)
       if typ not in matrixByType:
           matrixByType[typ] = []
       matrixByType[typ].append(matrix[i])
   MAX_DISTANCE = -1
   for typ in matrixByType:
      print(matrixByType[typ])
      tensorized = torch.FloatTensor(matrixByType[typ])
   #   print(tensorized[:,1].size())
      assert int(tensorized[:,1].min())==0
      MAX_DISTANCE = int(tensorized[:,1].max())+1
      matrixByType[typ] = tensorized.view(-1, MAX_DISTANCE, 4)
      #print(matrixByType[typ][0])
      misByType[typ] = matrixByType[typ][:,:,2]
   if MAX_DISTANCE == -1:
     print >> sys.stderr, ("ERROR nothing for this language? "+script)
     quit()
   distance = torch.FloatTensor(range(1,1+MAX_DISTANCE))
   
   cumMIs = {}
   cumMems = {}
   cumInterpolated = {}
   maximalMemory = 0
   for typ, mis in misByType.iteritems():
     mask = torch.FloatTensor([[1 if j <= i else 0 for j in range(MAX_DISTANCE)] for i in range(MAX_DISTANCE)])
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
     for i in range(0, interpolated.size()[0]):
         for j in range(0, len(xPoints)):
           print >> outFile, "\t".join(map(str, [script, typ, int(matrixByType[typ][i][0][0]), j, float(xPoints[j]), float(interpolated[i,j])]))
   
   
