import sys

language = sys.argv[1]
real_type = sys.argv[2] if len(sys.argv) > 2 else "REAL_REAL"

def readTSV(x):
    header = next(x).strip().split("\t")
    header = dict(zip(header, range(len(header))))
    data = [y.strip().split("\t") for y in x]
    if len(data) < 3:
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
  with open("../results/raw/word-level/"+language+"_after_tuning_onlyWordForms_boundedVocab.tsv", "r") as inFile:
     data = readTSV(inFile)
except IOError:
   print("\t".join(map(str, [language, 0,0  ])))
   quit()
#print(len(data))



#data = data %>% group_by(ModelID) %>% mutate(CumulativeMemory = cumsum(Distance*ConditionalMI), CumulativeMI = cumsum(ConditionalMI))

def g(frame, name, i):
    return frame[1][i][frame[0][name]]

matrix = [[g(data, "ModelID", i) ] for i in range(len(data[1]))]

matrixByType = {}
misByType = {}
#unigramCEByType = {}
for i in range(len(data[1])):
    typ = g(data, "Type", i)
#    print(i,typ, len(data[1]))
    if typ not in matrixByType:
        matrixByType[typ] = []
    matrixByType[typ].append(matrix[i])

print("\t".join(map(str, [language, len(matrixByType.get(real_type,[])), len(matrixByType.get("RANDOM_BY_TYPE",[]))  ])))

