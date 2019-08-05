# ./python27 tradeoffPrepareTable_OnlyWordForms_BoundedVocab.py > ../results-table-word-level.tex

with open("../../ud_languages.txt", "r") as inFile:
   languages = inFile.read().strip().split("\n")

languages = sorted(list(set(languages)))

def readTSV(x):
    header = next(x).strip().split("\t")
    header = dict(zip(header, range(len(header))))
    data = [y.strip().split("\t") for y in x]
    if len(data[-1]) == 1 and data[-1][0] == "":
       del data[-1]
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
def g(frame, name, i):
    return frame[1][i][frame[0][name]]
def h(frame, line, name):
    return line[frame[0][name]]


#with open("tradeoff/effectSize.tsv", "r") as inFile:
#   effectSize = readTSV(inFile)

#with open("tradeoff/effectSize_diff.tsv", "r") as inFile:
#   effectSize_diff = readTSV(inFile)

with open("../../results/tradeoff/stats-onlyWordForms-boundedVocab_REAL.tsv", "r") as inFile:
   stats = readTSV(inFile)
languageKey_stats = dict([(h(stats, line, "Language"), line) for line in stats[1]])



with open("../corpusSizes.tsv", "r") as inFile:
  corpusSizes = readTSV(inFile) #dict([x.split("\t") for x in inFile.read().strip().split("\n")])
languageKey_corpusSizes = dict([(h(corpusSizes, line, "Language"), line) for line in corpusSizes[1]])


output = []

def pretty(x):
    x = str(x)
    if len(x) > 3:
        x = x[:-3] + "," + x[-3:]
    return x

for language in languages:
#   line = languageKey[language]



   components = [language.replace("_"," ").replace("-Adap", "")]
   components.append(pretty(h(corpusSizes,languageKey_corpusSizes[language], "TrainingSents")))
   components.append(pretty(h(corpusSizes,languageKey_corpusSizes[language], "HeldoutSents")))
   output.append(components)

if len(output)/2 * 2 < len(output):
    output.append(["","",""])

with open("../../corpusSizes.tex", "w") as outFile:
 for i in range(len(output)/2):
    here = output[i] + output[len(output)/2+i]
    print >> outFile, ("  &  ".join([str(x) for x in here]) + "  \\\\") # [10.25ex] \\hline" )


