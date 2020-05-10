# ./python27 tradeoffPrepareTable_OnlyWordForms_BoundedVocab.py > ../results-table-word-level.tex

with open("../ud_languages.txt", "r") as inFile:
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




with open("../../results/tradeoff/listener-curve-onlyWordForms-boundedVocab_REAL.tsv", "r") as inFile:
   listener_curve = readTSV(inFile)

languageKey_listener_curve = dict([(h(listener_curve, line, "language"), line) for line in listener_curve[1]])

entries = []

for language in languages:
#   line = languageKey[language]

   components = [language.replace("_"," ").replace("-Adap", "")]
   components.append( "\includegraphics[width=0.25\\textwidth]{neural/figures/"+language+"-listener-surprisal-memory-QUANTILES_onlyWordForms_boundedVocab_noAssumption.pdf}" )
   entries.append(components) #   print >> outFile, ("  &  ".join([str(x) for x in components]) + "  \\\\ [10.25ex] \\hline" )

ROWS_PER_PART = 5
COLUMNS = 4
entries += [["",""] for x in range((COLUMNS-(len(entries)%COLUMNS))%COLUMNS)]

outputRows = []
if True:
    languages = [x[0] for x in entries]
    images = [x[1] for x in entries]

    for i in range(len(entries)/COLUMNS):
       outputRows.append( " & ".join(languages[i*COLUMNS:(i+1)*COLUMNS]))
       outputRows.append( " \\\\ ")
       outputRows.append( " & ".join(images[i*COLUMNS:(i+1)*COLUMNS]))
       outputRows.append( " \\\\ ")

for part in range(len(entries)/(COLUMNS*ROWS_PER_PART)+1):
  with open("../../writeup/tables/quantiles_noAssumption_"+str(part)+".tex", "w") as outFile:
      for row in range(part*ROWS_PER_PART, (part+1)*ROWS_PER_PART):
          if 4*row >= len(outputRows):
              break
          print >> outFile, outputRows[4*row]
          print >> outFile, outputRows[4*row+1]
          print >> outFile, outputRows[4*row+2]
          print >> outFile, outputRows[4*row+3]



