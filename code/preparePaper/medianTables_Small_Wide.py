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


entries = []

for language in languages:
#   line = languageKey[language]
   components = [language.replace("_"," ").replace("-Adap", "")]
   #components.append( "\\multirow{4}{*}{\includegraphics[width=0.25\\textwidth]{neural/figures/"+language+"-entropy-memory.pdf}}")
   components.append( "\includegraphics[width=0.05\\textwidth]{neural/figures/"+language+"-listener-surprisal-memory-MEDIANS_onlyWordForms_boundedVocab_REAL.pdf}" )
   entries.append(components)

ROWS_PER_PART = 7
COLUMNS = 18
while len(entries) % COLUMNS != 0:
     entries.append(["",""])

outputRows = []
if True:
    languages = [x[0] for x in entries]
    images = [x[1] for x in entries]

    for i in range(int(len(entries)/COLUMNS)):
       outputRows.append( " & ".join(languages[i*COLUMNS:(i+1)*COLUMNS]))
       outputRows.append( " \\\\ ")
       outputRows.append( " & ".join(images[i*COLUMNS:(i+1)*COLUMNS]))
       outputRows.append( " \\\\ ")

part=0
with open("../../medians_small_wide.tex", "w") as outFile:
      for row in range(len(outputRows)):
#          if COLUMNS*(row) >= len(outputRows):
 #             break
  #        for i in range(COLUMNS):
   #          print(COLUMNS, row, i, COLUMNS*row+i, COLUMNS*row, len(outputRows))
             print(outputRows[row], file=outFile)


