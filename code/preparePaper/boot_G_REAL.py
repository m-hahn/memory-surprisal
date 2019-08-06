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





with open("../../results/tradeoff/listener-curve-onlyWordForms-boundedVocab_REAL.tsv", "r") as inFile:
   listener_curve = readTSV(inFile)

languageKey_listener_curve = dict([(h(listener_curve, line, "language"), line) for line in listener_curve[1]])



entries = []

for language in languages:
   line_listener = languageKey_listener_curve[language]
   components = [language.replace("_"," ").replace("-Adap", "")]
   #components.append( "\\multirow{4}{*}{\includegraphics[width=0.25\\textwidth]{neural/figures/"+language+"-entropy-memory.pdf}}")
   components.append( round(h(listener_curve, line_listener, "result1Mean"), 2) )
   components.append( round(h(listener_curve, line_listener, "result1Low"), 2) )
   components.append( round(h(listener_curve, line_listener, "result1High"), 2) )

   entries.append([str(x) for x in components])

output = entries

if len(output)/2 * 2 < len(output):
    output.append(["","",""])


with open("../../boot_g_REAL.tex", "w") as outFile:
 for i in range(len(output)/2):
    here = output[i] + output[len(output)/2+i]
    print >> outFile, ("  &  ".join([str(x) for x in here]) + "  \\\\") # [10.25ex] \\hline" )


