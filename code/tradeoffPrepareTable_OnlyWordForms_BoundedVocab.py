# tradeoff/effectSize.tsv


languages = []
languages += ["Arabic", "Catalan", "Czech", "Dutch", "Finnish", "French", "German", "Hindi", "Norwegian", "Spanish"]
languages += ["Basque", "Bulgarian", "Croatian", "Estonian", "Hebrew", "Japanese", "Polish", "Romanian", "Slovak", "Slovenian", "Swedish"]
languages += ["Afrikaans", "Chinese", "Danish", "Greek", "Hungarian",  "North_Sami", "Persian", "Serbian", "Turkish", "Ukrainian", "Vietnamese"]
languages += ["Amharic-Adap", "Armenian-Adap",  "Breton-Adap",  "Buryat-Adap", "Cantonese-Adap","Faroese-Adap", "Kazakh-Adap", "Kurmanji-Adap", "Naija-Adap","Thai-Adap", "Uyghur-Adap"]
languages += ["Bambara-Adap", "Erzya-Adap", "Maltese", "Latvian"]

languages += ["Indonesian", "Urdu", "Portuguese"]
	

languages = sorted(list(set(languages)))

with open("corpusSizes.tsv", "r") as inFile:
  corpusSizes = dict([x.split("\t") for x in inFile.read().strip().split("\n")])


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


with open("tradeoff/effectSize.tsv", "r") as inFile:
   effectSize = readTSV(inFile)

with open("tradeoff/effectSize_diff.tsv", "r") as inFile:
   effectSize_diff = readTSV(inFile)

with open("tradeoff/stats.tsv", "r") as inFile:
   stats = readTSV(inFile)
languageKey_stats = dict([(h(stats, line, "Language"), line) for line in stats[1]])


effectSize = (effectSize[0], sorted(effectSize[1], key=lambda x:x[0]))
languageKey = dict([(h(effectSize, line, "Language"), line) for line in effectSize[1]])

languageKey_diff = dict([(h(effectSize_diff, line, "Language"), line) for line in effectSize_diff[1]])

with open("tradeoff/listener-curve.tsv", "r") as inFile:
   listener_curve = readTSV(inFile)

languageKey_listener_curve = dict([(h(listener_curve, line, "language"), line) for line in listener_curve[1]])

for language in languages:
   line = languageKey[language]
   line_diff = languageKey_diff[language]
   line_listener = languageKey_listener_curve[language]
   line_stats = languageKey_stats[language]

   assert language == h(effectSize, line, "Language")

   satisfied = h(stats, line_stats, "RANDOM_BY_TYPE") >= 20 and h(stats, line_stats, "REAL_REAL") >= 20 and (h(listener_curve, line_listener, "result1High") - h(listener_curve, line_listener, "result1Low") <= 0.15) and (h(effectSize, line, "hi2") - h(effectSize, line, "lo2") <= 0.3) # and (h(effectSize_diff, line_diff, "hi1") - h(effectSize_diff, line_diff, "lo1") <= 0.2)

   components = [language.replace("_"," ")+("*" if not satisfied else "")]
   components.append( "\\multirow{4}{*}{\includegraphics[width=0.25\\textwidth]{figures/"+language+"-entropy-memory.pdf}}")
   components.append( "\\multirow{4}{*}{\includegraphics[width=0.25\\textwidth]{figures/"+language+"-listener-surprisal-memory.pdf}}" )
   components.append("$D_x$")
   components.append(round(h(effectSize, line, "mean2"),2))
   components.append( "".join(map(str,["[", round(h(effectSize, line, "lo2"),2),", ", round(h(effectSize, line, "hi2"),2) , "]"])))
   print("  &  ".join([str(x) for x in components]) + "  \\\\ " )


   components = [corpusSizes[language]]
   components.append("")
   components.append("")
   components.append("$B_x-A_x$")
   components.append(round(h(effectSize_diff, line_diff, "mean1"),2))
   components.append( "".join(map(str,["[", round(h(effectSize_diff, line_diff, "lo1"),2),", ", round(h(effectSize_diff, line_diff, "hi1"),2) , "]"])))
   print("  &  ".join([str(x) for x in components]) + "  \\\\ " )


   components = [""] #h(stats, line_stats, "RANDOM_BY_TYPE")]
   components.append("")
   components.append("")
   components.append("$E_x$")
   components.append(round(h(effectSize_diff, line_diff, "mean2"),2))
   components.append( "".join(map(str,["[", round(h(effectSize_diff, line_diff, "lo2"),2),", ", round(h(effectSize_diff, line_diff, "hi2"),2) , "]"])))
   print("  &  ".join([str(x) for x in components]) + "  \\\\ " )

   components = [""] #h(stats, line_stats, "REAL_REAL")]
   components.append("")
   components.append("")
   components.append("$W_x$")
   components.append(round(h(listener_curve, line_listener, "result1Mean"),2))
   components.append( "".join(map(str,["[", round(h(listener_curve, line_listener, "result1Low"),2),", ", round(h(listener_curve, line_listener, "result1High"),2) , "]"])))
   print("  &  ".join([str(x) for x in components]) + "  \\\\ [10.25ex] \\hline" )


