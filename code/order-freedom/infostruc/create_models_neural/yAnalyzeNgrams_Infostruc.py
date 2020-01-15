import os

PATH = "memory-need-neural-wordforms_infostruc"

path = "/u/scr/mhahn/deps/"+PATH+"/"
files = os.listdir(path)
import sys


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sortBy", type=int)
parser.add_argument("--horizon", type=int, default=20)
parser.add_argument("--language", dest="language", type=str)
parser.add_argument("--restrictToFinished", type=bool)
parser.add_argument("--onlyOptimized", type=bool, default=True)


args=parser.parse_args()
print(args)


def f(a):
   x = list(map(float,a.split(" ")))[:args.horizon]

   decay = [(x[(i-1 if i>0 else 0)]-x[i]) for i in range(len(x))]
   assert len(decay) == args.horizon

   memory = sum([i*(x[(i-1 if i>0 else 0)]-x[i]) for i in range(len(x))])

   residual = x[args.horizon-1]
   balanced = sum([i*(x[(i-1 if i>0 else 0)]-x[i]) for i in range(args.horizon)]) + args.horizon*x[args.horizon-1]

   mi = x[0] - x[args.horizon-1]

   unigramCE = x[0]
   return balanced, memory, residual, mi, decay, unigramCE

resultsPerType = {}

finalShib = { "Russian" : " Russian 0.4 150 256 3 0.05 ", "Erzya-Adap" : "Erzya-Adap 0.4 200 128 1 0.05 ", "Bambara-Adap" : " Bambara-Adap 0.4 300 256 1 0.1 ", "Polish-LFG" : "Polish-LFG 0.2 200 512 1 0.05 ", "North_Sami" : " North_Sami 0.4 300 256 1 0.1 ", "Maltese" : (" Maltese 0.35 300 64 1 0.1 ", " 0.1 2 0.0 20"),
       "Arabic" : " Arabic 0.4 50 512 2 0.05 ",
       "Catalan" : " Catalan 0.45 100 512 1 0.01 ",
       "Czech" : " Czech 0.05 200 256 3 0.05 ",
       "Dutch" : " Dutch 0.3 50 512 2 0.1 ",
       "Finnish" : " Finnish 0.35 200 1024 1 0.1 ",
       "French" : " French 0.25 100 1024 3 0.05 ",
       "Hindi" : " Hindi 0.2 150 256 2 0.01 ",
       "Norwegian" : " Norwegian 0.25 200 512 1 0.01 ",
       "Spanish" : " Spanish 0.3 100 1024 3 0.1 ",
       "Basque" : " Basque 0.4 200 64 1 0.1 ",
       "Bulgarian" : " Bulgarian 0.2 300 512 1 0.1 ",
       "Croatian" : " Croatian 0.4 50 1024 2 0.1 ",
       "Estonian" : " Estonian 0.35 200 1024 1 0.1 ",
       "Hebrew" : " Hebrew 0.25 150 128 1 0.1 ",
       "Japanese" : " Japanese 0.05 100 1024 1 0.1 ",
       "Polish" : " Polish 0.4 150 64 1 0.1 ",
       "Romanian" : " Romanian 0.45 200 1024 1 0.05 ",
       "Slovak" : " Slovak 0.0 50 256 1 0.1 ",
       "Slovenian" : " Slovenian 0.1 150 512 1 0.1 ",
       "Swedish" : " Swedish 0.15 150 512 1 0.1 ",
       "Afrikaans" : " Afrikaans 0.4 300 1024 1 0.1 ",
       "Chinese" : " Chinese 0.45 200 1024 1 0.1 ",
       "Danish" : " Danish 0.45 300 512 1 0.1 ",
       "Greek" : " Greek 0.1 300 256 1 0.1 ",
       "Hungarian" : " Hungarian 0.35 200 128 1 0.1 ",
       "Persian" : " Persian 0.45 100 1024 1 0.1 ",
       "Serbian" : " Serbian 0.35 200 256 1 0.1 ",
       "Tamil" : " Tamil 0.0 50 512 2 0.1 ",
       "Turkish" : " Turkish 0.3 100 128 1 0.1 ",
       "Ukrainian" : " Ukrainian 0.45 200 512 1 0.1 ",
       "Vietnamese" : " Vietnamese 0.15 150 128 1 0.1 ",
       "Amharic-Adap" : " Amharic-Adap 0.4 200 64 1 0.1 ",
       "Armenian-Adap" : " Armenian-Adap 0.1 300 512 1 0.1 ",
       "Breton-Adap" : " Breton-Adap 0.25 150 128 1 0.1 ",
       "Buryat-Adap" : " Buryat-Adap 0.0 50 128 1 0.05 ",
       "Cantonese-Adap" : " Cantonese-Adap 0.3 200 128 1 0.1 ",
       "Faroese-Adap" : " Faroese-Adap 0.35 150 64 1 0.1 ",
       "Kazakh-Adap" : " Kazakh-Adap 0.45 50 128 1 0.1 ",
       "Kurmanji-Adap" : " Kurmanji-Adap 0.4 300 64 1 0.1 ",
       "Naija-Adap" : " Naija-Adap 0.2 100 256 1 0.1 ",
       "Thai-Adap" : " Thai-Adap 0.45 150 128 1 0.1 ",
       "Uyghur-Adap" : " Uyghur-Adap 0.0 200 64 1 0.001 ",
       "Latvian" : " Latvian 0.45 300 64 1 0.1 ",
       "Indonesian" : " Indonesian 0.25 300 128 1 0.1 ",
       "Urdu" : " Urdu 0.2 50 512 1 0.1 ",
       "Portuguese" : " Portuguese 0.3 200 1024 3 0.1 ", # corrected (there previously was a bug here)
       "German" : " German 0.3 100 512 1 0.05 ",
       "Italian" : " Italian 0.4 300 256 1 0.05 " ,
       "English" : " English 0.15 150 1024 2 0.1 ",
       "Korean" : " Korean 0.1 300 1024 1 0.05 ", 
       "Wolof_2.4" : " Wolof_2.4 0.3 50 64 1 0.1",
       "Welsh-Adap_2.4" : " Welsh-Adap_2.4 0.45 300 256 1 0.1",
       "Lithuanian_2.4" : " Lithuanian_2.4 0.4 50 256 2 0.001"}




types = [" REAL_REAL ", "RANDOM_BY_TYPE ", " GROUND ", " RANDOM_INFOSTRUC ", " GROUND_INFOSTRUC "]


averageUnigramCE = [0.0,0]

for fileName in files:
  if args.language not in fileName:
       continue
  if not fileName.startswith("estimates"):
     continue
  with open(path+fileName, "r") as inFile:
     result = inFile.read().strip().split("\n")
     if len(result) < 4:
       continue
     if "PARAMETER_SEARCH" in result:
       continue
     if not "SaveLast" in result[0]:
         continue
     if " "+args.language+" " not in result[0]:
        continue
     if args.onlyOptimized: # changed this line (March 8, 2019)
       if type(finalShib[args.language]) == type((1,2)):
         if not all([x in result[0] for x in  finalShib[args.language]]): # not in result[0] or finalShib[language][1] not in result[0]):
           continue
       else:
         if not finalShib[args.language] in result[0]: # all([x in result[0] for x in  ]): # not in result[0] or finalShib[language][1] not in result[0]):
           continue

     typeOfResult = filter(lambda x:x in result[0], types)[0][:-1]
     if len(result) < 3:
         continue
     if typeOfResult not in resultsPerType:
        resultsPerType[typeOfResult] = []
     result[1] = list(map(lambda x:x if x=="" else float(x), result[1].replace("[","").replace("]","").split(" ")))
     if args.restrictToFinished:
       if len(result[1]) == 1 or result[1][-1] < result[1][-2]: # this had not been stopped by early stopping
          continue
     balanced, memory, residual, mi, decay, unigramCE  = f(result[2])
     duration = len(result[1])
     uid = float(result[4].split(" ")[1]) if "Exponent" in fileName else "NA"


     q = fileName.index("_model_")
     q = fileName[q+7:]
     idOfModel = str(int(q[:q.index("_")]))

     averageUnigramCE[0] += unigramCE
     averageUnigramCE[1] += 1


     resultsPerType[typeOfResult].append([balanced, memory, residual, result[0], duration, uid, idOfModel, mi, memory/mi, decay, unigramCE])




if averageUnigramCE[1] == 0:
    print("no results")
    print("/u/scr/mhahn/"+args.language+"_after_tuning_onlyWordForms_boundedVocab_infostruc.tsv")
    print("/u/scr/mhahn/"+args.language+"_decay_after_tuning_onlyWordForms_boundedVocab_infostruc.tsv")
    quit()

averageUnigramCE = averageUnigramCE[0] / averageUnigramCE[1]


outpath1 = "/u/scr/mhahn/"+args.language+"_after_tuning_onlyWordForms_boundedVocab_infostruc.tsv"
outpath2 = "/u/scr/mhahn/"+args.language+"_decay_after_tuning_onlyWordForms_boundedVocab_infostruc.tsv"

header = ["Model", "Language", "Code", "Drop1", "Emb", "Dim", "Layers", "lr", "Type", "Drop2", "Batch", "Length", "Balanced", "Memory", "Residual", "Duration", "NonUniformity", "ModelID", "MI"]
headerDecay = ["Model", "Language", "Code", "Type", "Distance", "ConditionalMI", "TotalMI", "ModelID", "UnigramCE"]
with open(outpath1, "w") as outFile:
 with open(outpath2, "w") as outFileDecay:

  print >> outFile, "\t".join(header)
  print >> outFileDecay, "\t".join(headerDecay)
  
  for typeOfResult in types:
     if len(resultsPerType.get(typeOfResult[:-1], [])) == 0:
        continue
     print
     print typeOfResult[:-1]
     print "\n".join(map(lambda x:str(x[:-1]),sorted(resultsPerType.get(typeOfResult[:-1], []), key=lambda x:x[sortBy])))
     for rand in resultsPerType.get(typeOfResult[:-1], []):
        parameters = rand[3].split(" ")
        if len(parameters) + 7 > len(header):
           parameters = parameters[:len(header)-7]
        assert len(parameters) + 7 == len(header), (len(parameters)+7, len(header))

        unigramCEHere = rand[10]
        rand[9][1] += (-unigramCEHere + averageUnigramCE)
        rand[1] += (-unigramCEHere + averageUnigramCE)
        rand[10] = averageUnigramCE

        print >> outFile, "\t".join(map(str,parameters + [rand[0], rand[1], rand[2], rand[4], rand[5], rand[6], rand[7]]))
        for i in range(1,args.horizon):
           print >> outFileDecay, "\t".join(map(str,[parameters[0], parameters[1], parameters[2], parameters[8], i, max(0, rand[9][i]), rand[7], rand[6], rand[10]]))
 
print outpath1
print outpath2
