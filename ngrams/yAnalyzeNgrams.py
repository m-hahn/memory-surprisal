import os

PATH = "memory-need-ngrams"

path = "/u/scr/mhahn/deps/"+PATH+"/"
files = os.listdir(path)
import sys
sortBy = int(sys.argv[1])
#horizon = int(sys.argv[2])
language = sys.argv[2]
#restrictToFinished = (sys.argv[4] == "True")
#onlyOptimized = (not (len(sys.argv) > 5 and sys.argv[5] == "False"))
def f(a):
   
   x = list(map(float,a.split(" ")))
   horizon = len(x)

   decay = [(x[(i-1 if i>0 else 0)]-x[i]) for i in range(len(x))]
   assert len(decay) == horizon

   memory = sum([i*(x[(i-1 if i>0 else 0)]-x[i]) for i in range(len(x))])
#   memory = sum([i*max(0, x[(i-1 if i>0 else 0)]-x[i]) for i in range(len(x))])

   residual = x[horizon-1]
   # memory + aggregate surprisal -- this measures (as a lower bound) how many bits are required to encode the future, when the past is given (should  be low)
   balanced = sum([i*(x[(i-1 if i>0 else 0)]-x[i]) for i in range(horizon)]) + horizon*x[horizon-1]

   mi = x[0] - x[horizon-1]

   unigramCE = x[0]
   return balanced, memory, residual, mi, decay, unigramCE

resultsPerType = {}

#resultsReal = []
#resultsRandom = []
#resultsRandomByType = []
#resultsRandom_ST = []
#resultsRandomByType_ST = []

finalShib = {"Estonian" : "Estonian 0.16 200 256 1 0.0055",
             "Vietnamese" : "vi 0.33 50 128 1 0.02",
             "Slovenian" : "Slovenian 0.16 100 256 1 0.01",
             "Finnish" : "Finnish 0.16 100 512 1 0.01",
             "Polish" : "Polish 0.33 150 512 3 0.05",
             "Czech" : "Czech 0.2 250 512 2 0.05",
             "Swedish" : "Swedish 0.33 150 256 1 0.01",
             "Hindi" : "Hindi 0.25 300 256 3 0.01",
             "Basque" : "",
             "German" : "German 0.2 150 512 2 0.01",
             "Arabic" : "Arabic 0.2 300 1024 3 0.05",
             "Turkish" : "Turkish 0.3 250 512 1 0.05",
             "Spanish" : "Spanish 0.25 200 1024 3 0.01",
             "Latvian" : "Latvian 0.35 100 128 2 0.005",
             "Indonesian" : "Indonesian 0.45 50 256 2 0.005",
             "Hungarian" : "Hungarian 0.35 100 128 1 0.005",
             "Persian" : "Persian 0.4 300 256 2 0.005",
             "Croatian" : "Croatian 0.2 50 128 3 0.005",
             "French" : "French 0.4 100 1024 2 0.005",
             "Chinese" : "Chinese 0.3 300 128 3 0.005",
             "Serbian" : "Serbian 0.45 200 256 1 0.01",
             "Slovak" :  "Slovak 0.2 150 128 1 0.005",
             "North_Sami" : " North_Sami 0.25 50 64 1 0.01 ",
             "Irish" : " Irish 0.1 200 64 2 0.001 ",
             "Lithuanian" : "Lithuanian 0.15 200 64 1 0.001 ",
             "Armenian" : "Armenian 0.3 150 64 1 0.05 ",
             "Ukrainian" : "Ukrainian 0.25 100 128 1 0.005 ",
             "Armenian-Adap" : "Armenian-Adap 0.45 200 64 2 0.005 ",
             "Greek" : " Greek 0.35 300 512 3 0.01 ",
             "Uyghur-Adap" : "Uyghur-Adap 0.4 50 128 1 0.01 ",
             "Breton-Adap" : " Breton-Adap 0.4 300 64 3 0.005 ",
             "Thai-Adap" : " Thai-Adap 0.35 100 128 3 0.001 ",
             "Tamil" : " Tamil 0.4 150 128 2 0.001 ",
             "Faroese-Adap" : "Faroese-Adap 0.4 100 64 2 0.005 ",
             "Buryat-Adap" : " Buryat-Adap 0.15 150 128 3 0.001 ",
             "Naija-Adap" : "Naija-Adap 0.45 50 64 2 0.005",
             "Cantonese-Adap" : "Cantonese-Adap 0.45 150 64 3 0.001",
             "Japanese" : " Japanese 0.45 300 512 1 0.01 ",
             "Hebrew" : " Hebrew 0.2 50 256 2 0.01 ",
             "Marathi" : " Marathi 0.25 150 64 1 0.001 ",
             "Dutch" : " Dutch 0.1 50 128 1 0.01 ",
             "Kazakh-Adap" : " Kazakh-Adap 0.4 300 128 1 0.005 ",
             "Amharic-Adap" : " Amharic-Adap 0.35 100 64 1 0.005 ",
             "Afrikaans" : " Afrikaans 0.45 50 512 1 0.005 ",
             "Bulgarian" : "Bulgarian 0.0 300 256 1 0.001",
             "Danish" : "Danish 0.05 150 128 3 0.001",
             "Catalan" : "Catalan 0.3 150 256 2 0.005",
             "Belarusian" : "Belarusian 0.45 100 64 1 0.01",
             "Norwegian" : "Norwegian 0.45 300 256 2 0.005",
             "Romanian" : "Romanian 0.3 200 256 1 0.005",
             "Kurmanji-Adap" : "Kurmanji-Adap 0.35 200 256 2 0.005",
             "Bambara-Adap" : "Bambara-Adap 0.2 100 64 1 0.005 ",
             "Erzya-Adap" : "Erzya-Adap 0.2 50 64 2 0.005 ",
             "Maltese" : " Maltese 0.45 50 512 1 0.005 "
          }



filenames = [x for x in os.listdir("/u/scr/mhahn/deps/memory-need-ngrams/") if x.startswith("search-"+language+"_yWithMo") and len(open("/u/scr/mhahn/deps/memory-need-ngrams/"+x, "r").read().split("\n"))>=30]
if len(filenames) == 0:
   quit()
with open("/u/scr/mhahn/deps/memory-need-ngrams/"+filenames[0], "r") as inFile:
    params = next(inFile).strip().split("\t")[2:]
    assert len(params) == 4
  
version = "yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py"
argumentNames = ["alpha", "gamma", "delta", "cutoff"]

 
params2 = []
for i in range(len(params)):
  params2.append("--"+argumentNames[i])
  params2.append(params[i])
correctParameters = " ".join(params2)


types = [" REAL_REAL ", "RANDOM_MODEL ", "RANDOM_BY_TYPE "]

for fileName in files:
  if language not in fileName:
       continue
  if not fileName.startswith("estimates"):
     continue

  with open(path+fileName, "r") as inFile:
     result = inFile.read().split("\n")
     if (correctParameters not in result[0]):
         continue
     typeOfResult = filter(lambda x:x in result[0], types)[0][:-1]
     if len(result) < 3:
         continue
     if typeOfResult not in resultsPerType:
        resultsPerType[typeOfResult] = []
     result[1] = list(map(lambda x:x if x=="" else float(x), result[1].replace("[","").replace("]","").split(" ")))
     balanced, memory, residual, mi, decay, unigramCE  = f(result[2])
     duration = len(result[1])
     uid = float(result[4].split(" ")[1]) if "Exponent" in fileName else "NA"

     q = fileName.index("_model_")
     q = fileName[q+7:]
     idOfModel = str(int(q[:q.index("_")]))



     resultsPerType[typeOfResult].append({"Parameters" : result[0], "Memory" : memory, "Residual" : residual, "ModelID" : idOfModel, "TotalMI" : mi, "Decay" : decay, "UnigramCE" : unigramCE})



header = ["Type", "Memory", "Residual", "ModelID", "TotalMI"]
headerDecay = ["Type", "Distance", "ConditionalMI", "TotalMI", "ModelID", "UnigramCE"]
with open("/u/scr/mhahn/"+language+"_ngrams_after_tuning.tsv", "w") as outFile:
 with open("/u/scr/mhahn/"+language+"_ngrams_decay_after_tuning.tsv", "w") as outFileDecay:

  print >> outFile, "\t".join(header)
  print >> outFileDecay, "\t".join(headerDecay)
  
  for typeOfResult in types:
     if len(resultsPerType.get(typeOfResult[:-1], [])) == 0:
        continue
     print
     print typeOfResult[:-1]
#     print "\n".join(map(lambda x:str(x[:-1]),sorted(resultsPerType.get(typeOfResult[:-1], []), key=lambda x:x[sortBy])))
     for rand in resultsPerType.get(typeOfResult[:-1], []):
        parameters = rand["Parameters"].split(" ")
        print(rand) 
        rand["Type"] = parameters[4]
        outLine = [rand[x] for x in header]
        assert len(outLine) == len(header)
        print >> outFile, "\t".join(map(str,outLine))


        for i in range(1,len(rand["Decay"])):
           rand["Distance"] = i
           rand["ConditionalMI"] = max(0, rand["Decay"][i])
           print >> outFileDecay, "\t".join([str(rand[x]) for x in headerDecay]) #map(str,[parameters[0], parameters[1], parameters[2], parameters[8], i, max(0, rand[9][i]), rand[7], rand[6], rand[10]]))
 
  #yWithMorphologySequentialStreamDropoutDev_BaselineLanguage_Fast.py	Basque	eu	0.1	100	512	1	0.002	RANDOM_MODEL	0.23	16	20	43.3432767303	1.55933869897	4.17839380314
print "/afs/cs.stanford.edu/u/mhahn/scr/"+language+"_ngrams_after_tuning.tsv" 
print "/afs/cs.stanford.edu/u/mhahn/scr/"+language+"_ngrams_decay_after_tuning.tsv"
