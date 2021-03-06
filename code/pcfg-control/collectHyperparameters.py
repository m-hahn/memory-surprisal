from ud_languages import languages

import random

#random.shuffle(languages)

import glob

bounds = []
bounds.append(['VOCAB_FOR_RELATION_THRESHOLD', int, 10, 30, 50, 100, 300, 500, 800, 1000, 2000]) # variable
bounds.append(['OTHER_WORDS_SMOOTHING', float, 0.0001])
bounds.append(['MERGE_ACROSS_RELATIONS_THRESHOLD', int, 10, 20, 50, 100, 400, 800, 1000]) 
bounds.append(['REPLACE_WORD_WITH_PLACEHOLDER', float, 0.0, 0.2, 0.4]) 

names = [x[0] for x in bounds]
types = [x[1] for x in bounds]

with open("hyperparameters.tsv", "w") as outFile:
  print("\t".join(["Language", "EstimatedSurprisal"] + names), file=outFile)
  for language in sorted(languages):
     searches = glob.glob("/u/scr/mhahn/deps/memory-need-pcfg/search-"+language+"_2.4_cky_gpu_Stat10_FewNTs_Debug_UD3_GPU_Lexical_Rel_NoSmooth9.py_model_*_RANDOM_BY_TYPE.txt")
     if len(searches) == 0:
       print("Missing", language)
       continue
     searchResults = []
     for resu in searches:
         with open(resu, "r") as inFile:
            results = inFile.read().strip().split("\n")
         if len(results) < 15:
           continue
         for i in range(len(results)):
            results[i] = results[i].split("\t")
            vals = [float(x) for x in results[i][1][1:-1].split(", ")]
            vals = sum(vals)/len(vals)
         #   print(vals)
            results[i][1] = vals
         results = sorted(results, key=lambda x:x[1])
         result = results[0]
         searchResults.append(result)
         
     if len(searchResults) == 0:
       print("Incomplete", language)
       print(searches)
       continue
     searchResults = sorted(searchResults, key=lambda x:float(x[0]))
     print("\t".join([str(x) for x in ([language , searchResults[0][1]] + [t(x) for t, x in zip(types, searchResults[0][2:])])]), file=outFile)


