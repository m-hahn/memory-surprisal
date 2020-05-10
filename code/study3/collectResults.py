PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology/"
import os
MAX_MEMORY = 8
with open("output/curves.tsv", "w") as outFileCurves:
 print("Language", "Script", "ModelID", "Model", "Distance", "It", "Surprisal", "Memory", file=outFileCurves)
 with open("output/auc.tsv", "w") as outFileAUC:
  print("Language", "Script", "ModelID", "Model", "AUC", file=outFileAUC)

  for filename in sorted(os.listdir(PATH)):
     name = filename[10:-4].split("_")
     print(name)
     language = "_".join(name[:2])
     script = "_".join(name[2:-2])
     modelID = name[-2]
     model = name[-1]
     print(language,script,modelID, model)
     with open(PATH+filename, "r") as inFile:
        next(inFile)
        surprisals = [float(x) for x in next(inFile).strip().split(" ")]
        print(surprisals)
        mis = [surprisals[i] - surprisals[i+1] for i in range(len(surprisals)-1)]
        totalMemory = 0
        print(language, script, modelID, model, 0, 0.0, surprisals[0], totalMemory, file=outFileCurves)
        auc = 0
        for i in range(len(mis)):
           totalMemory += (i+1)*mis[i]
           print(language, script, modelID, model, i+1, mis[i], surprisals[i+1], totalMemory, file=outFileCurves)
           auc += (surprisals[0] - surprisals[i+1]) * (i+1)*mis[i]
        assert totalMemory <= MAX_MEMORY
        auc += (MAX_MEMORY-totalMemory) * (surprisals[0] - surprisals[len(mis)])
        print(language, script, modelID, model, MAX_MEMORY * (surprisals[0] - surprisals[len(mis)]) - auc + surprisals[len(mis)] * MAX_MEMORY, file=outFileAUC)
  
  
