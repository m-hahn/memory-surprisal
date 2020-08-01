import subprocess
for trainingSize in [500, 1000, 2000, 5000, 10000, 20000]:
   subprocess.call(["./python27", "yAnalyzeNeuralMorphFinalSaveLast_OnlyWordForms_BoundedVocab_Adjusted.py", "--trainingSize="+str(trainingSize)])
   subprocess.call(["./python27", "medianCI_All.py", str(trainingSize)])

