import sys
import os

PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology"

files = sorted([x for x in os.listdir(PATH) if "forWords_Sesotho" in x])

# MAK at https://stackoverflow.com/questions/14063195/python-3-get-2nd-to-last-index-of-occurrence-in-string
def find_second_last(text, pattern):
   return text.rfind(pattern, 0, text.rfind(pattern))
def find_third_last(text, pattern):
   return text.rfind(pattern, 0, text.rfind(pattern, 0, text.rfind(pattern)))


with open("results.tsv", "w") as outFile:
 print("\t".join([str(x) for x in ["Script", "Run", "Model", "Distance", "Surprisal", "MI", "Memory", "UnigramCE", "Type"]]), file=outFile)
 for f in files:
  with open(PATH+"/"+f, "r") as inFile:
     args, surps = inFile 
     args = args.strip()
     print(args)
     print(f)
     surps = [float(x) for x in surps.strip().split(" ")]
     script = f[f.index("forWords"):f.index("_model")]
     model2 = f[f.rfind("_")+1:-4]
     model1 = f[find_second_last(f, "_")+1:f.rfind("_")]
     model = model1+"_"+model2
     if "RANDOM" in model:
         model = "RANDOM"
     elif "REAL" in model:
         model = "REAL"
     elif "REVERSE" in model:
         model = "REVERSE"
     run = f[find_third_last(f, "_")+1:find_second_last(f, "_")]
     print(script, model, surps)
     mis = [surps[i] - surps[i+1] for i in range(len(surps)-1)]
     for i in range(len(mis), 12):
        mis.append(0)
     print(mis)
     tmis = [mis[i] * (i+1) for i in range(len(mis))]
     print(tmis)
     surprisals = [surps[0]]
     memories = [0]
     for i in range(len(mis)):
        surprisals.append(surprisals[-1]-mis[i])
        memories.append(memories[-1] + tmis[i])
     print(surprisals)
     print(memories)
     for i in range(len(mis)):
       print("\t".join([str(x) for x in [script, run, model, i, surprisals[i], mis[i], memories[i], surprisals[0], model if model in ["REAL", "RANDOM", "REVERSE"] else "OPTIM"]]), file=outFile)
