import sys
import os

PATH = "/u/scr/mhahn/deps/memory-need-ngrams-morphology"

files = [x for x in os.listdir(PATH) if "forWords_Sesotho" in x]

# MAK at https://stackoverflow.com/questions/14063195/python-3-get-2nd-to-last-index-of-occurrence-in-string
def find_second_last(text, pattern):
   return text.rfind(pattern, 0, text.rfind(pattern))


with open("results.tsv", "w") as outFile:
 print("\t".join([str(x) for x in ["Script", "Run", "Model", "Distance", "Surprisal", "MI", "Memory"]]), file=outFile)
 for f in files:
  with open(PATH+"/"+f, "r") as inFile:
     args, surps = inFile 
     args = args.strip()
     print(args)
     print(f)
     surps = [float(x) for x in surps.strip().split(" ")]
     script = f[f.index("forWords"):f.index("_model")]
     model = f[f.rfind("_")+1:-4]
     run = f[find_second_last(f, "_")+1:f.rfind("_")]
     print(script, model, surps)
     mis = [surps[i] - surps[i+1] for i in range(len(surps)-1)]
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
       print("\t".join([str(x) for x in [script, run, model, i, surps[i], mis[i], memories[i]]]), file=outFile)
