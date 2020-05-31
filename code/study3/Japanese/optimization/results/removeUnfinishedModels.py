import os
dirs = os.listdir(".")
for dir in dirs:
  try:
     files = os.listdir(dir+"/")
  except NotADirectoryError:
     continue
  foundModels = 0
  for filename in files:
    if filename == "ARCHIVE":
       continue
    with open(dir+"/"+filename, "r") as inFile:
      iterations = int(next(inFile).strip().split(" ")[0])
    if iterations == 999:
       foundModels += 1
    else:
       os.remove(dir+"/"+filename)
    #print(iterations)
  print(dir, foundModels)

