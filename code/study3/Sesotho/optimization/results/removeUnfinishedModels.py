import os
dirs = os.listdir(".")
for dir in dirs:
  try:
     files = os.listdir(dir+"/")
  except NotADirectoryError:
     continue
  foundModels = 0
  for filename in files:
    with open(dir+"/"+filename, "r") as inFile:
      iterations = int(next(inFile).strip().split(" ")[0])
    if iterations == 999 or (("Graphemes" in filename or "Heldout" not in filename) and iterations > 400):
       foundModels += 1
    else:
       os.remove(dir+"/"+filename)
 #   print(iterations)
  print(dir, foundModels)

