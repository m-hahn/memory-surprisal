import os
import random
dirs = os.listdir(".")
for dir in dirs:
  try:
     files = [x for x in os.listdir(dir+"/") if "ARCHIVE" not in x]
  except NotADirectoryError:
     continue
  print(files)  
  try:
    os.mkdir(f"{dir}/ARCHIVE")
  except FileExistsError:
    pass
  random.shuffle(files)
  if "HeldoutClip" not in dir or "Graphemes" in dir:
     extra = files[5:]
  else:
     extra = files[10:]
  for f in extra:
    os.rename(f"{dir}/{f}", f"{dir}/ARCHIVE/{f}")


