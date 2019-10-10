import pickle
f = open("/data/wuning/LSBN Data/dataset_tsmc2014/dataset_TSMC2014_NYC.txt", "r", encoding = 'latin-1')
lines = f.readlines()
rawid_func = {}
for line in lines:
    
#  print(line.split("\t")[0], line.split("\t")[1], line.split("\t")[2], line.split("\t")[3])  
  raw_id = line.split("\t")[1]
  func = line.split("\t")[3]
  rawid_func[raw_id] = func

pickle.dump(rawid_func, open("/data/wuning/NTLR/LSBN/rawid_func", "wb"))
