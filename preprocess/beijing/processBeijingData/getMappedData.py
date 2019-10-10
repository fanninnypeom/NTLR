import pickle
import numpy as np
File = open("/data/wuning/NTLR/beijing/beijing_map_matching/beijing_tras.txt", "r")
timeData = pickle.load(open('/data/wuning/NTLR/beijing/beijingTaxiTime', 'rb'))
userData = pickle.load(open('/data/wuning/NTLR/beijing/beijingTaxiUser', 'rb'))
timeData = timeData#[0:1000000]
lines = File.readlines()
lines = lines#[0:1000001]
tras = []
allTime = []
allUser = []
tras_batch = []
mask_batch = []
masks = []
count = 0
for line, times, user in zip(lines[1:], timeData, userData):
  if count % 1000 == 0:
    print(count)
  count += 1
  tra = []
  timeArr = []
  mask = []
  try:
    tem_1 = line.split(";")[2]
    tem_2 = tem_1.split(",")
    if len(tem_2) < 5:
      continue    
    last = -1
    for item, time in zip(tem_2, times):
      if not item == last:
        tra.append(item)
        timeArr.append(time)
        mask.append(1)
      last = item
#    timeArr = timeArr[:len(mask)]
    while len(mask) < len(tem_2):      
      mask.append(0)
    while len(timeArr) < len(tem_2):  # padding
      timeArr.append(0)
    while len(tra) < len(tem_2):      #padding
      tra.append(0)
    tras.append(tra)
    allTime.append(timeArr)
    masks.append(mask)
    allUser.append(user)
  except:
    print(line)
    pass
MaskFile = open("/data/wuning/NTLR/beijing/maskData", "wb")
TraFile = open("/data/wuning/NTLR/beijing/blockData50000", "wb")
TimeFile_ = open("/data/wuning/NTLR/beijing/blockDataTime50000", "wb")
UserFile = open("/data/wuning/NTLR/beijing/blockDataUser50000", "wb")
pickle.dump(tras, TraFile, -1)
pickle.dump(masks, MaskFile, -1)
pickle.dump(allTime, TimeFile_, -1)
pickle.dump(allUser, UserFile)


