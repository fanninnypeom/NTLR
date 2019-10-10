import pickle
import time
import numpy as np
history = {}
data_batches = []
time_batches = []
user_batches = []

tras = []
times = []
users = []
masks = []

fd = open("/data/wuning/NTLR/beijing/taxiTrainData", "rb")
ft = open("/data/wuning/NTLR/beijing/taxiTrainDataTime", "rb")
fu = open("/data/wuning/NTLR/beijing/taxiTrainDataUser", "rb")
fm = open('/data/wuning/NTLR/beijing/taxiTrainDataMask', 'rb')


maskData = pickle.load(fm)
rawData = pickle.load(fd)
timeData = pickle.load(ft)
userData = pickle.load(fu)
masks = []
raws = []
times = []
users = []

for bat_1, bat_2, bat_3, bat_4 in zip(maskData, rawData, timeData, userData):
  masks.extend(bat_1)
  raws.extend(bat_2)
  times.extend(bat_3)
  users.extend(bat_4)

lenIndexedRaw = {}
lenIndexedTime = {}
lenIndexedUser = {}

for mask, raw, time, user in zip(masks, raws, times, users):
  length = np.sum(np.array(mask))
  print(mask)
  if length in lenIndexedRaw:
    lenIndexedRaw[length].append(raw[:length])   
    lenIndexedTime[length].append(time[:length])
    lenIndexedUser[length].append(user)
  else:
    lenIndexedRaw[length] = [raw[:length]]
    lenIndexedTime[length] = [time[:length]]
    lenIndexedUser[length] = [user]
                  

time_batches = []
user_batches = []
data_batches = []

for key in lenIndexedRaw:
  print(key, len(lenIndexedRaw[key]))  
  for i in range(0, len(lenIndexedRaw[key]), 100):    
    print("i:", i)  
    time_batches.append(lenIndexedTime[key][i : i + 100])          
    user_batches.append(lenIndexedUser[key][i : i + 100])
    data_batches.append(lenIndexedRaw[key][i : i + 100])


fd = open("/data/wuning/NTLR/beijing/train_loc_set", "wb")
ft = open("/data/wuning/NTLR/beijing/train_time_set", "wb")
fu = open("/data/wuning/NTLR/beijing/train_user_set", "wb")

pickle.dump(time_batches, ft, -1)
pickle.dump(user_batches, fu, -1)
pickle.dump(data_batches, fd, -1)

