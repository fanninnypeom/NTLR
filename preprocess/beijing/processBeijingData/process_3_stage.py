import pickle
import time
history = {}
data_batches = []
time_batches = []
user_batches = []
mask_batches = []
tras = []
times = []
users = []
masks = []
locList = pickle.load(open("/data/wuning/map-matching/fmm-master/fmm-master/example/locList", "rb"))
rawData = pickle.load(open('/data/wuning/NTLR/beijing/blockData50000', 'rb'))
timeData = pickle.load(open('/data/wuning/NTLR/beijing/blockDataTime50000', 'rb'))
userData = pickle.load(open('/data/wuning/NTLR/beijing/blockDataUser50000', 'rb'))
maskData = pickle.load(open('/data/wuning/NTLR/beijing/maskData', 'rb'))
#userData = userData[0:1000000]

locList = []
for i in range(len(rawData)):
  for j in range(len(rawData[i])):
    if not rawData[i][j] in locList:
      locList.append(rawData[i][j])               

############  重新映射token  + 将时间戳转为token
for i in range(len(rawData)):
  for j in range(len(rawData[i])):
    if rawData[i][j] == 0:
      continue
    rawData[i][j] = locList.index(rawData[i][j])      
for i in range(len(timeData)):
  for j in range(len(timeData[i])):
    try:
      time_struct = list(time.localtime(timeData[i][j]))
      temp = [0, 0]
      temp[0] = time_struct[3]*4 + int(time_struct[4]/15)
      temp[1] = time_struct[6]
    except:
      new_time = int(str(timeData[i][j])[0:10])
      time_struct = list(time.localtime(new_time))
      temp = [0, 0]
      temp[0] = time_struct[3]*4 + int(time_struct[4]/15)
      temp[1] = time_struct[6]
    timeData[i][j] = temp   
#############
print("map over")

userIndexedData = {}
for tra, time, user, mask in zip(rawData, timeData, userData, maskData):
  if user in userIndexedData:
    userIndexedData[user].append([tra, time, mask])
  else:
    userIndexedData[user] = []
#for key in userIndexedData:
#  if len(userIndexedData[key]) > 10:
#    history[key] = userIndexedData[key][:10]  

#fw = open("/data/wuning/map-matching/userIndexedHistoryAttention", "wb")
#pickle.dump(history, fw, -1)

for key in userIndexedData:
  if len(userIndexedData[key]) > 10:
    for i in range(10, len(userIndexedData[key])):
      tras.append(userIndexedData[key][i][0]) 
      times.append(userIndexedData[key][i][1]) 
      masks.append(userIndexedData[key][i][2])
      users.append(key)    
  if(len(tras) >= 100):
    data_batches.append(tras[0 : 100])
    time_batches.append(times[0 : 100])
    user_batches.append(users[0 : 100])
    mask_batches.append(masks[0 : 100])
    tras = []
    times = []
    users = []
    masks = []


fd = open("/data/wuning/NTLR/beijing/taxiTrainData", "wb")
ft = open("/data/wuning/NTLR/beijing/taxiTrainDataTime", "wb")
fu = open("/data/wuning/NTLR/beijing/taxiTrainDataUser", "wb")
fm = open("/data/wuning/NTLR/beijing/taxiTrainDataMask", "wb")

pickle.dump(time_batches, ft, -1)
pickle.dump(user_batches, fu, -1)
pickle.dump(data_batches, fd, -1)
pickle.dump(mask_batches, fm, -1)

