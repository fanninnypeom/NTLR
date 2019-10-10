import pickle

result = []
time_batches = []
user_batches = []

rawData = pickle.load(open('/data/wuning/map-matching/taxiTrainData_', 'rb'))
timeData = pickle.load(open('/data/wuning/map-matching/taxiTrainDataTime_', 'rb'))
userData = pickle.load(open('/data/wuning/map-matching/taxiTrainDataUser_', 'rb'))

print(len(rawData), len(timeData), len(userData))

tras = []
times = []
users = []
for tra, time, user in zip(rawData, timeData, userData):
  if(len(tra) > 40):
    for i in range(0, int(len(tra)/41), 1):
      users.append(user)
      tras.append(tra[i*41 : (i+1)*41])
      times.append(time[i*41 : (i+1)*41])
#  if(len(tras) >= 100):
#    result.append(tras[0 : 100])
#    time_batches.append(times[0 : 100])
#    user_batches.append(users[0 : 100])
#    tras = []


fw = open("/data/wuning/map-matching/beijingTaxi_40", "wb")
ft = open("/data/wuning/map-matching/beijingTaxiTime_40", "wb")
fu = open("/data/wuning/map-matching/beijingTaxiUser_40", "wb")

pickle.dump(tras, fw, -1)
pickle.dump(times, ft, -1)
pickle.dump(users, fu, -1)

fw.close()

