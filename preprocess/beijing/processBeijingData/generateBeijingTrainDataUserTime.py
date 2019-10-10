import os
import pickle

dir = "/data/wuning/taxiTrajectory/201504/"

fw = open("/data/wuning/NTLR/beijing/beijingTaxi", "wb")
ft = open("/data/wuning/NTLR/beijing/beijingTaxiTime", "wb")
fu = open("/data/wuning/NTLR/beijing/beijingTaxiUser", "wb")
count = 0

list = os.listdir(dir) #列出文件夹下所有的目录与文件
tras = []
times = []
users = []
for i in range(0,len(list)): 
#  if(count == 40000):
#    break;
  count += 1
  if count % 1000 == 0:
    print(count)
    print(len(tras))
  path = os.path.join(dir,list[i])
  if os.path.isfile(path):
    tra = []
    time = []
    f = open(path , "r") 
    try:
      lines = f.readlines()
    except:
      continue
    for line in lines:
      arr = line.split(",")
      if(len(arr)<8):
        break
      lat = arr[6]
      lon = arr[7] 
      try:
        lat_ = float(lat)/100000.0
        lon_ = float(lon)/100000.0
        ti = int(arr[3])     
        if(len(tra) >= 1 and abs(lat_ - tra[-1][0]) < 0.0001 and abs(lon_ - tra[-1][1]) < 0.0001):
          continue
#        if(lat_>39.84597 and lat_<39.96949 and lon_>116.310882 and lon_<116.465378):
#          break;
        if(lat_<39.84597 or lat_>39.96949 or lon_<116.310882 or lon_>116.465378 ):
          if(not len(tra) == 0):
            users.append(count)
            tras.append(tra)
            times.append(time)
            tra = []
            time = []
          continue;
        time.append(ti)
        tra.append([lat_, lon_])
      except:
        break
    if(not len(tra) == 0):
      users.append(count)
      times.append(time)  
      tras.append(tra)
pickle.dump(tras, fw, -1)
pickle.dump(times, ft, -1)
pickle.dump(users, fu, -1)
fw.close()
