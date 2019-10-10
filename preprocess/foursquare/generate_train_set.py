import pickle
import numpy as np

with open("/data/wuning/NTLR/LSBN/foursquare.pk", "rb") as f:
  u = pickle._Unpickler(f)
  u.encoding = 'latin1'
  p = u.load()
train_sets = p["data_neural"]
loc_cor = p["vid_lookup"]
geo_lists = []
for key in loc_cor.keys():
  geo_lists.append(loc_cor[key])
geo_mean = np.mean(np.array(geo_lists), 0)
for key in loc_cor.keys():
  loc_cor[key] = (loc_cor[key] - geo_mean) * 100


train_loc_set = []
train_time_set = []
train_user_set = []
train_geo_set = []
train_tra = []
train_time = []
train_user = []

test_loc_set = []
test_time_set = []
test_user_set = []
test_geo_set = []
for key_1 in train_sets.keys():
  for key_2 in train_sets[key_1]["sessions"].keys():    
    locs = []
    times = []
    for item in train_sets[key_1]["sessions"][key_2]:
      locs.append(item[0])
      times.append(item[1]) 
    if not key_2 in train_sets[key_1]["test"]:  
      train_tra.append(locs)
      train_time.append(times)
      train_user.append(key_1)    
    else:
      test_loc_set.append(locs)
      test_time_set.append(times)  
      test_user_set.append(key_1)           

lenIndexedRaw = {}
lenIndexedTime = {}
lenIndexedUser = {}

for raw, time, user in zip(train_tra, train_time, train_user):
  length = len(raw)
  if length in lenIndexedRaw:
    lenIndexedRaw[length].append(raw[:length])
    lenIndexedTime[length].append(time[:length])
    lenIndexedUser[length].append(user)
  else:
    lenIndexedRaw[length] = [raw[:length]]
    lenIndexedTime[length] = [time[:length]]
    lenIndexedUser[length] = [user]


for key in lenIndexedRaw:
  for i in range(0, len(lenIndexedRaw[key]), 50):
    train_time_set.append(lenIndexedTime[key][i : i + 50])
    train_user_set.append(lenIndexedUser[key][i : i + 50])
    train_loc_set.append(lenIndexedRaw[key][i : i + 50])

for batch in train_loc_set:
  geo_batch = []
  for tra in batch:
    geo_tra = []
    for loc in tra:
      geo_tra.append(loc_cor[loc])            
    geo_batch.append(geo_tra) 
  train_geo_set.append(geo_batch)     

for tra in test_loc_set:
  geo_tra = []
  for loc in tra:
    geo_tra.append(loc_cor[loc]) 
  test_geo_set.append(geo_tra)  
  
pickle.dump(train_loc_set, open("/data/wuning/NTLR/LSBN/train_loc_set", "wb")) 
pickle.dump(train_time_set, open("/data/wuning/NTLR/LSBN/train_time_set", "wb")) 
pickle.dump(train_user_set, open("/data/wuning/NTLR/LSBN/train_user_set", "wb")) 
pickle.dump(train_geo_set, open("/data/wuning/NTLR/LSBN/train_geo_set", "wb")) 
pickle.dump(test_loc_set, open("/data/wuning/NTLR/LSBN/test_loc_set", "wb")) 
pickle.dump(test_time_set, open("/data/wuning/NTLR/LSBN/test_time_set", "wb")) 
pickle.dump(test_user_set, open("/data/wuning/NTLR/LSBN/test_user_set", "wb")) 
pickle.dump(test_geo_set, open("/data/wuning/NTLR/LSBN/test_geo_set", "wb")) 
