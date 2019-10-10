import pickle
with open("/data/wuning/NTLR/LSBN/foursquare.pk", "rb") as f:
  u = pickle._Unpickler(f)
  u.encoding = 'latin1'
  p = u.load()
trainid_rawid = {}
for key in p["vid_list"].keys():
  trainid_rawid[str(p["vid_list"][key][0])] = key   
pickle.dump(trainid_rawid, open("/data/wuning/NTLR/LSBN/trainid_rawid", "wb"))


