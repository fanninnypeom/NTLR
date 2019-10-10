import pickle
import numpy as np

mu = pickle.load(open("/data/wuning/NTLR/LSBN/mu", "rb"))

cov = pickle.load(open("/data/wuning/NTLR/LSBN/cov", "rb"))

with open("/data/wuning/NTLR/LSBN/foursquare.pk", "rb") as f:
  u = pickle._Unpickler(f)
  u.encoding = 'latin1'
  p = u.load()
loc_cor = p["vid_lookup"]
geo_lists = []
for key in loc_cor.keys():
  geo_lists.append(loc_cor[key])
geo_mean = np.mean(np.array(geo_lists), 0)

mean = mu / 100 + geo_mean
cov = cov 

print("mean:", str(mean.tolist()), "cov:", cov.tolist())
#for key in loc_cor.keys():
#  loc_cor[key] = (loc_cor[key] - geo_mean) * 100

