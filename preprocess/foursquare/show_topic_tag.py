import pickle
import numpy as np
trainid_rawid = pickle.load(open("/data/wuning/NTLR/LSBN/trainid_rawid", "rb"))
rawid_func = pickle.load(open("/data/wuning/NTLR/LSBN/rawid_func", "rb"))
lda = pickle.load(open("/data/wuning/NTLR/LSBN/topics", "rb"))
top_loc = np.array(lda.expElogbeta).argsort()[:,-20:]

for top in top_loc:
  for item in top:
    try:
      print(rawid_func[trainid_rawid[lda.id2word[item]]])
    except Exception:
      pass
  print("------------")
