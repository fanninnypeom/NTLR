import pickle
data_path = "/data/wuning/NTLR/"
def get_train_data(dataset, model_name):
  if dataset == "beijing" and model_name == "topic":
    train_sets = pickle.load(open(data_path + "beijing/OSMBeijingCompleteTrainData_50", "rb"))    
    loc_list = pickle.load(open(data_path + "beijing/CompletelocList", "rb"))
    topic_sets = pickle.load(open(data_path + "beijing/beijing_topics", "rb"))
    train_sets_new = []
    for batch in train_sets:
      if len(batch) == 100:
        train_sets_new.append(batch)    
    topic_sets_new = []      
    for batch in topic_sets:
      if len(batch) == 100:
        topic_sets_new.append(batch)
    return train_sets_new, topic_sets_new, 500, len(loc_list)
  if dataset == "foursquare" and (model_name == "region" or model_name == "multi_mode"):
      
    train_geo_set = pickle.load(open(data_path + "LSBN/train_geo_set", "rb"))    
    train_time_set = pickle.load(open(data_path + "LSBN/train_time_set", "rb"))
    train_user_set = pickle.load(open(data_path + "LSBN/train_user_set", "rb"))
    train_loc_set = pickle.load(open(data_path + "LSBN/train_loc_set", "rb"))
    train_topic_set = pickle.load(open(data_path + "LSBN/foursquare_topics", "rb"))
    with open("/data/wuning/NTLR/LSBN/foursquare.pk", "rb") as f:
      u = pickle._Unpickler(f)
      u.encoding = 'latin1'
      p = u.load()
    loc_cor = p["vid_lookup"]
    loc_num = len(loc_cor.keys())
    return train_topic_set, train_loc_set, train_user_set, train_time_set, train_geo_set, loc_num

def get_test_data(dataset, model_name):
  if dataset == "foursquare" and (model_name == "region" or model_name == "multi_mode"):
    test_geo_set = pickle.load(open(data_path + "LSBN/test_geo_set", "rb"))                    
    test_time_set = pickle.load(open(data_path + "LSBN/test_time_set", "rb"))
    test_user_set = pickle.load(open(data_path + "LSBN/test_user_set", "rb"))
    test_loc_set = pickle.load(open(data_path + "LSBN/test_loc_set", "rb"))
    test_topic_set = pickle.load(open(data_path + "LSBN/test_topic_set", "rb"))
    return test_topic_set, test_loc_set, test_user_set, test_time_set, test_geo_set
                          

