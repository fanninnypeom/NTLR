import torch
def evaluate(model, test_topic_set, test_loc_set,test_geo_set, test_user_set, test_time_set, device):
  hit_num = 0
  all_num = 0
  for topics, locs, geos, users, times in zip(test_topic_set, test_loc_set, test_geo_set, test_user_set, test_time_set):
    for i in range(len(locs) - 1):
      loc_hidden, geo_hidden = model.initHidden(1) 
      topic_dist, topic_action_log_prob, loc_dist_dropout, loc_dist, loc_prob, action_log_prob, geo_hidden, loc_hidden = model(0,
                                                                                            torch.tensor([topics[i + 1]], dtype=torch.long, device=device),
                                                                                            torch.tensor([(int(geos[i][0]) + 50) * (int(geos[i][1]) + 50) % 10000], dtype=torch.long, device=device),
                                                                                            torch.tensor([geos[i]], dtype=torch.float, device=device), 
                                                                                            torch.tensor([locs[i]], dtype=torch.long, device=device), 
                                                                                            torch.tensor([users], dtype=torch.long, device=device),
                                                                                            torch.tensor([times[i]], dtype=torch.long, device=device),
                                                                                            geo_hidden, 
                                                                                            loc_hidden) 
    pred = torch.argmax(loc_dist, 1)
    if pred[0] == locs[-1]:
      hit_num += 1
    all_num += 1       
  return hit_num / all_num   

def multi_mode_evaluate(model, test_topic_set, test_loc_set,test_geo_set, test_user_set, test_time_set, device):
  hit_num = 0
  all_num = 0
  for topics, locs, geos, users, times in zip(test_topic_set, test_loc_set, test_geo_set, test_user_set, test_time_set):
    for i in range(len(locs) - 1):
      loc_hidden, topic_hidden = model.initHidden(1) 
      region_action, topic_action, region_dist, topic_dist, loc_dist, loc_hidden, topic_hidden = model(0,
                                                            torch.tensor([topics[i + 1]], dtype=torch.long, device=device),
                                                            torch.tensor([(int(geos[i][0]) + 50) * (int(geos[i][1]) + 50) % 10000], dtype=torch.long, device=device),
                                                            torch.tensor([geos[i]], dtype=torch.float, device=device), 
                                                            torch.tensor([locs[i]], dtype=torch.long, device=device), 
                                                            torch.tensor([users], dtype=torch.long, device=device),
                                                            torch.tensor([times[i]], dtype=torch.long, device=device),
                                                            loc_hidden,
                                                            topic_hidden) 
    pred = torch.argmax(loc_dist, 1)
    if pred[0] == locs[-1]:
      hit_num += 1
    all_num += 1       
  return hit_num / all_num   

