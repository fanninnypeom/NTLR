import torch
from model import *
from utils import *
from torch import optim
from infer import *
import numpy as np
import time
import random
import sys
import pickle
import math
import os
import copy
import sys

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(42)

print(str(int(sys.argv[1]) // 2))
os.environ["CUDA_VISIBLE_DEVICES"] = str(int(sys.argv[1]) // 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dataset = "foursquare"        # beijing foursquare wuxi

model_name = "multi_mode"         # topic region topic_region multi-mode

def pretrain_sem(topic_gru, 
                 loc_gru, 
                 epoches, 
                 train_sets, 
                 topic_train_sets, 
                 learning_rate):
  topic_gru_optimizer = optim.Adam(topic_gru.parameters(), lr=learning_rate)
  loc_gru_optimizer = optim.Adam(loc_gru.parameters(), lr=learning_rate)
  criterion = torch.nn.CrossEntropyLoss()
  softmax = torch.softmax
  act = torch.tanh
  loc2topic = nn.Linear(topic_gru.hidden_size + loc_gru.hidden_size, topic_gru.vocab_size).to(device)
  topic2loc = nn.Linear(topic_gru.embedding_size + loc_gru.hidden_size, loc_gru.vocab_size).to(device)
  loc2loc = nn.Linear(loc_gru.hidden_size, loc_gru.vocab_size).to(device)
  topic2topic = nn.Linear(topic_gru.hidden_size, topic_gru.vocab_size).to(device)

  loc2topic_optimizer = optim.Adam(loc2topic.parameters(), lr=learning_rate)
  topic2loc_optimizer = optim.Adam(topic2loc.parameters(), lr=learning_rate)
  loc2loc_optimizer = optim.Adam(loc2loc.parameters(), lr=learning_rate)
  topic2topic_optimizer = optim.Adam(topic2topic.parameters(), lr=learning_rate)
#  final = nn.Linear(hidden_size, vocab_size)
  for ite in range(0, epoches):
    topic_losses =[]
    loc_losses = []  
    counter = 0
    for topic_batch, loc_batch in zip(topic_train_sets, train_sets):
      topic_gru_hidden = topic_gru.initHidden()
      topic_gru_optimizer.zero_grad()
      loc_gru_hidden = loc_gru.initHidden()
      loc_gru_optimizer.zero_grad()
      input_topic = torch.tensor(topic_batch, dtype=torch.long, device=device) 
      input_loc = torch.tensor(loc_batch, dtype=torch.long, device=device)   
      topic_gru_loss = 0
      loc_gru_loss = 0  
#      topic_gru_output, topic_gru_hidden = topic_gru(input_topic[:, 0], topic_gru_hidden)
      for time_step in range(0, topic_batch.shape[1] - 1):
        loc_gru_output, loc_gru_hidden = loc_gru(input_loc[:, time_step], loc_gru_hidden)     
        topic_gru_output, topic_gru_hidden = topic_gru(input_topic[:, time_step],  topic_gru_hidden)
        loc_emb = loc_gru.embedding(input_loc[:, time_step]).view(-1, loc_gru.embedding_size)
#        topic_dist = loc2topic(torch.cat((loc_emb, topic_gru_output.squeeze()), 1))
        topic_dist = topic2topic(topic_gru_output[0])
        topic_emb = topic_gru.embedding(input_topic[:, time_step + 1]).view(-1, topic_gru.embedding_size)
#        loc_dist = topic2loc(torch.cat((topic_emb, loc_gru_output[0]), 1))
        loc_dist = loc2loc(loc_gru_output[0])
        topic_gru_loss += criterion(topic_dist, input_topic[:, time_step + 1])
        loc_gru_loss += criterion(loc_dist, input_loc[:, time_step + 1])

      loc_gru_loss.backward(retain_graph=True)

      loc_gru_optimizer.step()
#      topic_gru_loss.backward()
#      topic_gru_optimizer.step()
        

      topic2loc_optimizer.step()
      loc2loc_optimizer.step()
      loc2topic_optimizer.step()
      topic2topic_optimizer.step()

      topic_losses.append(topic_gru_loss.item())
      loc_losses.append(loc_gru_loss.item())
      if counter % 100 == 0:
        print(counter, topic_gru_loss.item() / 50, loc_gru_loss.item() / 50)    
      counter += 1
    torch.save(loc2topic, "/data/wuning/NTLR/beijing/model/loc2topic.model") 
    torch.save(topic2loc, "/data/wuning/NTLR/beijing/model/topic2loc.model") 
    torch.save(topic_gru, "/data/wuning/NTLR/beijing/model/topic_gru.model")
    torch.save(loc_gru, "/data/wuning/NTLR/beijing/model/loc_gru.model")
    print("epoch:", ite, "topic loss", np.array(topic_losses).mean() / 50, "loc loss", np.array(loc_losses).mean() / 50)                
    
  return topic_gru, loc_gru

def rl_pretrain(model, 
                epoches, 
                pre_learning_rate, 
                loc_num, 
                train_topic_set, 
                train_loc_set, 
                train_user_set, 
                train_time_set, 
                train_geo_set,
                test_topic_set,
                test_loc_set,
                test_user_set,
                test_time_set,
                test_geo_set):
  criterion = torch.nn.CrossEntropyLoss()
  model_optimizer = optim.Adam(model.parameters(), lr=pre_learning_rate)               
  for ite in range(epoches):
    loc_losses = []
    topic_losses = []
    counter = 0
    for topic_batch, loc_batch, geo_batch, user_batch, time_batch in zip(train_topic_set, train_loc_set, train_geo_set, train_user_set, train_time_set):
      model_optimizer.zero_grad()
      input_topic = torch.tensor(topic_batch, dtype=torch.long, device=device)
      input_loc = torch.tensor(loc_batch, dtype=torch.long, device=device)   
      input_geo = torch.tensor(geo_batch, dtype=torch.float, device=device)
      input_geo_token = torch.tensor((np.array(geo_batch)[:, :, 0].astype(np.int32) + 50) * (np.array(geo_batch)[:, :, 1].astype(np.int32) + 50) % 10000, dtype=torch.long, device=device)
      input_user = torch.tensor(user_batch, dtype=torch.long, device=device)
      input_time = torch.tensor(time_batch, dtype=torch.long, device=device)
      topic_loss = 0
      loc_loss = 0  
      loc_hidden, geo_hidden = model.initHidden(input_geo.shape[0])
      for time_step in range(0, input_geo.shape[1] - 1):
        topic_dist, topic_action_log_prob, loc_dist_dropout, loc_dist, loc_prob, action_log_prob, geo_hidden, loc_hidden = model(1, input_topic[:, time_step + 1], input_geo_token[:, time_step], input_geo[:, time_step], input_loc[:, time_step], input_user, input_time[:, time_step], geo_hidden, loc_hidden)
        loc_loss += criterion(loc_dist, input_loc[:, time_step + 1])
        topic_loss += criterion(topic_dist, input_topic[:, time_step + 1])
      topic_loss.backward(retain_graph=True)
      loc_loss.backward()
      
      model_optimizer.step()

      topic_losses.append(topic_loss.item())
      loc_losses.append(loc_loss.item())
      if counter % 10 == 0:
        print("epoch: ", ite, "batch: ", counter, "loc_loss: ", loc_loss.item() / 50, "topic_loss: ", topic_loss.item() / 50)    
      counter += 1
    torch.save(model, "/data/wuning/NTLR/LSBN/model/region.model") 
    print("acc@1: ", evaluate(model, test_topic_set, test_loc_set, test_geo_set, test_user_set, test_time_set, device))      
    print("epoch:", ite, "topic loss", np.array(topic_losses).mean() / 50, "loc loss", np.array(loc_losses).mean() / 50)                
              
  pass
def rl_train_multi_mode( 
               model, 
               epoches, 
               pre_epoches,
               learning_rate,
               topic_num,
               region_num,
               loc_num,
               train_topic_set,
               train_loc_set,
               train_user_set,
               train_time_set,
               train_geo_set,
               test_topic_set,
               test_loc_set,
               test_user_set,
               test_time_set,
               test_geo_set):

  criterion = torch.nn.CrossEntropyLoss()
  model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  for ite in range(pre_epoches):
    topic_losses = []
    counter = 0
    for topic_batch, loc_batch, geo_batch, user_batch, time_batch in zip(train_topic_set, train_loc_set, train_geo_set, train_user_set, train_time_set):
      model_optimizer.zero_grad()
      input_topic = torch.tensor(topic_batch, dtype=torch.long, device=device)
      input_loc = torch.tensor(loc_batch, dtype=torch.long, device=device)   
      input_geo = torch.tensor(geo_batch, dtype=torch.float, device=device)
      input_geo_token = torch.tensor((np.array(geo_batch)[:, :, 0].astype(np.int32) + 50) * (np.array(geo_batch)[:, :, 1].astype(np.int32) + 50) % 10000, dtype=torch.long, device=device)
      input_user = torch.tensor(user_batch, dtype=torch.long, device=device)
      input_time = torch.tensor(time_batch, dtype=torch.long, device=device)
      region_loss = 0
      loc_loss = 0  
      reward_loss = 0
      topic_loss = 0
      loc_hidden, topic_hidden = model.initHidden(input_geo.shape[0])
      topic_print = None
      loc_print = None
      reward_print = None
      for time_step in range(0, input_geo.shape[1] - 1):
        region_action, topic_action, region_dist, topic_dist, loc_dist, loc_hidden, topic_hidden = model(0, input_topic[:, time_step], input_geo_token[:, time_step], input_geo[:, time_step], input_loc[:, time_step], input_user, input_time[:, time_step], loc_hidden, topic_hidden)
        topic_loss += criterion(topic_dist[: input_loc.shape[0], :], input_topic[:, time_step + 1])
      topic_loss.backward(retain_graph=True) 
#      loc_loss.backward()
      model_optimizer.step()

      topic_losses.append(topic_loss.item())
      if counter % 10 == 0:
        print("epoch: ", ite, "batch: ", counter, "topic_loss: ", topic_loss.item())    
      counter += 1
    torch.save(model, "/data/wuning/NTLR/LSBN/model/topic.model") 
    print("epoch:", ite, "topic loss", np.array(topic_losses).mean() / 50)                
  pass

#  if pre_epoches == 0:
#    model = torch.load("/data/wuning/NTLR/LSBN/model/topic.model")  

  for ite in range(epoches):
    loc_losses =[]
    topic_losses = []
    region_losses = []
    reward_losses = []
    counter = 0
    for topic_batch, loc_batch, geo_batch, user_batch, time_batch in zip(train_topic_set, train_loc_set, train_geo_set, train_user_set, train_time_set):
      model_optimizer.zero_grad()
      input_topic = torch.tensor(topic_batch, dtype=torch.long, device=device)
      input_loc = torch.tensor(loc_batch, dtype=torch.long, device=device)   
      input_geo = torch.tensor(geo_batch, dtype=torch.float, device=device)
      input_geo_token = torch.tensor((np.array(geo_batch)[:, :, 0].astype(np.int32) + 50) * (np.array(geo_batch)[:, :, 1].astype(np.int32) + 50) % 10000, dtype=torch.long, device=device)
      input_user = torch.tensor(user_batch, dtype=torch.long, device=device)
      input_time = torch.tensor(time_batch, dtype=torch.long, device=device)
      region_loss = 0
      loc_loss = 0  
      reward_loss = 0
      topic_loss = 0
      loc_hidden, topic_hidden = model.initHidden(input_geo.shape[0])
      topic_print = None
      loc_print = None
      reward_print = None
      for time_step in range(0, input_geo.shape[1] - 1):
        region_action, topic_action, region_dist, topic_dist, loc_dist, loc_hidden, topic_hidden = model(0, input_topic[:, time_step], input_geo_token[:, time_step], input_geo[:, time_step], input_loc[:, time_step], input_user, input_time[:, time_step], loc_hidden, topic_hidden)
        loc_loss += criterion(loc_dist[:, :], input_loc[:, time_step + 1])
        loc_prob = torch.distributions.Categorical(torch.softmax(loc_dist, dim = 1))
        rewards = loc_prob.log_prob(input_loc[:, time_step + 1])
#        loc_loss += - rewards[: input_loc.shape[0]].mean()
#        rewards = torch.softmax(loc_dist, dim = 1).gather(1, input_loc[:, time_step + 1].view(-1, 1))#[input_loc[:, time_step + 1]]
        reward = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
#        print("reward_shape:", reward.shape, input_loc[:, time_step + 1].shape)
#        reward_detach = reward.detach()

#        topic_action_prob = torch.distributions.Categorical(torch.softmax(topic_dist, dim = 1)).log_prob(topic_action)
#        region_action_prob = torch.distributions.Categorical(torch.softmax(region_dist, dim = 1)).log_prob(region_action)
        
        reward_loss += rewards.mean()
#        topic_loss += criterion(topic_dist[: input_loc.shape[0], :], input_topic[:, time_step + 1])
        topic_print = torch.argmax(topic_dist, 1)
        loc_print = torch.argmax(loc_dist, 1)
        reward_print = reward[:20]
#        topic_loss += torch.mul(reward.detach(), -topic_action_prob).mean()      
        topic_loss += torch.tensor(1.0)
        region_loss += torch.tensor(1.0)#torch.mul( reward.detach(), - region_action_prob).mean()
#      topic_loss.backward(retain_graph=True) 
#      region_loss.backward(retain_graph=True)
      loc_loss.backward()
#      loc_optimizer.step()
#      topic_gru_loss.backward()
      model_optimizer.step()
#      print("grad:", model.r2t_attn.weight.grad)

      region_losses.append(region_loss.item())
      topic_losses.append(topic_loss.item())
      loc_losses.append(loc_loss.item())
      reward_losses.append(reward_loss.item())
      if counter % 10 == 0:
#        print(reward_print, topic_print[:10], loc_print[:10])  
        print("epoch: ", ite, "batch: ", counter, "region_loss: ", region_loss.item() , topic_loss.item() , "loc_loss: ", loc_loss.item() / 50, "reward_loss: ", reward_loss.item() )    
      counter += 1
    torch.save(model, "/data/wuning/NTLR/LSBN/model/region.model") 
    print("acc@1: ", multi_mode_evaluate(model, test_topic_set, test_loc_set, test_geo_set, test_user_set, test_time_set, device))      
    print("epoch:", ite, "region loss", np.array(region_losses).mean() / 50, "loc loss", np.array(loc_losses).mean() / 50, "reward loss", np.array(reward_losses).mean())                
  pass


def rl_train_region( 
               model, 
               epoches, 
               learning_rate,
               region_num,
               loc_num,
               train_topic_set,
               train_loc_set,
               train_user_set,
               train_time_set,
               train_geo_set,
               test_topic_set,
               test_loc_set,
               test_user_set,
               test_time_set,
               test_geo_set):
  criterion = torch.nn.CrossEntropyLoss()
  model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  for ite in range(epoches):
    loc_losses =[]
    region_losses = []
    reward_losses = []
    counter = 0
    for topic_batch, loc_batch, geo_batch, user_batch, time_batch in zip(train_topic_set, train_loc_set, train_geo_set, train_user_set, train_time_set):
      model_optimizer.zero_grad()
      input_topic = torch.tensor(topic_batch, dtype=torch.long, device=device)
      input_loc = torch.tensor(loc_batch, dtype=torch.long, device=device)   
      input_geo = torch.tensor(geo_batch, dtype=torch.float, device=device)
      input_geo_token = torch.tensor((np.array(geo_batch)[:, :, 0].astype(np.int32) + 50) * (np.array(geo_batch)[:, :, 1].astype(np.int32) + 50) % 10000, dtype=torch.long, device=device)
      input_user = torch.tensor(user_batch, dtype=torch.long, device=device)
      input_time = torch.tensor(time_batch, dtype=torch.long, device=device)
      region_loss = 0
      loc_loss = 0  
      reward_loss = 0
      loc_hidden, geo_hidden = model.initHidden(input_geo.shape[0])
      for time_step in range(0, input_geo.shape[1] - 1):
        loc_dist_dropout, loc_dist, loc_prob, action_log_prob, geo_hidden, loc_hidden = model(0, input_topic[:, time_step + 1], input_geo_token[:, time_step], input_geo[:, time_step], input_loc[:, time_step], input_user, input_time[:, time_step], geo_hidden, loc_hidden)
        loc_loss += criterion(loc_dist, input_loc[:, time_step + 1])
        reward = loc_prob.log_prob(input_loc[:, time_step + 1])
#        reward = torch.softmax(loc_dist, dim = 1).gather(1, input_loc[:, time_step + 1].view(-1, 1))#[input_loc[:, time_step + 1]]
#        print("reward_shape:", reward.shape, input_loc[:, time_step + 1].shape)
#        reward_detach = reward.detach()
        reward = torch.tensor(0.0)
        action_log_prob = torch.tensor(0.0)
        reward_loss += reward.mean()
        region_loss += torch.mul(reward.detach(), - action_log_prob).mean()      
#      region_loss.backward(retain_graph=True)
      loc_loss.backward()
      
#      print("grad:", model.loc_topic_r2t_layer.weight.grad)
#      loc_optimizer.step()
#      topic_gru_loss.backward()
      model_optimizer.step()

      region_losses.append(region_loss.item())
      loc_losses.append(loc_loss.item())
      reward_losses.append(reward_loss.item())
      if counter % 10 == 0:
        print("epoch: ", ite, "batch: ", counter, "region_loss: ", region_loss.item() / 50, "loc_loss: ", loc_loss.item() / 50, "reward_loss: ", reward_loss.item() / 50)    
      counter += 1
    torch.save(model, "/data/wuning/NTLR/LSBN/model/region.model") 
    print("acc@1: ", evaluate(model, test_topic_set, test_loc_set, test_geo_set, test_user_set, test_time_set, device))      
    print("epoch:", ite, "region loss", np.array(region_losses).mean() / 50, "loc loss", np.array(loc_losses).mean() / 50, "reward loss", np.array(reward_losses).mean())                
  pass

def SEM(topic_gru, phi, epoches, train_sets, loc_size, topic_num, learning_rate):
  topic_gru_optimizer = optim.Adam(topic_gru.parameters(), lr=learning_rate)
  criterion = torch.nn.CrossEntropyLoss()
  softmax = torch.softmax
  topic_sets = torch.randint(0, topic_num, (train_sets.shape[0], train_sets.shape[1], train_sets.shape[2]))
  for ite in range(0, epoches):
    topic_sets_new = copy.deepcopy(topic_sets)  
    batch_count = 0 
## E step 
    gru_losses = []   
    for batch, loc_batch in zip(topic_sets_new, train_sets):
      topic_gru_hidden = topic_gru.initHidden()
      topic_gru_optimizer.zero_grad() 
      input_topic = torch.tensor(batch, dtype=torch.long, device=device) 
      input_loc = torch.tensor(loc_batch, dtype=torch.long, device=device)
      gru_hidden = topic_gru_hidden
      for time_step in range(batch.shape[1] - 1):  #E step
        gru_output, gru_hidden = topic_gru(input_topic[:, time_step], gru_hidden)  
        topic_dist = softmax(gru_output, dim = 1)
        word_topic_mat = softmax(phi[:, input_loc[:, time_step + 1]].transpose(1, 0), dim = 1)
        sample_prob = torch.distributions.Categorical(softmax(topic_dist * word_topic_mat, dim = 1))
        samples = sample_prob.sample()
        topic_sets[batch_count, :, time_step + 1] = samples
## M step  
      input_topic_new = torch.tensor(topic_sets[batch_count], dtype=torch.long, device=device)  
      loss = 0
      for time_step in range(1, batch.shape[1] - 1):
        gru_output, gru_hidden = topic_gru(input_topic_new[:, time_step], topic_gru_hidden)
        loss += criterion(gru_output, input_topic_new[:, time_step + 1])   
      gru_losses.append(loss.item())   
      loss.backward()
      topic_gru_optimizer.step()
      batch_count += 1  
    phi_raw = torch.zeros(phi.shape[0], phi.shape[1], device = device)           
    for batch, loc_batch in zip(topic_sets, train_sets):
      for topics, tra in zip(batch, loc_batch):
        for loc, topic in zip(tra[1:], topics[1:]):
          phi_raw[topic][loc] += 1
    phi_new = softmax(phi_raw, dim = 1)                      
    print("epoch:", ite, "phi_loss:", (phi_new - phi).abs(), "gru_loss:", np.array(gru_losses).mean() / 50)    
    phi = phi_new
                    
      
def MLE(model):
  pass



def train(dataset, model_name):
  if model_name == "topic":
    train_sets, topic_train_sets, topic_num, loc_size = get_train_data(dataset, model_name)
    epoches = 100     
    pre_epoches = 50
    learning_rate = 0.0001
    pre_learning_rate = 0.0001
    topic_gru = GRU(device, hidden_size = 64, vocab_size = topic_num + 1, embedding_size = 32).to(device)
    loc_gru = GRU(device,  vocab_size = loc_size + 1).to(device)
    phi = torch.ones(topic_num, loc_size, device = device) / topic_num #topic2location
    topic_gru, loc_gru = pretrain_sem(topic_gru, loc_gru, pre_epoches, np.array(train_sets[:20000]), np.array(topic_train_sets[:20000]), pre_learning_rate)
    SEM(topic_gru, phi, epoches, np.array(train_sets[ : 10000]), loc_size, topic_num, learning_rate)
  if model_name == "region":
    train_topic_set, train_loc_set, train_user_set, train_time_set, train_geo_set, loc_num = get_train_data(dataset, model_name)  
    test_topic_set, test_loc_set, test_user_set, test_time_set, test_geo_set = get_test_data(dataset, model_name)
    epoches = 100
    pre_epoches = 50
    learning_rate = 0.0001
    pre_learning_rate = 0.0001
    region_num = 51
    mu = (np.random.rand(region_num, 2) - 0.5) * 10
    cov = (np.random.rand(region_num, 2, 2)) * 5
    for i in range(cov.shape[0]):
      cov[i][0][1] = 0.0
      cov[i][1][0] = 0.0
    loc_num = 20000  
    model = RL_Region(device, mu, cov, region_num = region_num, loc_num = loc_num).to(device)
    rl_pretrain(model, epoches, pre_learning_rate, loc_num, train_topic_set, train_loc_set, train_user_set, train_time_set, train_geo_set, test_topic_set, test_loc_set, test_user_set, test_time_set, test_geo_set)
    rl_train_region(model, epoches, learning_rate, region_num, loc_num, train_topic_set, train_loc_set, train_user_set, train_time_set, train_geo_set, test_topic_set, test_loc_set, test_user_set, test_time_set, test_geo_set)
  if model_name == "multi_mode":
    train_topic_set, train_loc_set, train_user_set, train_time_set, train_geo_set, loc_num = get_train_data(dataset, model_name)  
    test_topic_set, test_loc_set, test_user_set, test_time_set, test_geo_set = get_test_data(dataset, model_name)
    epoches = 50
    pre_epoches = 0
    learning_rate = 0.0001
    pre_learning_rate = 0.0001
    region_num = 51
    topic_num = 201
    mu = (np.random.rand(region_num, 2) - 0.5) * 10
    cov = (np.random.rand(region_num, 2)) * 5 + 1
 #   for i in range(cov.shape[0]):
#      cov[i][0][1] = 0.0
#      cov[i][1][0] = 0.0
    loc_num = 20000  
    model = multi_mode(device, mu, cov, region_num = region_num, topic_num = topic_num, loc_num = loc_num).to(device)
    rl_train_multi_mode(model, epoches, pre_epoches, learning_rate, topic_num, region_num, loc_num, train_topic_set, train_loc_set, train_user_set, train_time_set, train_geo_set, test_topic_set, test_loc_set, test_user_set, test_time_set, test_geo_set)
       
def main():
  train(dataset, model_name)
    
if __name__ == '__main__':
  main()      
