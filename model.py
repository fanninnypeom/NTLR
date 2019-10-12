import torch
import torch.nn.functional as F
import math
import torch.nn as nn
class GRU(nn.Module):
  def __init__(self, device, hidden_size = 512, vocab_size = 10000, embedding_size = 512, batch_size = 100):
    super(GRU, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.embedding_size = embedding_size
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.gru = nn.GRU(self.embedding_size, hidden_size)
    self.out = nn.Linear(hidden_size, vocab_size)
    self.embedding.weight.data.uniform_(-0.1, 0.1)
  def forward(self, token, hidden):
#    print(self.embedding(token).shape)
    token_emb = self.embedding(token).view(1, -1, self.embedding_size)  
#    target_emb = self.embedding(target).view(1, -1, self.embedding_size)  
#    fuse_emb = torch.cat((token_emb, target_emb), 2)

    output, hidden = self.gru(token_emb, hidden)
#    output = self.out(output[0])
    return output, hidden 

  def initHidden(self):
    return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)

class RL_Topic(nn.Module):
  def __init_(self):
    super(RL_TOpic, self).__init__()
  def forward(self):
    pass
  def initHidden(self):
    pass  

class RL_Region(nn.Module):
  def __init__(self, 
               device, 
               init_mu,
               init_cov,
               region_num = 501, 
               loc_num = 10000 + 1, 
               region_emb_size = 256, 
               loc_emb_size = 512,
               user_num = 900,
               user_emb_size = 256,
               time_num = 100, 
               time_emb_size = 128,
               geo_num = 10000,
               geo_emb_size = 128,
               topic_num = 100,
               topic_emb_size = 96,
               loc_hidden_size = 512, 
               geo_hidden_size = 256, 
               batch_size = 50):
    super(RL_Region, self).__init__()
    self.device = device
    self.loc_hidden_size = loc_hidden_size
    self.loc_emb_size = loc_emb_size
    self.region_emb_size = region_emb_size
    self.topic_emb_size = topic_emb_size
    self.user_emb_size = user_emb_size
    self.time_emb_size = time_emb_size
    self.geo_emb_size = geo_emb_size
    self.topic_emb = nn.Embedding(topic_num, topic_emb_size)
    self.region_emb = nn.Embedding(region_num, region_emb_size)
    self.loc_emb = nn.Embedding(loc_num, loc_emb_size)
    self.user_emb = nn.Embedding(user_num, user_emb_size)
    self.time_emb = nn.Embedding(time_num, time_emb_size)
    self.geo_emb = nn.Embedding(geo_num, geo_emb_size)
    self.region_num = region_num
    self.topic_num = topic_num
    self.loc_num = loc_num
    self.user_num = user_num
    self.time_num = time_num
    self.loc_hidden_size = loc_hidden_size
    self.geo_hidden_size = geo_hidden_size
    self.batch_size = batch_size
    self.geo_gru = nn.GRU(2, geo_hidden_size)
    self.loc_gru = nn.GRU(self.loc_emb_size + self.user_emb_size + self.time_emb_size, loc_hidden_size)
    self.geo_output_layer = nn.Linear(geo_hidden_size, 2)
    self.loc_output_layer = nn.Linear(loc_hidden_size, loc_num)
    self.loc_region_layer = nn.Linear(loc_hidden_size, region_num)
    self.loc_topic_layer = nn.Linear(loc_hidden_size, topic_num)
    self.emb_region_layer = nn.Linear(region_emb_size, region_num)
    self.cat_output_layer = nn.Linear(topic_emb_size + loc_hidden_size, loc_num)
    self.emb_loc_layer = nn.Linear(region_emb_size, loc_num)
    self.mu = torch.nn.Parameter(torch.tensor(init_mu, dtype=torch.float, device=device)) # region_num * 2
    self.cov = torch.nn.Parameter(torch.tensor(init_cov, dtype=torch.float, device=device)) # region_num * 4   a
    self.register_parameter("region_mu", self.mu)
    self.register_parameter("region_cov", self.cov)
    self.region_parameter = torch.distributions.MultivariateNormal(self.mu, self.cov)
    self.softmax = torch.softmax
    self.dropout = torch.nn.Dropout(0.0)
  def forward(self, pretrain, topic_batch, geo_token_batch, geo_batch, loc_batch, user_batch, time_batch, geo_hidden, loc_hidden):    
    loc_batch_emb = self.loc_emb(loc_batch).view(1, -1, self.loc_emb_size) 
    user_batch_emb = self.user_emb(user_batch).view(1, -1, self.user_emb_size)
    time_batch_emb = self.time_emb(time_batch).view(1, -1, self.time_emb_size)
    geo_batch_emb = self.geo_emb(geo_token_batch).view(1, -1, self.geo_emb_size)
    topic_batch_emb = self.topic_emb(topic_batch)
    #self.geo_emb_layer(geo_batch).view(1, -1, self.geo_emb_size)
    self.geo_output, geo_hidden = self.geo_gru(geo_batch.view(1, geo_hidden.shape[1], 2), geo_hidden)
    self.loc_output, loc_hidden = self.loc_gru(torch.cat((loc_batch_emb, user_batch_emb, time_batch_emb), 2), loc_hidden)
    geo_cor = self.geo_output_layer(self.geo_output[0])

    topic_dist = self.loc_topic_layer(self.loc_output[0])
    topic_prob = torch.distributions.Categorical(self.softmax(topic_dist, dim = 1))
    topic_sample = torch.argmax(topic_dist, 1)#topic_prob.sample()
    topic_sample_emb = self.topic_emb(topic_sample).view(-1, self.topic_emb_size)
    topic_action_log_prob = topic_prob.log_prob(topic_sample)

###########    Draw region
#    region_dists_ = [self.region_parameter.log_prob(geo_cor[i, :]) for i in range(geo_batch.shape[0])]  # batch_size * region_num
#    region_dists = torch.stack(region_dists_, 0)
#    region_info = torch.matmul(self.softmax(region_dists, dim = 1), self.region_emb.weight.data)
#    region_pred_dist = self.softmax(self.loc_region_layer(self.loc_output[0].detach()) + self.emb_region_layer(region_info), dim = 1)
#    sample_prob = torch.distributions.Categorical(region_pred_dist)
#    samples = sample_prob.sample()
#    action_log_prob = sample_prob.log_prob(samples) 
#    samples_emb = self.region_emb(samples).view(-1, self.region_emb_size)
##########

    action_log_prob = 0

#    loc_dist = self.loc_output_layer(self.loc_output[0])# + self.emb_loc_layer(samples_emb)
    if pretrain:
      loc_dist = self.cat_output_layer(torch.cat((self.loc_output[0], topic_batch_emb), 1))  
    else:        
      loc_dist = self.cat_output_layer(torch.cat((self.loc_output[0], topic_sample_emb), 1))
    loc_dist_dropout = self.dropout(loc_dist)

    loc_prob = torch.distributions.Categorical(loc_dist)
     
    return topic_dist, topic_action_log_prob, loc_dist_dropout, loc_dist, loc_prob, action_log_prob, geo_hidden, loc_hidden
#    loc_loss += criterion(loc_dist, input_loc[:, time_step + 1])
#    region_loss += torch.multiply(reward, -action_log_prob).sum()    

    
#    output = self.out(output[0])
  def initHidden(self, batch_size):
    return torch.zeros(1, batch_size, self.loc_hidden_size, device=self.device), torch.zeros(1, batch_size, self.geo_hidden_size, device=self.device)

class multi_mode(nn.Module):
  def __init__(self, 
               device, 
               init_mu,
               init_cov,
               region_num = 201, 
               loc_num = 10000 + 1, 
               region_emb_size = 32, 
               loc_emb_size = 512,
               user_num = 900,
               user_emb_size = 256,
               time_num = 100, 
               time_emb_size = 128,
               topic_num = 201,
               topic_emb_size = 32,
               topic_emb_plus = 32,
               loc_hidden_size = 512,
               topic_hidden_size = 512, 
               batch_size = 50):
    super(multi_mode, self).__init__()
    self.device = device
    self.loc_hidden_size = loc_hidden_size
    self.loc_emb_size = loc_emb_size
    self.region_emb_size = region_emb_size
    self.topic_emb_size = topic_emb_size
    self.user_emb_size = user_emb_size
    self.time_emb_size = time_emb_size
    self.topic_emb = nn.Embedding(topic_num, topic_emb_size)
    self.region_emb = nn.Embedding(region_num, region_emb_size)
    self.loc_emb = nn.Embedding(loc_num, loc_emb_size)
    self.user_emb = nn.Embedding(user_num, user_emb_size)
    self.time_emb = nn.Embedding(time_num, time_emb_size)
    self.region_num = region_num
    self.topic_num = topic_num
    self.loc_num = loc_num
    self.user_num = user_num
    self.time_num = time_num
    self.loc_hidden_size = loc_hidden_size
    self.topic_hidden_size = topic_hidden_size
    self.batch_size = batch_size
    self.mode_gru = nn.GRU(self.loc_emb_size + self.user_emb_size + self.time_emb_size, loc_hidden_size)
    self.loc_gru = nn.GRU(self.loc_emb_size + self.user_emb_size + self.time_emb_size, loc_hidden_size)
    self.topic_gru = nn.GRU(self.loc_emb_size + self.user_emb_size + self.time_emb_size, topic_hidden_size)
    self.topic_region_layer = nn.Linear(topic_emb_size, region_num)
    self.loc_region_layer = nn.Linear(loc_hidden_size, region_num)
    self.loc_topic_layer = nn.Linear(loc_hidden_size, topic_num)
    self.loc_region_topic_layer = nn.Linear(topic_hidden_size + region_emb_size, topic_num)
    self.emb_region_layer = nn.Linear(region_emb_size, region_num)
    self.cat_output_layer = nn.Linear(region_emb_size + topic_emb_size + loc_hidden_size, loc_num)
    self.m1_cat_output_layer = nn.Linear(2 * region_emb_size + topic_emb_size + loc_hidden_size, loc_num)
    self.m2_cat_output_layer = nn.Linear(region_emb_size + topic_emb_size + loc_hidden_size, loc_num)
 
    self.loc_output_layer = nn.Linear(loc_hidden_size, loc_num)
    self.topic_output_layer = nn.Linear(topic_emb_size, loc_num)
    self.gmb_topic_emb_layer = nn.Linear(topic_num, topic_emb_size, bias = False)
    self.gmb_region_emb_layer = nn.Linear(region_num, region_emb_size, bias = False)
    self.emb_loc_layer = nn.Linear(region_emb_size, loc_num)
    self.r2t_attn = nn.Linear(region_emb_size + topic_emb_size, 1)
    self.t2r_attn = nn.Linear(region_emb_size + topic_emb_size, 1)
    self.loc_topic_r2t_layer = nn.Linear(loc_hidden_size + topic_emb_size, topic_num)
    self.topic_dim_plus = nn.Linear(topic_emb_size, topic_emb_plus)
    self.mu = torch.nn.Parameter(torch.tensor(init_mu, dtype=torch.float, device=device)) # region_num * 2
    self.cov = torch.nn.Parameter(torch.tensor(init_cov, dtype=torch.float, device=device)) # region_num * 4   a
    self.register_parameter("region_mu", self.mu)
    self.register_parameter("region_cov", self.cov)
#    self.region_parameter = torch.distributions.MultivariateNormal(self.mu, self.cov)
    self.softmax = torch.softmax
    self.topic_thre = torch.nn.Threshold(-0.06, -999.0)
    self.region_thre = torch.nn.Threshold(-0.06, -999.0)

    self.sigmoid = torch.sigmoid
    self.relu = torch.relu
    self.dropout = torch.nn.Dropout(0.0)
#    self.topic_thre = 0.001
#    self.region_thre = 0.000
  def forward(self, pretrain, topic_batch, geo_token_batch, geo_batch, loc_batch, user_batch, time_batch, loc_hidden, topic_hidden):
    self.batch_size = loc_batch.shape[0]
    loc_batch_emb = self.loc_emb(loc_batch).view(1, -1, self.loc_emb_size) 

    user_batch_emb = self.user_emb(user_batch).view(1, -1, self.user_emb_size)
    time_batch_emb = self.time_emb(time_batch).view(1, -1, self.time_emb_size)

#    last_topic_emb = self.topic_emb(last_topic).view(1, -1, self.user_emb_size)
#    last_region_emb = self.region_emb(last_region).view(1, -1, self.user_emb_size)   


    topic_batch_emb = self.topic_emb(topic_batch)
    self.loc_output, loc_hidden = self.loc_gru(torch.cat((loc_batch_emb, user_batch_emb, time_batch_emb), 2), loc_hidden)
    self.topic_output, topic_hidden = self.topic_gru(torch.cat((loc_batch_emb, user_batch_emb, time_batch_emb), 2), topic_hidden)


##############  draw mode and continue variable

#    ctn_dist = self.loc_ctn_layer(self.loc_output[0])
#    mode_prob = torch.distributions.bernoulli.Bernoulli(self.sigmoid(mode_dist))
#    ctn_prob = torch.distribution.bernoulli.Bernoulli(self.sigmoid(ctn_dist))
#    mode_sample = mode_prob.sample()
#    ctn_sample = ctn_prob.sample()
################ mode 1 topic-region   topic2region matrix

    m1_topic_dist = self.loc_topic_layer(self.topic_output[0])
    m1_topic_prob = torch.distributions.Categorical(self.softmax(m1_topic_dist, dim = 1))

    m1_topic_gumbel_sample = F.gumbel_softmax(m1_topic_dist, tau = 0.1, hard = False)  
    m1_topic_gumbel_sample_emb = self.gmb_topic_emb_layer(m1_topic_gumbel_sample)
    m1_region_attn = self.t2r_attn(torch.cat((m1_topic_gumbel_sample_emb.unsqueeze(1).repeat(1, self.region_num, 1), torch.transpose(self.gmb_region_emb_layer.weight, 1, 0).unsqueeze(0).repeat(self.batch_size, 1, 1)), 2)).squeeze(dim = -1)
    m1_t2r_info = self.gmb_region_emb_layer(F.gumbel_softmax(self.topic_region_layer(m1_topic_gumbel_sample_emb), tau = 0.1, hard = False))
#    torch.matmul(self.softmax(m1_region_attn, 1), torch.transpose(self.gmb_region_emb_layer.weight, 1, 0)) 


    m1_stack_geo_batch = geo_batch.view(self.batch_size, 1, 2).repeat(1, self.region_num, 1).view(-1, 2)    
    m1_stack_mu =  self.mu.repeat(self.batch_size, 1)
    m1_stack_cov = self.cov.repeat(self.batch_size, 1)
    
    m1_aaa = -0.5 * torch.matmul((m1_stack_geo_batch - m1_stack_mu).view(-1, 1, 2) * (1 / m1_stack_cov).view(-1, 1, 2), (m1_stack_geo_batch - m1_stack_mu).view(-1, 2, 1))
    m1_aaa = m1_aaa.view(-1, 1)
    m1_bbb = 2 * math.pi * torch.sqrt(m1_stack_cov[:, 0] * m1_stack_cov[:, 1]).view(-1, 1)
    m1_geo_gaussian_dist = m1_aaa / m1_bbb
#    print(m2_region_attn, m2_geo_gaussian_dist.view(self.batch_size, self.region_num).shape)
    m1_region_dist = m1_geo_gaussian_dist.view(self.batch_size, self.region_num) 

    m1_region_prob = torch.distributions.Categorical(self.softmax(m1_region_dist, 1))
    m1_region_sample = m1_region_prob.sample()
    m1_region_gumbel_sample = F.gumbel_softmax(m1_region_dist, tau = 0.1, hard = False)
    m1_region_gumbel_sample_emb = self.gmb_region_emb_layer(m1_region_gumbel_sample)

    m1_region_sample_emb = self.region_emb(m1_region_sample).view(-1, self.region_emb_size)


    m1_loc_dist = self.m1_cat_output_layer(torch.cat((self.loc_output[0], m1_topic_gumbel_sample_emb, m1_region_gumbel_sample_emb, m1_topic_gumbel_sample_emb), 1))
#    m2_loc_dist = self.loc_output_layer(self.loc_output[0])
#    return m1_region_gumbel_sample, m1_topic_gumbel_sample, m1_region_dist, m1_topic_dist, m1_loc_dist, loc_hidden, topic_hidden

################
      
################ mode 2 region-topic
    m2_stack_geo_batch = geo_batch.view(self.batch_size, 1, 2).repeat(1, self.region_num, 1).view(-1, 2)    
    m2_stack_mu =  self.mu.repeat(self.batch_size, 1)
    m2_stack_cov = self.cov.repeat(self.batch_size, 1)
    
    m2_aaa = -0.5 * torch.matmul((m2_stack_geo_batch - m2_stack_mu).view(-1, 1, 2) * (1 / m2_stack_cov).view(-1, 1, 2), (m2_stack_geo_batch - m2_stack_mu).view(-1, 2, 1))
    m2_aaa = m2_aaa.view(-1, 1)
    m2_bbb = 2 * math.pi * torch.sqrt(m2_stack_cov[:, 0] * m2_stack_cov[:, 1]).view(-1, 1)
    m2_geo_gaussian_dist = m2_aaa / m2_bbb
     
    m2_region_dist = m2_geo_gaussian_dist.view(self.batch_size, self.region_num) #* self.loc_region_layer(self.loc_output[0]) 
    m2_region_prob = torch.distributions.Categorical(m2_region_dist)
    m2_region_sample = m2_region_prob.sample()
    m2_region_gumbel_sample = F.gumbel_softmax(m2_region_dist, tau = 0.1, hard = False)
    m2_region_gumbel_sample_emb = self.gmb_region_emb_layer(m2_region_gumbel_sample)
    m2_region_sample_emb = self.region_emb(m2_region_sample).view(-1, self.region_emb_size)
    m2_topic_attn = self.r2t_attn(torch.cat((m2_region_gumbel_sample_emb.unsqueeze(1).repeat(1, self.topic_num, 1), torch.transpose(self.gmb_topic_emb_layer.weight, 1, 0).unsqueeze(0).repeat(self.batch_size, 1, 1)), 2)).squeeze(dim = -1)

    m2_r2t_info = torch.matmul(self.softmax(m2_topic_attn, 1), torch.transpose(self.gmb_topic_emb_layer.weight, 1, 0)) 

    m2_topic_dist = m2_topic_attn + self.loc_topic_layer(self.topic_output[0])
    m2_topic_prob = torch.distributions.Categorical(self.softmax(m2_topic_dist, dim = 1))
    m2_topic_gumbel_sample = F.gumbel_softmax(m2_topic_dist, tau = 0.1, hard = False)  
    m2_topic_gumbel_sample_emb = self.gmb_topic_emb_layer(m2_topic_gumbel_sample)

    m2_loc_dist = self.m2_cat_output_layer(torch.cat((self.loc_output[0], m2_topic_gumbel_sample_emb, m2_region_gumbel_sample_emb), 1))

#############  mode 3  unchange


    mode_dist = self.loc_mode_layer(self.mode_output[0]) 
    mode_gumbel_sample = F.gumbel_softmax(mode_dist, tau = 0.1, hard = False)
    mode_gumbel_sample_emb = self.gmb_mode_emb_layer(mode_gumbel_sample)
    m_loc_dist = torch.einsum("bi,bij->bj", mode_gumbel_sample, torch.cat(m1_loc_dist.unsqueeze(1), m2_loc_dist.unsqueeze(1), 1))


    return m2_region_gumbel_sample, m2_topic_gumbel_sample, m2_region_dist, m2_topic_dist, m_loc_dist, loc_hidden, topic_hidden

#    return f_loc_dist, f_topic_sample, f_region_sample
   
  def initHidden(self, batch_size):
    return torch.zeros(1, batch_size, self.loc_hidden_size, device=self.device), torch.zeros(1, batch_size, self.topic_hidden_size, device=self.device)

   
       
