import pickle
import gensim
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from time import *

lda = pickle.load(open("/data/wuning/NTLR/beijing/topics", "rb"))
data_path = "/data/wuning/NTLR/beijing/OSMBeijingCompleteTrainData_50"
batch_texts = pickle.load(open(data_path, 'rb'))
texts = []
length_list = []
for batch in batch_texts:
  length_list.append(len(batch))  
  texts.extend(batch)

texts = np.array(texts).tolist()
for i in range(len(texts)):
  for j in range(len(texts[i])):
    texts[i][j] = str(texts[i][j])
dictionary = Dictionary(texts)
texts = texts
train_topics = []
word2topic = {}
topics = np.argmax(lda.expElogbeta, axis = 0)
for i in range(len(topics)):
  word2topic[dictionary[i]] = topics[i]     
print("good")
for i in range(len(texts)):
  tops = []  
  for j in range(len(texts[i])):
    tops.append(word2topic[texts[i][j]])        
  train_topics.append(tops)     
'''
corpus = [dictionary.doc2bow(text) for text in texts]
temp = lda.get_document_topics(corpus, per_word_topics = True)
temp = [item for item in temp]      
location2id = dictionary.token2id
print("--------")
for i in range(len(texts)):
  print(i)  
  tops = []
  word2topic = {}
  last_topic = 1
  for k in range(len(temp[i][1])):
    if len(temp[i][1][k][1]) == 0:
      begin_time = time()  
      word2topic[temp[i][1][k][0]] = last_topic
      end_time = time() 
      print("time1:", end_time - begin_time)
    else:    
      begin_time = time()  
      zzz = temp[i][1][k][1][0]    
      ind = temp[i][1][k][0]
      end_time = time()
      print("time2:", end_time - begin_time)
      begin_time = time()
      word2topic[ind] = zzz
      end_time = time()
      print("time4:", end_time - begin_time)
    begin_time = time()  
    last_topic = word2topic[temp[i][1][k][0]]
    end_time = time()
    print("time3:", end_time - begin_time)
  for j in range(len(texts[i])):
    tops.append(word2topic[location2id[texts[i][j]]])  
  topics.append(tops)  
'''   
batch_train_topics = []
s_ptr = 0
for item in length_list: 
  batch_train_topics.append(train_topics[s_ptr: s_ptr + item])
    
pickle.dump(batch_train_topics, open("/data/wuning/NTLR/beijing/beijing_topics", "wb"))
#print(temp[1][1],len(temp),len(temp[0][1]))

      
