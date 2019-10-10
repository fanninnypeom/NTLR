import pickle
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
data_path = "/data/wuning/NTLR/beijing/OSMBeijingCompleteTrainData_50"
batch_texts = pickle.load(open(data_path, 'rb'))
texts = []
for batch in batch_texts:
  texts.extend(batch) 
print(texts[0])  
texts = np.array(texts).tolist()
for i in range(len(texts)):
  for j in range(len(texts[i])):
    texts[i][j] = str(texts[i][j]) 
print("--------")    
print(texts[0])
print("-------")             
print(texts[10])
texts = texts
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda = LdaModel(corpus, num_topics=500, per_word_topics = True)
#print(lda.get_topics())
#temp_file = datapath("/data/wuning/NTLR/beijing/topics")
#lda.save(temp_file)
#unknown_corpus = [dictionary.doc2bow(text) for text in texts[:2]]
#print(unknown_corpus[0])
#print(np.array(lda.get_topics()).sum())
#print(lda[unknown_corpus[0]]) 
pickle.dump(lda, open("/data/wuning/NTLR/beijing/topics", "wb"))
