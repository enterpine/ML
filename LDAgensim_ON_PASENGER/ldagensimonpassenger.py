import numpy as np
import logging
import pyLDAvis.gensim
import json
import pandas as pd
import warnings
import jieba
warnings.filterwarnings('ignore')
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from numpy import array
# Import dataset
df = pd.read_csv('../data/buwenminglvke.csv',header=None,sep=',',encoding='GBK').astype(str)
# 2、分词

#从文件导入停用词表
stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()

segment =[]  #存储分词结果

for index,row in df.iterrows():
    content = row[7]
    if content != 'nan':
        #print(content)
        words = jieba.cut(content)
        splitedStr=''
        rowcut=[]
        #print(words)
        for word in words:
            if word not in stpwrdlst:
                splitedStr += word + ' '
                rowcut.append(word)
        segment.append(rowcut)
print(segment)  #生成文档变量
docs=segment    #赋值给docs
dictionary = Dictionary(docs) #生成字典
#dictionary.filter_extremes(no_below=10, no_above=0.2) #字典筛选
print(dictionary)
corpus = [dictionary.doc2bow(doc) for doc in docs] #生成语料库
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))
print(corpus[:1])


num_topics = 3
chunksize = 500
passes = 20
iterations = 400
eval_every = 1
# Make a index to word dictionary.
temp = dictionary[0]  # only to "load" the dictionary.
id2word = dictionary.id2token

lda_model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                       alpha='auto', eta='auto', \
                       iterations=iterations, num_topics=num_topics, \
                       passes=passes, eval_every=eval_every)
# Print the Keyword in the 5 topics
print(lda_model.print_topics())

#Compute Coherence Score using c_v
coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

