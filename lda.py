import jieba

jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)
# 第一个文档分词#
with open('./nlp_test0.txt',encoding='UTF-8') as f:
    document = f.read()

    document_decode = document.encode('gbk')
    document_cut = jieba.cut(document_decode)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document_cut)
    result = result.encode('utf-8')
    with open('./nlp_test1.txt', 'wb+') as f2:
        f2.write(result)
f.close()
f2.close()

# 第二个文档分词#
with open('./nlp_test2.txt',encoding='UTF-8') as f:
    document2 = f.read()

    document2_decode = document2.encode('UTF8')
    document2_cut = jieba.cut(document2_decode)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document2_cut)
    result = result.encode('utf-8')
    with open('./nlp_test3.txt', 'wb+') as f2:
        f2.write(result)
f.close()
f2.close()

# 第三个文档分词#
jieba.suggest_freq('桓温', True)
with open('./nlp_test4.txt',encoding='UTF-8') as f:
    document3 = f.read()

    document3_decode = document3.encode('GBK')
    document3_cut = jieba.cut(document3_decode)
    # print  ' '.join(jieba_cut)
    result = ' '.join(document3_cut)
    result = result.encode('utf-8')
    with open('./nlp_test5.txt', 'wb+') as f3:
        f3.write(result)
f.close()
f3.close()

with open('./nlp_test1.txt',encoding='UTF-8') as f3:
    res1 = f3.read()
print (res1)
with open('./nlp_test3.txt',encoding='UTF-8') as f4:
    res2 = f4.read()
print (res2)
with open('./nlp_test5.txt',encoding='UTF-8') as f5:
    res3 = f5.read()
print (res3)

#从文件导入停用词表
stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
corpus = [res1,res2,res3]
cntVector = CountVectorizer(stop_words=stpwrdlst)
cntTf = cntVector.fit_transform(corpus)
print (cntTf)





lda = LatentDirichletAllocation(n_topics=2,
                                learning_offset=50.,
                                random_state=0)
docres = lda.fit_transform(cntTf)

print (docres)
print (lda.components_)

print(lda)

def print_top_words(model, feature_names, n_top_words):
    #打印每个主题下权重较高的term
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic #%d:" % topic_idx)
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print
    #打印主题-词语分布矩阵
    print (model.components_)

n_top_words=3
tf_feature_names = cntVector.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)