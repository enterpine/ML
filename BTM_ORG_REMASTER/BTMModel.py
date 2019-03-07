import numpy as np
import codecs
import random

class Biterm(object):
	"""
	 biterm 数据结构，{wid1,wid2,frequence}
	"""
	def __init__(self,wid_1,wid_2):
		"""
		初始化
		:param wid_1: 词1 ID
		:param wid_2: 词2 ID
		"""
		#始终 1id 小于 2id
		if wid_2 < wid_1:
			self.word_id_1 = wid_2
			self.word_id_2 = wid_1
		else:
			self.word_id_1 = wid_1
			self.word_id_2 = wid_2
		self.word_z = None      #biterm 分配的主题
		self.freq = 0

	def get_word(self,index=1):
		if index == 1:
			return self.word_id_1
		elif index == 2:
			return self.word_id_2
		else:
			return None

	def __cmp__(self, other):
		if other.word_id_1 == self.word_id_1 and other.word_id_2 == self.word_id_2:
			return True
		else:
			return False

	def setTopic(self, z):
		"""
		设置 bittem 的主题分布
		:param z:
		:return:
		"""
		self.word_z = z

	def getTopic(self):
		return self.word_z

	def resetTopic(self):
		self.word_z = -1

class BtmModel(object):
	"""
		biterm Topic Model
	"""
	def __init__(self, docs,dictionary,topic_num, iter_times, alpha, beta,has_background=False):
		"""
		初始化 模型对象
		:param voca_size: 词典大小
		:param topic_num: 话题数目
		:param iter_times: 迭代次数
		:param save_step:
		:param alpha: hyperparameters of p(z)
		:param beta:  hyperparameters of p(w|z)
		"""
		self.biterms = list()
		# self.voca_size = voca_size
		self.topic_num = topic_num
		self.n_iter = iter_times

		self.alpha = alpha
		self.beta = beta
		self.docs=docs
		self.dictionary=dictionary
		self.nb_z = list()  # int 类型，n(b|z) ,size topic_num+1
		self.nwz = None     # int 类型矩阵， n(w,z), size topic_num * voca_size
		self.pw_b = list()  # double 词的统计概率分布
		self.has_background = has_background

	def build_word_dic(self,sentences):
		"""
			构建词典,并统计 各词的词频（整个语料中）
		:param sentences:
		:return:
		"""
		self.word_dic = self.dictionary   # 词典 索引
		self.word_fre = {}   # 每个单词在整个文本中存在的频次 7	0.00539854107587557 6	0.0009001317134261325
		word_id = 1

		for sentence in sentences:
			for word in sentence:
				word_index = self.word_dic[word]
				if word_index in self.word_fre:
					self.word_fre[word_index] += 1
				else:
					self.word_fre[word_index] = 1

		self.voca_size = len(self.word_fre.keys())#计入词频统计的单词数量
		sum_val = sum(self.word_fre.values())#所有词频相加，即总单词数
		smooth_val = 0.001

		# 归一化，正则化
		for key in self.word_fre:
			# (频次+0.001) / (总词数+0.001*（主题数+1））获取单词频率
			self.word_fre[key] = (self.word_fre[key] + smooth_val) / (sum_val + smooth_val * (self.topic_num + 1))

		with codecs.open("tmp_btm_word_freq.txt", 'w', encoding='utf8') as fp:
			for key in self.word_fre:
				fp.write("{}\t{}\n".format(key,self.word_fre[key]))
		with open("tmp_btm_word_dic.txt", 'w') as fp:
			for key in self.word_dic:
				# print "{}\t{}".format(str(key), self.word_dic[key])
				fp.write(str(key)+"\t"+str(self.word_dic[key])+"\n")

	def build_wordId(self,sentences):
		"""
		将文本 中word 映射到 word_id 并将结果存储到文件
		:param sentences: 切词后的文档
		:return: 当回 文档的 [wid,...,wid]列表
		"""
		with codecs.open("tmp_btm_word_id.txt",'w',encoding='utf8') as fp:
			for sentence in sentences:
				doc = []
				# print sentence
				for word in sentence:
					doc.append(self.word_dic[word])
				wid_list = [str(wid) for wid in doc]
				# print wid_list
				fp.write(' '.join(wid_list)+"\n")

	def build_Biterms(self, sentence):
		"""
		获取 document 的 biterms
		:param sentence: word id list sentence 是切词后的每一词的ID 的列表
		:return: biterm list
		"""
		#sentence['7', '6', '2', '1', '4', '5', '3']

		win = 15 # 设置窗口大小
		biterms = []
		# with codecs.open("tmp_btm_word_id.txt", 'r', encoding="utf8") as fp:
		# 	sentence = []
		# sentence =
		for i in range(len(sentence)-1):#0-6
			for j in range(i+1, min(i+win+1, len(sentence))): #从i+1 到  7 及 i+16最小值。。
				biterms.append(Biterm(int(sentence[i]),int(sentence[j])))
		# ########################################
		# 76 none
		# 72 none
		# 71 none
		# 74 none
		# 75 none
		# 73 none
		# ########################################
		# 62 none
		# 61 none
		# 64 none
		# 65 none
		# 63 none
		# ########################################
		# 21 none
		# 24 none
		# 25 none
		# 23 none
		# ########################################
		# 14 none
		# 15 none
		# 13 none
		# ########################################
		# 45 none
		# 43 none
		##########################
		# 53 none
		return biterms

	def loadwordId(self,file='tmp_btm_word_id.txt'):
		"""
		获取语料的词 ID
		:param file:
		:return:
		"""
		sentences_wordId = []
		with open(file, 'r') as fp:
			[sentences_wordId.append(line.strip().split(" ")) for line in fp]
		return sentences_wordId

	def staticBitermFrequence(self):
		"""
		统计 biterms 的频率
		:param sentences: 使用word id 表示的 sentence 列表
		:return: 返回corpus 中各个 biterm （wid,wid）: frequence 的频率
		"""
		sentences = []
		with codecs.open("tmp_btm_word_id.txt", 'r', encoding="utf8") as fp:
			sentences = [ line.strip().split(" ") for line in fp]
		#sentences   [['7', '6', '2', '1', '4', '5', '3'], ['29', '13',....]
		self.biterms = []
		for sentence in sentences:
			#sentence ['7', '6', '2', '1', '4', '5', '3']
			bits = self.build_Biterms(sentence) #每个句子 生成一个Biterm 类型的数据，每个句子 两两组合的单词，主题为none
			self.biterms.extend(bits)

		with open("tmp_btm_biterm_freq.txt", 'w') as fp:
			for key in self.biterms:
				#将每个句子中的单词组合写到文件
				fp.write(str(key.get_word())+" "+str(key.get_word(2))+"\n")

	def model_init(self):
		"""
		模型初始化
		:return:
		"""
		# 初始化 话题 biterm 队列和word -topic 矩阵
		self.nb_z = [0]*(self.topic_num+1) #[0, 0, 0, 0, 0, 0, 0, 0] 7个主题就是8个0
		#每个主题在biterms中的数量
		self.nwz = np.zeros((self.topic_num,self.voca_size)) #topic_num行 voca_size列 矩阵
		#第i个主题的中，含第j个单词的数量
		for bit in self.biterms:
			k = random.randint(0, self.topic_num-1)#给每一个biterm  分配0-7之前的随机数
			self.assign_biterm_topic(bit, k)

	def assign_biterm_topic(self, bit, topic_id):
		"""
		为 biterm 赋予 topic ，并更新 相关nwz 及 nb_z 数据
		:param bit:
		:param topic_id:
		:return:
		"""
		w1 = int(bit.get_word())-1
		w2 = int(bit.get_word(2))-1
		bit.setTopic(topic_id)
		self.nb_z[topic_id] += 1
		self.nwz[int(topic_id)][w1] = self.nwz[int(topic_id)][w1] + 1
		self.nwz[int(topic_id)][w2] = self.nwz[int(topic_id)][w2] + 1

	def runModel(self,res_dir="./output/"):
		"""
		运行构建模型
		:param doc_pt: 数据源文件路径
		:param res_dir: 结果存储文件路径
		:return:
		"""
		sentences = self.docs
		self.build_word_dic(sentences) 	#获取单词频率列表
		self.build_wordId(sentences)   	#把分词后文本的单词替换为单词的ID
		self.staticBitermFrequence()   	#生成每个单词在整体词频  以及  将每个句子中的单词组合写到文件
		self.model_init()				#给每对单词组合分配一个随机的主题号 0～7之间
		#print ("Begin iteration")
		out_dir = res_dir + "k" + str(self.topic_num)+'.'
		for iter in range(self.n_iter):
			#print("\r当前迭代{}，总迭代{}".format(iter,self.n_iter))
			for bit in self.biterms:
				self.updateBiterm(bit)

	def updateBiterm(self, bit):
		self.reset_biterm(bit)  #把主题号变为-1
		pz = [0]*self.topic_num	#生成8个0

		self.compute_pz_b(bit, pz)		#更新单词对儿的主题
		# 第k个主题的pz值 = pk（主题频率） * pw1k（单词1频率） * pw2k（单词二频率）

		topic_id = self.mult_sample(pz)	#更新单词对儿的主题

		self.assign_biterm_topic(bit, topic_id) #设置单词对儿的主题

	def compute_pz_b(self, bit, pz):
		"""
		更新 话题的概率分布
		:param bit:
		:param pz:
		:return:
		"""
		w1 = bit.get_word()-1
		w2 = bit.get_word(2)-1
		for k in range(self.topic_num): #0～7
			if self.has_background and k == 0: #has_background默认为false
				pw1k = self.pw_b[w1]
				pw2k = self.pw_b[w2]
			else:
				#pw1k=(第k主题w1词出现词数+beta) / (2*主题k出现次数+单词数*beta)
				pw1k = (self.nwz[k][w1] + self.beta)/ (2*self.nb_z[k] + self.voca_size*self.beta)
				pw2k = (self.nwz[k][w2] + self.beta) / (2 * self.nb_z[k] + 1 + self.voca_size * self.beta)
			#pk = (主题k出现次数+alpha)/(单词对儿数+出题词数*alpha)
			pk = (self.nb_z[k] + self.alpha) / (len(self.biterms) + self.topic_num * self.alpha)

			#第k个主题的pz值 = pk * pw1k * pw2k
			pz[k] = pk * pw1k * pw2k

	def mult_sample(self, pz):
		"""
		sample from mult pz
		:param pz:
		:return:
		"""
		for i in range(1,self.topic_num):#1～8
			pz[i] += pz[i-1]  #pz[1] = pz[1]+pz[0] pz[2]=pz[2]+pz[1] ... pz[8] = pz[8]+pz[7] 转化为前n项和
		u = random.random()
		k = None
		for k in range(0,self.topic_num):
			if pz[k] >= u * pz[self.topic_num-1]:
				break
		if k == self.topic_num:
			k -= 1
		return k

	def show(self, top_num=10):
		print ("BTM topic model \t",)
		print ("topic number {}, voca word size : {}".format(self.topic_num, self.voca_size))
		word_id_dic = {}
		for key in self.word_dic:
			word_id_dic[self.word_dic[key]] = key

		for topic in range(self.topic_num):
			print ("\nTopic: #{}".format(topic),)
			print ("Topic top word \n",)
			b = list(zip(self.nwz[int(topic)],range(self.voca_size)))
			b.sort(key=lambda x: x[0], reverse=True)
			for index in range(top_num):
				print (word_id_dic[b[index][1]+1], b[index][0],)
			print

	def get_topics(self):
		result = list()
		for topic in range(self.topic_num):
			sm = sum(self.nwz[int(topic)])
			tmp = [i/sm for i in self.nwz[int(topic)]]
			result.append(tmp)
		return np.array(result)




	def SentenceProcess(self,doc):
		"""
		文本预处理
		:param sentence: 输入文本
		:return:
		"""
		words = doc
		words_id = []
		# 将文本转换为 word ID
		for w in words:
			if w in list(self.word_dic.keys()):
				words_id.append(self.word_dic[w])
		return self.build_Biterms(words_id)

	def sentence_topic(self, doc, topic_num=1, min_pro=0.01):
		"""
		计算 sentence 最可能的话题属性,基于原始的LDA 方法
		:param sentence: sentence
		:param topic_num: 返回 可能话题数目 最多返回
		:param min_pro: 话题概率最小阈值，只有概率大于该值，才是有效话题，否则不返回
		:return: 返回可能的话题列表，及话题概率
		"""
		words_id = self.SentenceProcess(doc)
		topic_pro = [0.0]*self.topic_num
		sentence_word_dic = [0]*self.voca_size
		weigth = 1.0/len(words_id)
		for word_id in words_id:
			sentence_word_dic[word_id] = weigth
		for i in range(self.topic_num):
			topic_pro[i] = sum(map(lambda x, y: x*y, self.nwz[i], sentence_word_dic))
		sum_pro = sum(topic_pro)
		topic_pro = map(lambda x: x/sum_pro, topic_pro)
		# print topic_pro
		min_result = list(zip(topic_pro, range(self.topic_num)))
		min_result.sort(key=lambda x: x[0], reverse=True)
		result = []
		for re in min_result:
			if re[0] > min_pro:
				result.append(re)
		return result[:topic_num]

	def infer_sentence_topic(self, doc, topic_num=1, min_pro=0):
		"""
		BTM topic model to infer a document or sentence 's topic
		基于 biterm s 计算问题
		:param sentence: sentence
		:param topic_num: 返回 可能话题数目 最多返回
		:param min_pro: 话题概率最小阈值，只有概率大于该值，才是有效话题，否则不返回
		:return: 返回可能的话题列表，及话题概率
		"""
		sentence_biterms = self.SentenceProcess(doc)
		topic_pro = [0]*self.topic_num
		# 短文本分析中，p (b|d) = nd_b/doc(nd_b)  doc(nd_b) 表示 计算的query 的所有biterm的计数
		# 因此，在short text 的p(b|d) 计算为1／biterm的数量
		bit_size = len(sentence_biterms)
		for bit in sentence_biterms:
			# cal p(z|d) = p(z|b)*p(b|d)
			# cal p(z|b)
			pz = [0]*self.topic_num
			self.compute_pz_b(bit, pz)
			pz_sum = sum(pz)
			pz = map(lambda pzk: pzk/pz_sum, pz)
			for x, y in list(zip(range(self.topic_num), pz)):
				topic_pro[x] += y/bit_size
		min_result = list(zip(topic_pro, range(self.topic_num)))
		min_result.sort(key=lambda x: x[0], reverse=True)
		result = []
		for re in min_result:
			if re[0] > min_pro:
				result.append(re)
		return result[:topic_num]

	def get_topic(self, doc):
		"""
		BTM topic model to infer a document or sentence 's topic
		基于 biterm s 计算问题
		:param sentence: sentence
		:param topic_num: 返回 可能话题数目 最多返回
		:return: 返回可能的话题列表，及话题概率
		"""
		sentence_biterms = self.SentenceProcess(doc)
		topic_pro = [0]*self.topic_num
		# 短文本分析中，p (b|d) = nd_b/doc(nd_b)  doc(nd_b) 表示 计算的query 的所有biterm的计数
		# 因此，在short text 的p(b|d) 计算为1／biterm的数量
		bit_size = len(sentence_biterms)
		for bit in sentence_biterms:
			# cal p(z|d) = p(z|b)*p(b|d)
			# cal p(z|b)
			pz = [0]*self.topic_num
			self.compute_pz_b(bit, pz)
			pz_sum = sum(pz)
			pz = map(lambda pzk: pzk/pz_sum, pz)
			for x, y in list(zip(range(self.topic_num), pz)):
				topic_pro[x] += y/bit_size
		min_result = list(topic_pro)
		#min_result.sort(key=lambda x: x[0], reverse=True)
		result = []
		for re in min_result:
			result.append(re)
		return result

	def reset_biterm(self, bit):
		k = bit.getTopic()
		w1 = int(bit.get_word())-1
		w2 = int(bit.get_word(2))-1

		self.nb_z[k] -= 1
		self.nwz[k][w1] -= 1
		self.nwz[k][w2] -= 1
		min_val = -(10**(-7))
		# if self.nb_z[k] > min_val and self.nwz[k][w1] > min_val and
		bit.resetTopic()