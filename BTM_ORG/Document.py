# -*- coding: utf-8 -*-
# writer : lgy
# data : 2017-10-09

from BTM_ORG.Biterm import Biterm

class Document(object):
	"""
		文档类，sentence 类，表示一个长的文本序列
	"""
	def __init__(self,doc):
		"""
			初始化 类
			:param doc :文本变量
		"""
		term = doc.strip().split(" ")
		self.ws = [int(w) for w in term] # word sequence

	def get_word(self, index):
		"""
		extract word from word sequence
		:param index: 词的句子中的位置索引
		:return:
		"""
		if index < len(self.ws):
			return self.ws[index]
		return None

	def gen_biterms(self, win=15):
		"""
		获取 biterms from sentence word sequence
		:param win:
		:return:
		"""
		ws_size = len(self.ws)
		if ws_size < 2:
			return None
		self.biterms = []
		for i in xrange(ws_size-1):
			for j in xrange(i+1,min(ws_size,i + win)):
				self.biterms.append(Biterm(self.ws[i], self.ws[j]))
		return self.biterms
