# -*- coding: utf-8 -*-
# writer : lgy
# data : 2017-10-09

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