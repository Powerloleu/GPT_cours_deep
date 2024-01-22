# Python 3 program to build Bloom Filter 
# Check:
#   'https://en.wikipedia.org/wiki/Bloom_filter'
#   'https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/'
#   'https://www.youtube.com/watch?v=heEDL9usFgs&ab_channel=RobEdwards'

from math import log 
from mmh3 import hash # Set of fast and robust non-cryptographic hash functions invented by Austin Appleby.
from bitarray import bitarray # Efficiently represents an array of booleans.

class BloomFilter(object): 

	''' 
	Class for Bloom filter, using murmur3 hash function 
	'''

	def __init__(self, items_count, fp_prob): 
		''' 
		items_count : int 
			Number of items expected to be stored in bloom filter 
		fp_prob : float 
			False Positive probability in decimal 
		'''
		self.fp_prob = fp_prob 
		self.size = self.get_size(items_count, fp_prob) 
		self.hash_count = self.get_hash_count(self.size, items_count) 
		self.bit_array = bitarray(self.size) 
		self.bit_array.setall(0) 

	def add(self, item): 
		''' 
		Refresh filter with item
		'''
		for i in range(self.hash_count):  
			digest = hash(key = item, seed = i, signed = False) % self.size 
			self.bit_array[digest] = True

	def check(self, item): 
		''' 
		Returns true if positive or false positive 
		'''
		for i in range(self.hash_count): 
			digest = hash(key = item, seed = i, signed = False) % self.size 
			if self.bit_array[digest] == False: 
				return False
		return True

	@classmethod
	def get_size(self, n, p): 
		''' 
		Returns the size of bitarray
		'''
		m = -(n * log(p))/(log(2)**2) 
		return int(m) 

	@classmethod
	def get_hash_count(self, m, n): 
		''' 
		Returns the optimal number of hash functions
		'''
		k = (m/n) * log(2) 
		return int(k) 
