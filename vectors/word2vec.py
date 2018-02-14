import tensorflow as tf
import numpy as np
import collections
import random
import math
import logging
import zipfile
import os
import string
from nltk.corpus import stopwords
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# to-do: add dictionary
class Word2Vec:
    def __init__(self,
                 data_path=None,
                 vocabulary_size=10000,
                 skip_window=4,
                 num_skips=8,
                 batch_size=128,
                 embedding_size=128,
                 num_sampled=64,
                 num_steps=10000001,
                 learning_rate=1
                 ):
        """
        To-do: Add a module for minimum-present, which presents vocabularies whose
        frequency is bigger than the minimum amount.
        :param data_path:
        :param vocabulary_size:
        :param skip_window: How many words to consider left and right
        :param num_skips: How many times to reuse an input to generate a label
        :param batch_size: Dimension of the batch vector
        :param embedding_size: Dimension of the embedding vector
        :param num_sampled: Number of negative examples to sample
        """
        self.data_path = data_path
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.num_sampled = num_sampled
        self.num_steps = num_steps
        self.learning_rate = learning_rate

        # To be generated from self.build_dataset(self, words)

        # '$words' store all the words, while '$data' store all the indexes correspondingly
        self.data = list()
        # '$count' stores the frequencies of words
        self.count = [['UNK', -1]]
        # '$dictionary' stores the index of words
        self.dictionary = dict()
        # '$reverse_dictionary' stores the reverse of '$dictionary'
        self.reverse_dictionary = dict()

        self.data_index = 0

        # 'tensors' store all the components in tensorflow
        self.tensors = {}
        # 'data_container' stores all the data in tensorflow
        self.data_container = {}
