import tensorflow as tf 
from tensorflow import keras 
import numpy as np 


class MemoryBank:
	def __init__(self, m, feature_dim, filepath):
		self.m = m 
		self.feature_dim = feature_dim
		self.filepath = filepath

		if filepath is not None:
            memory_initializer = tf.constant(pickle.load(open(filepath, "rb")))
        else:
            memory_initializer = tf.random.truncated_normal(feature_dim)

        self.memory_bank = tf.Variable(memory_initializer, trainable=False)


    def save_memory_bank(self):
        if self.filepath and not os.path.isfile(self.filepath):
            with open(self.filepath, "wb") as f:
                pickle.dump(self.memory_bank.numpy(), f)
                print("Memory bank saved as:", self.filename)
        else:
            print("Memory bank empty.")