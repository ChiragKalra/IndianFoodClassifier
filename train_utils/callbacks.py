import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


class LossHistory(tf.keras.callbacks.Callback):
	def __init__(self, batches, l_r=(-5, -1)):
		super().__init__()
		self.losses = []
		self.batches = batches
		l, r = l_r
		self.exp_lrs = np.arange(l, r, step=(r - l) / self.batches)

	def on_batch_begin(self, batch, logs={}):
		lr = np.power(10, self.exp_lrs[batch])
		K.set_value(self.model.optimizer.lr, lr)

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs['loss'])


class LRCallBack(tf.keras.callbacks.Callback):
	def __init__(self, epochs, l_r=(-3, -4)):
		super().__init__()
		self.batch_losses = []
		self.epochs = epochs
		l, r = l_r
		self.exp_lrs = np.arange(l, r, step=(r - l) / self.epochs)

	def on_epoch_begin(self, epoch, logs={}):
		lr = np.power(10, self.exp_lrs[epoch])
		K.set_value(self.model.optimizer.lr, lr)
		print('lr set to:', lr)

	def on_batch_end(self, batch, logs={}):
		self.batch_losses.append(logs['loss'])



