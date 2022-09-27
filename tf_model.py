import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import sys

class fcMnist(keras.Model):
	def __init__(self,nclass,name=None,**kwargs):
		super(fcMnist,self).__init__(name=name)
		self.img_height=28
		self.img_width =14
		self.img_ch= 1
		self.nclass = nclass
		self.latent_dim = 32

		enc_input = keras.Input(shape=(self.img_height,self.img_width,self.img_ch))
		x = layers.Flatten()(enc_input)
		#x = layers.Reshape(shape=(self.img_height*self.img_width*self.img_ch))(enc_input)
		x = layers.Dense(32,activation="relu")(x)
		enc_out = layers.Dense(self.latent_dim,activation="relu")(x)
		self.encoder = keras.Model(enc_input,enc_out,name="encoder")

		# linear classifier
		dec_input = keras.Input(shape=(self.latent_dim,))
		logits_out = layers.Dense(self.nclass,activation="linear")(dec_input) # linear classifier
		self.decoder = keras.Model(dec_input,logits_out,name="decoder")

		self.ce_loss_tracker = tf.metrics.Mean(name="ce_loss")
		self.acc_tracker = tf.metrics.Accuracy(name="accuracy")
		self.test_ce_loss_tracker = tf.metrics.Mean(name="val_loss")
		self.test_acc_tracker = tf.metrics.Accuracy(name="val_accuracy")
	@property
	def metrics(self):
		return [self.ce_loss_tracker,self.acc_tracker,self.test_ce_loss_tracker,self.test_acc_tracker]
	def train_step(self,data):
		x_train, y_train = data
		with tf.GradientTape() as tape:
			latent_feat = self.encoder(x_train,training=True)
			logits = self.decoder(latent_feat,training=True)
			ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_train,logits)
		grad = tape.gradient(ce_loss,self.trainable_variables)
		self.optimizer.apply_gradients(zip(grad,self.trainable_variables))
		self.ce_loss_tracker.update_state(ce_loss)
		self.acc_tracker.update_state(y_train,tf.math.argmax(logits,axis=1))
		return {"ce_loss":self.ce_loss_tracker.result(),"accuracy":self.acc_tracker.result()}
	def test_step(self,data):
		x_test,y_test = data
		latent_feat = self.encoder(x_test,training=False)
		logits = self.decoder(latent_feat,training=False)
		ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_test,logits)
		self.test_ce_loss_tracker.update_state(ce_loss)
		self.test_acc_tracker.update_state(y_test,tf.math.argmax(logits,axis=1))
		return {"ce_loss":self.test_ce_loss_tracker.result(),"accuracy":self.test_acc_tracker.result()}
	def call(self,data):
		latent_feat = self.encoder(data,training=False)
		logits = self.decoder(latent_feat,training=False)
		return logits # returning logits for soft prediction