import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import os
import matplotlib.pyplot as plt
import datetime
import pickle
# 
sel_class = [3,5,6,8]

relabel = {}
rcnt =0
for item in sel_class:
	relabel[item]= rcnt
	rcnt += 1

d_nc = len(sel_class)  # pick four digits    # [3,5,6,8]

# this script separate MNIST dataset into two views

(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()

x_train = np.expand_dims(x_train,-1).astype("float32") / 255.0
x_test = np.expand_dims(x_test,-1).astype("float32") / 255.0
y_train = y_train.astype("int")
y_test = y_test.astype("int")

y_train_sel = np.zeros((len(y_train,)))
for idx,item in enumerate(y_train):
	y_train_sel[idx] = int(item in sel_class)
y_train_sel = y_train_sel.astype("bool")

y_test_sel = np.zeros((len(y_test)))
for idx, item in enumerate(y_test):
	y_test_sel[idx] = int(item in sel_class)
y_test_sel = y_test_sel.astype("bool")

new_y_train = np.array([relabel[item] for item in y_train[y_train_sel]]).astype("int")
new_y_test  = np.array([relabel[item] for item in y_test[y_test_sel]]).astype("int")

print("selected number of Y-train=",new_y_train.shape)
print("selected number of Y_test=",new_y_test.shape)
# separate into two views... how...
x_train_v1 = x_train[y_train_sel,:,0:14]
x_train_v2 = x_train[y_train_sel,:,14:]
x_test_v1 = x_test[y_test_sel,:,0:14]
x_test_v2 = x_test[y_test_sel,:,14:]
#x_test_v1 = x_test[:,0:14,0:14]
#x_test_v2 = x_test[:,14:,14:]


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

d_lr= 1e-3

d_ep = 50
d_bs = 128
#optimizer = keras.optimizers.Adam(learning_rate=d_lr)
model_left = fcMnist(nclass=d_nc)
model_left.compile(optimizer=keras.optimizers.Adam(learning_rate=d_lr))
history_left = model_left.fit(x=x_train_v1,y=new_y_train,epochs=d_ep,batch_size=d_bs,shuffle=True,verbose=1)
#print(history_left.history) # {'ce_loss':[...],'accuracy':[....]}

eval_result_left = model_left.evaluate(x=x_test_v1,y=new_y_test,batch_size=d_bs,return_dict=True)
print("Testing Performance for left digit")
print(eval_result_left) #{"ce_loss":val,"accuracy":val}


# let's pick some 
n_pick = 25
#print(y_test[:n_pick])
py = np.ones((d_nc,))
for idx in range(n_pick):
	py[new_y_train[idx]]+=1
py/=np.sum(py)

#logits_left = model_left.predict(x_test_v1) # (10000,10) # p(y|x1)
logits_left = model_left.predict(x_train_v1[:n_pick])
print(logits_left.shape) # logits of the first view
pycx1 = tf.math.softmax(logits_left,axis=1).numpy()
print("pycx1 shape=",pycx1.shape)

model_right = fcMnist(nclass=d_nc)
model_right.compile(optimizer=keras.optimizers.Adam(learning_rate=d_lr))
history_right = model_right.fit(x=x_train_v2,y=new_y_train,epochs=d_ep,batch_size=d_bs,shuffle=True,verbose=1)
eval_result_right = model_right.evaluate(x=x_test_v2,y=new_y_test,batch_size=d_bs,return_dict=True)
print("Testing Performance for the right digit")
print(eval_result_right)
logits_right = model_right.predict(x_train_v2[:n_pick]) # (10000,10) # p(y|x2)
print(logits_right.shape)
pycx2 = tf.math.softmax(logits_right,axis=1).numpy()
print("pycx2 shape=",pycx2.shape)

# assume uniform marginal for the 10 x1,x2
part_pycx1 = pycx1[:n_pick,:].T # because x is row
part_pycx2 = pycx2[:n_pick,:].T
px1 = np.ones((n_pick,))/n_pick
px2 = np.ones((n_pick,))/n_pick

px1cy = (part_pycx1 * px1[None,:] / py[:,None]).T
px2cy = (part_pycx2 * px2[None,:] / py[:,None]).T
# use conditional independence to generate 
# p(x_1,x_2|y) = p(x_1|y)p(x_2|y)
# then the marginal can be calculated 
# \sum_{y}p(y)p(x_1,x_2|y)
px1x2 = np.zeros((n_pick,n_pick))
for ix1 in range(n_pick):
	for ix2 in range(n_pick):
		tmp_sum = 0
		for iy in range(d_nc):
			tmp_sum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
		px1x2[ix1,ix2] = tmp_sum
px1x2 /= np.sum(tmp_sum)

# storing all information for later use
cur_dir= os.getcwd()
timenow = datetime.datetime.now()
datestr = "{:04}{:02}{:02}".format(timenow.year,timenow.month,timenow.day)
mddir_name = "mnist_2v_{:}_npk_{:}".format(datestr,n_pick)
inner_path = os.path.join(cur_dir,mddir_name) 
os.makedirs(inner_path,exist_ok=True)

left_md_name = "mnist_left_{:}_npk_{:}.h5".format(datestr,n_pick)
model_left.save_weights(os.path.join(inner_path,left_md_name))
right_md_name = "mnist_right_{:}_npi_{:}.h5".format(datestr,n_pick)
model_right.save_weights(os.path.join(inner_path,right_md_name))
with open(os.path.join(inner_path,"px1x2_mnist_2v_{:}_n{:}.npy".format(datestr,n_pick)),"wb") as fid:
	np.save(fid,px1x2)

config_dict = {"selected_classes":sel_class,"batch_size":d_bs,"learning_rate":d_lr,"epochs":d_ep,"n_pick":n_pick}
with open(os.path.join(inner_path,"mnist_2v_{:}_n{:}_config.pkl".format(datestr,n_pick)),"wb") as fid:
	pickle.dump(config_dict,fid)

plt.imshow(px1x2,'gray',origin='upper')
plt.show()


