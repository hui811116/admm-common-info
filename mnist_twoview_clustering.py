import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys
import algorithm as alg
import utils as ut
import pickle
import dataset as dt
import tf_model as mds
import evaluation as ev

npk_train = 4

img_height = 28
img_width = 14
img_chs =1
(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()

x_train = np.expand_dims(x_train,-1).astype("float64") / 255.0
x_test = np.expand_dims(x_test,-1).astype("float64") / 255.0
y_train = y_train.astype("int")
y_test = y_test.astype("int")

data_path = os.path.join(os.getcwd(),"mnist_2v_20220926_npk_4")

with open(os.path.join(data_path,"px1x2_mnist_2v_20220926_n4.npy"),"rb") as fid:
	data_prob = np.load(fid)

#data_dict = dt.synToy()
#data_prob = data_dict['p_joint']
data_prob = data_prob.astype("float64")

#data_dict= dt.synExpandToy(2,2,0)
#data_prob = data_dict['p_joint']
data_prob/= np.sum(data_prob)

px1 = np.sum(data_prob,axis=1)
px2 = np.sum(data_prob,axis=0)
px1cx2 = data_prob/px2[None,:]
px2cx1 = (data_prob/px1[:,None]).T

#print("sum prob:{:}".format(np.sum(data_prob)))
with open(os.path.join(data_path,"mnist_2v_20220926_n4_config.pkl"),"rb") as fid:
	data_config = pickle.load(fid)

left_model_path = os.path.join(data_path,"mnist_left_20220926_npk_4.h5")
right_model_path= os.path.join(data_path,"mnist_right_20220926_npk_4.h5")

selected_labels = data_config["selected_classes"]
# loading the generating hidden models
model_left = mds.fcMnist(nclass = len(selected_labels))
model_left(tf.ones((1,img_height,img_width,img_chs))) # dummy call
model_left.load_weights(left_model_path)

model_right = mds.fcMnist(nclass = len(selected_labels))
model_right(tf.ones((1,img_height,img_width,img_chs)))
model_right.load_weights(right_model_path)

# take the testing index
test_sel_idx = []
for idx in range(len(y_test)):
	test_sel_idx.append(y_test[idx] in selected_labels)
test_sel_idx= np.array(test_sel_idx).astype("bool")

x_test_sel = x_test[test_sel_idx]
y_test_sel = y_test[test_sel_idx]

new_y_test = np.array([selected_labels.index(item) for item in y_test_sel]).astype("int")
x_test_v1 = x_test[test_sel_idx,:,0:14]
x_test_v2 = x_test[test_sel_idx,:,14:]
# in testing phase, need to first transform the testing features to probabilistic observations
# this means no dependency on x, the testing vectors are (x1,x2) only
logits_v1 = model_left.predict(x_test_v1)
test_pycx1 = (tf.nn.softmax(logits_v1,axis=1).numpy()).T

logits_v2 = model_right.predict(x_test_v2)
test_pycx2 = (tf.nn.softmax(logits_v2,axis=1).numpy()).T


# take the training indices
train_sel_idx = []
for idx in range(len(y_train)):
	train_sel_idx.append(y_train[idx] in selected_labels)
train_sel_idx = np.array(train_sel_idx).astype("bool")

x_train_sel = x_train[train_sel_idx]
y_train_sel = y_train[train_sel_idx]

new_y_train = np.array([selected_labels.index(item) for item in y_train_sel]).astype("int")

#print(new_y_train[:npk_train])
x_train_v1 = x_train[train_sel_idx,:,0:14]
x_train_v2 = x_train[train_sel_idx,:,14:]

train_logits_left = model_left.predict(x_train_v1[:npk_train])
train_pycx1 = tf.nn.softmax(train_logits_left,axis=1).numpy().T
train_logits_right= model_right.predict(x_train_v2[:npk_train])
train_pycx2 = tf.nn.softmax(train_logits_right,axis=1).numpy().T

test_realization = np.zeros((test_pycx1.shape[1],2))
for ix in range(test_pycx1.shape[1]):
	min_dtv = np.Inf
	min_idx = -1
	for it in range(train_pycx1.shape[1]):
		dtv = 0.5 * np.sum(np.abs(test_pycx1[:,ix]-train_pycx1[:,it]))
		if dtv < min_dtv:
			min_dtv = dtv
			min_idx = it
	test_realization[ix,0] = min_idx

for ix in range(test_pycx2.shape[1]):
	min_dtv = np.Inf
	min_idx = -1
	for it in range(train_pycx2.shape[1]):
		dtv = 0.5 * np.sum(np.abs(test_pycx2[:,ix]-train_pycx2[:,it]))
		if dtv < min_dtv:
			min_dtv = dtv
			min_idx = it 
	test_realization[ix,1] = min_idx
test_realization = test_realization.astype("int")
#print(test_realization[:100,:])
#sys.exit()

# NOT working condition for det
#alg_params = {"ss_init":1e-3,"penalty_coeff":512,"ss_scale":0.25,"maxiter":40000,"convthres":1e-5,"nz":len(selected_labels),"px1x2":data_prob,"seed":None}
#gamma_range = np.geomspace(0.7,1.0,num=5) # this is for sto version

# working condition for sto 
alg_params = {"ss_init":1e-3,"penalty_coeff":128,"ss_scale":0.25,"maxiter":40000,"convthres":1e-5,"nz":len(selected_labels),"px1x2":data_prob,"seed":None}
gamma_range = np.geomspace(1.2,2.0,num=5) # this is for sto version
nrun = 5
for gamma in gamma_range:
	for nn in range(nrun):
		#alg_out = alg.tfDetComAdmm(gamma=gamma,**alg_params)
		# DET numpy
		#alg_out = alg.detComAdmm(gamma=gamma,**alg_params) # the gamma range should < 1
		#alg_out = alg.tfStoComAdmm(gamma=gamma,**alg_params)
		# STO numpy
		alg_out = alg.admmHighDim(gamma=gamma,**alg_params) # the gamma range should > 1
		pzcx1x2 = alg_out['pzcx1x2']
		if alg_out["conv"]:
			pz = alg_out['pz']
			pzcx1 = alg_out['pzcx1']
			pzcx2 = alg_out['pzcx2']
		else:
			pz = np.sum(pzcx1x2 * data_prob[None,:,:],axis=(1,2))
			pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
			pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
		entz = ut.calcEnt(pz)
		#mizx1 = ut.calcMI(pzcx1 * px1[None,:])
		#mizx2 = ut.calcMI(pzcx2 * px2[None,:])
		cmix1x2z = ut.calcMIcond(np.transpose(pzcx1x2 * data_prob[None,:,:],axes=[1,2,0]))
		#cmizx1cx2 = ut.calcMIcond(pzcx1x2 * data_prob[None,:,:])
		# testing evaluation
		out_label, out_acc = ev.clusteringTest2View(new_y_test,test_realization,pzcx1x2)
		print("gamma,{:.5f}, nidx,{:}, conv,{:}, niter,{:}, Hz,{:.5f}, I(X1;X2|Z),{:.5f},acc,{:.5f}".format(gamma,nn,int(alg_out['conv']),alg_out["niter"],entz,cmix1x2z,out_acc))
		#sys.exit()