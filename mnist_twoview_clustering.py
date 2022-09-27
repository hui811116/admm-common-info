import numpy as np
#import tensorflow as tf
#from tensorflow import keras
import os
import sys
import algorithm as alg
import utils as ut
import pickle
import dataset as dt
#import tf_model as mds

img_height = 28
img_width = 14
img_chs =1
#(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()

#x_train = np.expand_dims(x_train,-1).astype("float32") / 255.0
#x_test = np.expand_dims(x_test,-1).astype("float64") / 255.0
#y_train = y_train.astype("int")
#y_test = y_test.astype("int")

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

#left_model_path = os.path.join(data_path,"mnist_left_20220926_npk_25.h5")
#right_model_path= os.path.join(data_path,"mnist_right_20220926_npk_25.h5")

selected_labels = data_config["selected_classes"]
# loading the generating hidden models
#model_left = mds.fcMnist(nclass = len(selected_labels))
#model_left(tf.ones((1,img_height,img_width,img_chs))) # dummy call
#model_left.load_weights(left_model_path)

#model_right = mds.fcMnist(nclass = len(selected_labels))
#model_right(tf.ones((1,img_height,img_width,img_chs)))
#model_right.load_weights(right_model_path)

# in testing phase, need to first transform the testing features to probabilistic observations
# this means no dependency on x, the testing vectors are (x1,x2) only

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
		print("gamma,{:.5f}, nidx,{:}, conv,{:}, niter,{:}, Hz,{:.5f}, I(X1;X2|Z),{:.5f}".format(gamma,nn,int(alg_out['conv']),alg_out["niter"],entz,cmix1x2z))
		#sys.exit()