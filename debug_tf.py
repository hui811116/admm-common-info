import numpy as np
import sys
import os
import tensorflow as tf
import dataset as dt
import algorithm as alg
import utils as ut
import copy

ny = 2
nb = 2

data_dict = dt.synExpandToy(ny,nb,0)
nz = data_dict['py'].shape[0]
prob_joint = data_dict["p_joint"]
px1 = np.sum(prob_joint,axis=1)
px2 = np.sum(prob_joint,axis=0)

px1cx2 = prob_joint/px2[None,:]
px2cx1 = (prob_joint/px1[:,None]).T

(nx1,nx2) = prob_joint.shape
rng = np.random.default_rng()
pzcx1x2 = rng.random((nz,nx1,nx2))
pzcx1x2/= np.sum(pzcx1x2,axis=0)

copy_pzcx1x2 = copy.deepcopy(pzcx1x2)
print("sum of initialized pzcx1x2")
print(np.sum(pzcx1x2,axis=0))
pz = np.sum(pzcx1x2 * prob_joint[None,:,:],axis=(1,2))
pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2 )
pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:], axis=1)
copy_pz = copy.deepcopy(pz)
copy_pzcx1 = copy.deepcopy(pzcx1)
copy_pzcx2 = copy.deepcopy(pzcx2)


debug_iter = 2
penalty = 512
ss_init = 1e-3
ss_scale =0.25
convthres = 1e-6
gamma = 0.25
dual_z = np.zeros((nz,))
dual_x1 = np.zeros((nz,nx1))
dual_x2 = np.zeros((nz,nx2))
det_dict = {"penalty_coeff":penalty,"ss_init":ss_init,"ss_scale":ss_scale,"seed":None,"init_load":True,
	"pzcx1x2":pzcx1x2, "pz":pz, "pzcx1":pzcx1,"pzcx2":pzcx2,"dual_z":dual_z,"dual_x1":dual_x1,"dual_x2":dual_x2,
}
'''
tf_px1x2 = tf.convert_to_tensor(prob_joint,dtype=tf.float64)
tf_pzcx1x2 = tf.convert_to_tensor(pzcx1x2,dtype=tf.float64)
tf_px1cx2 = tf.convert_to_tensor(px1cx2,dtype=tf.float64)
tf_px2cx1 = tf.convert_to_tensor(px2cx1,dtype=tf.float64)
tf_pz= tf.reduce_sum(tf_pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2))
#print("norm pz",np.linalg.norm(tf_pz.numpy() - pz))
tf_pzcx1 = tf.reduce_sum(tf_pzcx1x2 * tf.transpose(tf_px2cx1)[None,:,:],axis=2)
#print("norm pzcx1",np.linalg.norm(tf_pzcx1.numpy()-pzcx1))
tf_pzcx2 = tf.reduce_sum(tf_pzcx1x2 * tf_px1cx2[None,:,:],axis=1)
#print("norm pzcx2",np.linalg.norm(tf_pzcx2.numpy()- pzcx2))
'''
err_z = rng.random((nz,))
err_x1= rng.random((nz,nx1))
err_x2= rng.random((nz,nx2))

'''
tf_dual_z = tf.convert_to_tensor(dual_z,dtype=tf.float64) 
tf_err_z = tf.convert_to_tensor(err_z,dtype=tf.float64)

tf_dual_x1 = tf.convert_to_tensor(dual_x1,dtype=tf.float64)
tf_err_x1 = tf.convert_to_tensor(err_x1,dtype=tf.float64)
tf_dual_x2 = tf.convert_to_tensor(dual_x2,dtype=tf.float64)
tf_err_x2 = tf.convert_to_tensor(err_x2,dtype=tf.float64)

tf_px1 = tf.convert_to_tensor(px1,dtype=tf.float64)
tf_px2 = tf.convert_to_tensor(px2,dtype=tf.float64)

np_grad_p = gamma * (np.log(pzcx1x2)+1) + (dual_z + penalty * err_z)[:,None,None] + ((dual_x1+penalty*err_x1)/px1[None,:])[:,:,None] + np.repeat(np.expand_dims((dual_x2+penalty*err_x2)/px2[None,:],axis=1),repeats=len(px1),axis=1)
tf_grad_p = gamma * (tf.math.log(pzcx1x2)+1) + (tf_dual_z + penalty * tf_err_z)[:,None,None] + ((tf_dual_x1+penalty*tf_err_x1)/tf_px1[None,:])[:,:,None] + tf.repeat(tf.expand_dims((tf_dual_x2+penalty*tf_err_x2)/tf_px2[None,:],axis=1),repeats=len(px2),axis=1)
print("norm gradp",np.linalg.norm(tf_grad_p.numpy() - np_grad_p))
'''
np_out = alg.detComAdmm(prob_joint,nz,gamma,debug_iter,convthres,**det_dict)
#print(np_out)
#print("sum np pz")
#print(np.sum(np_out['pz']))
#print("sum np pzcx1")
#print(np.sum(np_out['pzcx1'],0))
#print("sum np pzcx2")
#print(np.sum(np_out['pzcx2'],0))
'''
print("np dualz")
print(np_out["dual_z"])
print("np dual_x1")
print(np_out['dual_x1'])
print("np dual_x2")
print(np_out['dual_x2'])
'''
dual_z = np.zeros((nz,))
dual_x1 = np.zeros((nz,nx1))
dual_x2 = np.zeros((nz,nx2))
det_dict = {"penalty_coeff":penalty,"ss_init":ss_init,"ss_scale":ss_scale,"seed":None,"init_load":True,
	"pzcx1x2":copy_pzcx1x2, "pz":copy_pz, "pzcx1":copy_pzcx1,"pzcx2":copy_pzcx2,"dual_z":dual_z,"dual_x1":dual_x1,"dual_x2":dual_x2,
}
tf_out = alg.tfDetComAdmm(prob_joint,nz,gamma,debug_iter,convthres,**det_dict)
'''
print("tf dual_z")
print(tf_out['dual_z'])
print("tf dual_x1")
print(tf_out['dual_x1'])
print('tf dual_x2')
print(tf_out['dual_x2'])
'''