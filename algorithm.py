import numpy as np
import sys
import os
import gradient_descent as gd
import utils as ut
import tensorflow as tf
import copy
#import scipy as sp
#from scipy.special import softmax

def admmHighDim(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale= kwargs['ss_scale']
	penalty = kwargs['penalty_coeff']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px2 = np.sum(px1x2,0)
	px1 = np.sum(px1x2,1)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T

	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs["pzcx1x2"]
		pz = kwargs["pz"]
		pzcx1 = kwargs["pzcx1"]
		pzcx2 = kwargs["pzcx2"]
		dual_z = kwargs["dual_z"]
		dual_x1 = kwargs["dual_x1"]
		dual_x2 = kwargs["dual_x2"]
	else:
		# random initialization
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

		# augmented variables
		pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)

		# dual vars
		dual_z = np.zeros(pz.shape)
		dual_x1= np.zeros(pzcx1.shape)
		dual_x2= np.zeros(pzcx2.shape)
	# gradient masking
	mask_pzcx1 = np.ones(pzcx1.shape)
	mask_pzcx2 = np.ones(pzcx2.shape)
	mask_pzcx1x2 = np.ones(pzcx1x2.shape)
	itcnt = 0
	conv_flag= False
	while itcnt < maxiter:
		itcnt += 1

		err_z = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		err_x1= np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		err_x2= np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# compute the gradient 

		# NOTE: debugging
		'''
		grad_p = ( (np.log(pzcx1x2)+1) \
				+ (dual_z + penalty * err_z)[:,None,None]\
				+ ((dual_x1 + penalty * err_x1)/px1[None,:])[:,:,None]\
				+np.repeat(np.expand_dims((dual_x2 + penalty * err_x2)/px2[None,:],axis=1),repeats=len(px1),axis=1) )*px1x2[None,:,:]
		'''
		# original
		grad_p = ( (1+gamma) * (np.log(pzcx1x2)+1) \
				+ (dual_z + penalty * err_z)[:,None,None]\
				+ ((dual_x1 + penalty * err_x1)/px1[None,:])[:,:,None]\
				+np.repeat(np.expand_dims((dual_x2 + penalty * err_x2)/px2[None,:],axis=1),repeats=len(px1),axis=1) )*px1x2[None,:,:]

		mean_grad_p = grad_p - np.mean(grad_p,axis=0)
		ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p * mask_pzcx1x2,ss_init,ss_scale)
		if ss_p == 0:
			#print("pzcx1x2 break")
			bad_fibers = np.any(pzcx1x2 - mean_grad_p * 1e-9 >= 1,axis=0)
			mask_pzcx1x2[:,bad_fibers] = 0
			#break
		new_pzcx1x2 = pzcx1x2 - mean_grad_p * ss_p * mask_pzcx1x2
		# new errors
		err_z = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		err_x1= np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		err_x2= np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2

		# daul updates
		dual_z += penalty * err_z
		dual_x1+= penalty * err_x1
		dual_x2+= penalty * err_x2

		# compute the augmented gradients
		grad_z = -(1-gamma) * (np.log(pz)+1) - dual_z - penalty*err_z
		mean_grad_z = grad_z - np.mean(grad_z,axis=0)
		ss_z = gd.naiveStepSize(pz,-mean_grad_z,ss_init,ss_scale)
		if ss_z == 0:
			break
		grad_x1= -gamma  * (np.log(pzcx1)+1)*px1[None,:] - dual_x1 - penalty * err_x1
		mean_grad_x1 = grad_x1 - np.mean(grad_x1,axis=0)
		ss_x1 = gd.naiveStepSize(pzcx1,-mean_grad_x1*mask_pzcx1,ss_z,ss_scale)
		if ss_x1 == 0:
			bad_cols = np.any(pzcx1 - mean_grad_x1 * 1e-9 >= 1,axis=0)
			mask_pzcx1[:,bad_cols] = 0
		grad_x2 = -gamma * (np.log(pzcx2)+1)*px2[None,:] - dual_x2 - penalty * err_x2
		mean_grad_x2 = grad_x2 - np.mean(grad_x2,axis=0)
		ss_x2 = gd.naiveStepSize(pzcx2,-mean_grad_x2*mask_pzcx2,ss_x1,ss_scale)
		if ss_x2 == 0:
			bad_cols = np.any(pzcx2 - mean_grad_x2 * 1e-9 >= 1,axis=0)
			mask_pzcx2[:,bad_cols] = 0
		new_pz = pz - mean_grad_z * ss_x2
		new_pzcx1 = pzcx1 - mean_grad_x1 * ss_x2 * mask_pzcx1
		new_pzcx2 = pzcx2 - mean_grad_x2 * ss_x2 * mask_pzcx2
		# new errors
		err_z = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - new_pz
		err_x1= np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - new_pzcx1
		err_x2= np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - new_pzcx2
		# convergence
		conv_z = 0.5 *np.sum(np.abs(err_z))
		conv_x1 = 0.5 *np.sum(np.abs(err_x1 * mask_pzcx1),axis=0)
		conv_x2 = 0.5 * np.sum(np.abs(err_x2  * mask_pzcx2),axis=0)
		if conv_z< convthres and np.all(conv_x1<convthres) and np.all(conv_x2<convthres):
			conv_flag = True
			break
		else:
			pzcx1x2 = new_pzcx1x2
			pz = new_pz
			pzcx1 = new_pzcx1
			pzcx2 = new_pzcx2
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"dual_z":dual_z,"dual_x1":dual_x1,"dual_x2":dual_x2}
'''
def stoLogAdmm(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	#ss_init = kwargs['ss_init']
	#ss_scale= kwargs['ss_scale']
	ss_fixed = kwargs['ss_init']
	penalty = kwargs['penalty_coeff']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px2 = np.sum(px1x2,0)
	px1 = np.sum(px1x2,1)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T

	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs["pzcx1x2"]
		pz = kwargs["pz"]
		pzcx1 = kwargs["pzcx1"]
		pzcx2 = kwargs["pzcx2"]
		dual_z = kwargs["dual_z"]
		dual_x1 = kwargs["dual_x1"]
		dual_x2 = kwargs["dual_x2"]
	else:
		# random initialization
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

		# augmented variables
		pz = np.sum(pzcx1x2 * px1x2[None,...],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,...],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,...],axis=1)

		# dual vars
		dual_z = np.zeros(pz.shape)
		dual_x1= np.zeros(pzcx1.shape)
		dual_x2= np.zeros(pzcx2.shape)
	# variables
	mlog_pzcx1x2= -np.log(pzcx1x2)
	mlog_pz = -np.log(pz)
	mlog_pzcx1 = -np.log(pzcx1)
	mlog_pzcx2 = -np.log(pzcx2)

	conv_flag = False
	itcnt = 0
	while itcnt < maxiter:
		itcnt +=1 
		# precomputing the error
		exp_mlog_pzcx1x2 = np.exp(-mlog_pzcx1x2)
		exp_mlog_pz = np.exp(-mlog_pz)
		exp_mlog_pzcx1 = np.exp(-mlog_pzcx1)
		exp_mlog_pzcx2 = np.exp(-mlog_pzcx2)
		est_pz = np.sum(exp_mlog_pzcx1x2 * px1x2[None,...],axis=(1,2))
		est_pzcx1 = np.sum(exp_mlog_pzcx1x2 * (px2cx1.T)[None,...],axis=2)
		est_pzcx2 = np.sum(exp_mlog_pzcx1x2 * px1cx2[None,...],axis=1)
		err_z = -np.log(est_pz) - mlog_pz
		err_x1 = -np.log(est_pzcx1) - mlog_pzcx1
		err_x2 = -np.log(est_pzcx2) - mlog_pzcx2
		# gradient of pzcx1x2
		grad_p = -(gamma+1)*exp_mlog_pzcx1x2*((1-mlog_pzcx1x2)*px1x2[None,...]) + \
				(exp_mlog_pzcx1x2 *px1x2[None,...])* np.repeat( np.expand_dims(np.repeat(((dual_z+penalty*err_z)/est_pz)[:,None],repeats=nx1,axis=1),axis=-1),repeats=nx2,axis=-1) +\
				(exp_mlog_pzcx1x2 *(px2cx1.T)[None,...]) * np.repeat( np.expand_dims((dual_x1+penalty*err_x1)/est_pzcx1,axis=-1),repeats=nx2,axis=-1) + \
				(exp_mlog_pzcx1x2 *px1cx2[None,...]) * np.repeat(np.expand_dims((dual_x2+penalty*err_x2)/est_pzcx2,axis=1),repeats=nx1,axis=1)
		raw_mlog_pzcx1x2 = mlog_pzcx1x2 - ss_fixed * grad_p
		raw_mlog_pzcx1x2 -= np.amin(raw_mlog_pzcx1x2,axis=0)
		raw_pzcx1x2 = np.exp(-raw_mlog_pzcx1x2) + 1e-9
		new_mlog_pzcx1x2 =-np.log(raw_pzcx1x2 / np.sum(raw_pzcx1x2,axis=0,keepdims=True))

		# update estimates
		exp_mlog_pzcx1x2 = np.exp(-new_mlog_pzcx1x2)
		est_pz = np.sum(exp_mlog_pzcx1x2*px1x2[None,...],axis=(1,2))
		est_pzcx1 = np.sum(exp_mlog_pzcx1x2*(px2cx1.T)[None,...],axis=2)
		est_pzcx2 = np.sum(exp_mlog_pzcx1x2*px1cx2[None,...],axis=1)
		err_z = -np.log(est_pz) - mlog_pz
		err_x1 = -np.log(est_pzcx1) - mlog_pzcx1
		err_x2 = -np.log(est_pzcx2) - mlog_pzcx2

		#dual updates
		dual_z += penalty * err_z 
		dual_x1 += penalty * err_x1
		dual_x2 += penalty* err_x2
		# grad_z
		grad_z = (1-gamma)*exp_mlog_pz *(1-mlog_pz) - (dual_z+penalty*err_z)
		raw_mlog_pz = mlog_pz - ss_fixed * grad_z
		raw_mlog_pz -= np.amin(raw_mlog_pz)
		raw_pz = np.exp(-raw_mlog_pz) + 1e-9
		new_mlog_pz = -np.log(raw_pz/np.sum(raw_pz,keepdims=True))
		# grad_x1
		grad_x1 = gamma * exp_mlog_pzcx1 * (1-mlog_pzcx1) - (dual_x1 + penalty*err_x1)
		raw_mlog_pzcx1 = mlog_pzcx1 - ss_fixed * grad_x1
		raw_mlog_pzcx1 -= np.amin(raw_mlog_pzcx1,axis=0)
		raw_pzcx1 = np.exp(-raw_mlog_pzcx1) + 1e-9
		new_mlog_pzcx1 = -np.log(raw_pzcx1/np.sum(raw_pzcx1,axis=0,keepdims=True))
		# grad_x2
		grad_x2 = gamma * exp_mlog_pzcx2 * (1-mlog_pzcx2) - (dual_x2 + penalty * err_x2)
		raw_mlog_pzcx2 = mlog_pzcx2 - ss_fixed * grad_x2
		raw_mlog_pzcx2 -= np.amin(raw_mlog_pzcx2,axis=0)
		raw_pzcx2 = np.exp(-raw_mlog_pzcx2) + 1e-9
		new_mlog_pzcx2 = -np.log(raw_pzcx2/np.sum(raw_pzcx2,axis=0,keepdims=True))

		# convergence
		conv_z = 0.5 * np.sum(np.fabs(est_pz - np.exp(-new_mlog_pz)))
		conv_x1 = 0.5 * np.sum(np.fabs(est_pzcx1 - np.exp(-new_mlog_pzcx1)),axis=0)
		conv_x2 = 0.5 * np.sum(np.fabs(est_pzcx2 - np.exp(-new_mlog_pzcx2)),axis=0)
		if conv_z < convthres and np.all(conv_x1<convthres) and np.all(conv_x2<convthres):
			conv_flag = True
			break
		else:
			mlog_pzcx1x2 = new_mlog_pzcx1x2
			mlog_pz = new_mlog_pz
			mlog_pzcx1 = new_mlog_pzcx1
			mlog_pzcx2 = new_mlog_pzcx2
	pzcx1x2 = np.exp(-mlog_pzcx1x2)
	pz = np.sum(pzcx1x2 * px1x2[None,...],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,...],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,...],axis=1)

	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"dual_z":dual_z,"dual_x1":dual_x1,"dual_x2":dual_x2}
'''

def detComAdmm(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs["ss_init"]
	ss_scale = kwargs["ss_scale"]
	penalty = kwargs["penalty_coeff"]
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px2 = np.sum(px1x2,0)
	px1 = np.sum(px1x2,1)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T

	#_debug_alpha = 0
	#_debug_alpha_scale = 1- 1e-4

	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
		pz = kwargs['pz']
		pzcx1 = kwargs['pzcx1']
		pzcx2 = kwargs['pzcx2']
		dual_z = kwargs['dual_z']
		dual_x1 = kwargs['dual_x1']
		dual_x2 = kwargs['dual_x2']
	else:
		# random initialization
		pzcx1x2 = rng.random((nz,nx1,nx2),dtype="float64")
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

		# precomputed variables
		pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)

		# dual variables
		dual_z = np.zeros((len(pz),))
		dual_x1 = np.zeros((nz,len(px1),))
		dual_x2 = np.zeros((nz,len(px2),))
	mask_pz = np.ones(pz.shape)
	mask_pzcx1 = np.ones(pzcx1.shape)
	mask_pzcx2 = np.ones(pzcx2.shape)
	mask_pzcx1x2 = np.ones(pzcx1x2.shape)
	conv_flag = False
	itcnt =0
	while itcnt < maxiter:
		itcnt +=1
		# errors
		err_z = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		err_x1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		err_x2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# compute the gradient of the convex part
		# -gamma H(Z|X_1,X_2) + <dual,penalty>
		grad_p = ( gamma * (np.log(pzcx1x2)+1)\
				 + (dual_z+penalty*err_z)[:,None,None]\
				 + ((dual_x1+penalty*err_x1)/px1[None,:])[:,:,None]\
				 + np.repeat(np.expand_dims((dual_x2+penalty*err_x2)/px2[None,:],1),repeats=nx1,axis=1) )*px1x2[None,:,:]
		#print(np.repeat(np.expand_dims((dual_x2+penalty*err_x2)/px2[None,:],1),repeats=nx1,axis=1))
		mean_grad_p = grad_p - np.mean(grad_p,axis=0)

		ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p * mask_pzcx1x2,ss_init,ss_scale)
		if ss_p == 0:

			bad_fibers = np.any(pzcx1x2 - mean_grad_p * 1e-7 >= 1.0, axis=0)
			mask_pzcx1x2[:,bad_fibers] = 0
			# pass
		new_pzcx1x2 = pzcx1x2 - ss_p * mean_grad_p * mask_pzcx1x2
		# new error
		err_z = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		err_x1 = np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		err_x2 = np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# dual update
		dual_z += penalty * err_z
		dual_x1 += penalty * err_x1
		dual_x2 += penalty * err_x2

		# grad z, compute (1-gamma)H(Z)
		if not  np.all(pz == pz):
			print(pz)
			sys.exit("bug")
		grad_z = -(1-gamma) * (np.log(pz)+1) - dual_z - penalty * err_z
		mean_grad_z = grad_z - np.mean(grad_z,axis=0)
		# grad x1, compute -gamma H(Z|X_1)
		grad_x1 = -gamma * (np.log(pzcx1)+1) * px1[None,:] - dual_x1 - penalty * err_x1
		mean_grad_x1 = grad_x1 - np.mean(grad_x1,axis=0)
		grad_x2 = -gamma * (np.log(pzcx2)+1) * px2[None,:] - dual_x2 - penalty * err_x2
		mean_grad_x2 = grad_x2 - np.mean(grad_x2,axis=0)
		ss_z = gd.naiveStepSize(pz,-mean_grad_z * mask_pz, ss_init,ss_scale)
		if ss_z == 0:
			if np.any(pz - mean_grad_z * 1e-7 >= 1.0):
				mask_pz = np.zeros(pz.shape)
		ss_x1 = gd.naiveStepSize(pzcx1,-mean_grad_x1 * mask_pzcx1,ss_z,ss_scale)
		if ss_x1 == 0:
			bad_cols = np.any(pzcx1 - mean_grad_x1 * 1e-7 >= 1.0 ,axis=0)
			mask_pzcx1[:,bad_cols] = 0
		ss_x2 = gd.naiveStepSize(pzcx2,-mean_grad_x2 * mask_pzcx2,ss_x1,ss_scale)
		
		if ss_x2 == 0:
			bad_cols = np.any(pzcx2 - mean_grad_x2 * 1e-7 >= 1.0, axis=0)
			mask_pzcx2[:,bad_cols] = 0
		new_pz = pz - mean_grad_z * ss_x2 * mask_pz
		new_pzcx1 = pzcx1 - mean_grad_x1 * ss_x2 * mask_pzcx1
		new_pzcx2 = pzcx2 - mean_grad_x2 * ss_x2 * mask_pzcx2

		# error
		err_z = np.sum(new_pzcx1x2* px1x2[None,:,:],axis=(1,2)) - new_pz
		err_x1 = np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=2) - new_pzcx1
		err_x2 = np.sum(new_pzcx1x2 *(px2cx1.T)[None,:,:],axis=1) - new_pzcx2

		# debugging
		conv_z = 0.5 *np.sum(np.abs(err_z * mask_pz))
		conv_x1 = 0.5 *np.sum(np.abs(err_x1 * mask_pzcx1),axis=0)
		conv_x2 = 0.5 * np.sum(np.abs(err_x2 * mask_pzcx2),axis=0)
		if conv_z<convthres and np.all(conv_x1<convthres) and np.all(conv_x2<convthres):
			conv_flag = True
			break
		else:
			# last residuals
			pzcx1x2 = new_pzcx1x2
			pz = new_pz
			pzcx1 = new_pzcx1
			pzcx2 = new_pzcx2
	# NOTE:
	# if need to map to deterministic encoder, then use argmax operation
	# det_pzcx1x2 = np.max(pzcx1x2,axis=0,keepdims=True) == pzcx1x2
	# det_pzcx1x2 = det_pzcx1x2.astype("float32") + 1e-7
	# det_pzcx1x2/= np.sum(det_pzcx1x2,axis=0,keepdims=True)
	return {"conv":conv_flag, "niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"dual_z":dual_z,"dual_x1":dual_x1,"dual_x2":dual_x2}
'''
def tfDetComAdmm(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs["ss_init"]
	ss_scale = kwargs["ss_scale"]
	penalty = kwargs['penalty_coeff']
	if kwargs["seed"]:
		rng = tf.random.Generator.from_seed(kwargs["seed"])
	else:
		rng = tf.random.Generator.from_non_deterministic_state()
	tf_px1x2 = tf.convert_to_tensor(px1x2,dtype=tf.float64)
	(nx1,nx2) = px1x2.shape
	px1 = tf.reduce_sum(tf_px1x2,axis=1)
	px2 = tf.reduce_sum(tf_px1x2,axis=0)
	px1cx2 = tf_px1x2/px2[None,:]
	px2cx1 = tf.transpose(tf_px1x2/px1[:,None])
	if "init_load" in kwargs.keys():
		pzcx1x2 = tf.convert_to_tensor(kwargs["pzcx1x2"],dtype=tf.float64)
		pz = tf.convert_to_tensor(kwargs["pz"],dtype=tf.float64)
		pzcx1 = tf.convert_to_tensor(kwargs["pzcx1"],dtype=tf.float64)
		pzcx2 = tf.convert_to_tensor(kwargs["pzcx2"],dtype=tf.float64)
		dual_z = tf.convert_to_tensor(kwargs["dual_z"],dtype=tf.float64)
		dual_x1 = tf.convert_to_tensor(kwargs["dual_x1"],dtype=tf.float64)
		dual_x2 = tf.convert_to_tensor(kwargs["dual_x2"],dtype=tf.float64)
	else:
		# random initialization
		pzcx1x2 = tf.nn.softmax(rng.normal(shape=(nz,nx1,nx2),dtype=tf.float64),axis=0)
		# augmented variables
		pz = tf.reduce_sum(pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2))
		pzcx1 = tf.reduce_sum(pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2)
		pzcx2 = tf.reduce_sum(pzcx1x2 * px1cx2[None,:,:],axis=1)

		# dual vars
		dual_z = tf.zeros(pz.shape,dtype=tf.float64)
		dual_x1= tf.zeros(pzcx1.shape,dtype=tf.float64)
		dual_x2= tf.zeros(pzcx2.shape,dtype=tf.float64)
	conv_flag = False
	itcnt = 0
	while itcnt < maxiter:
		itcnt += 1
		err_z = tf.reduce_sum(pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2)) - pz
		err_x1 = tf.reduce_sum(pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2) - pzcx1
		err_x2 = tf.reduce_sum(pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# errors
		# gradient of convex part
		grad_p = (gamma * (tf.math.log(pzcx1x2)+1)\
				+ (dual_z + penalty * err_z)[:,None,None]\
				+ ((dual_x1 + penalty * err_x1)/px1[None,:])[:,:,None]\
				+ tf.repeat(tf.expand_dims((dual_x2+penalty*err_x2)/px2[None,:],1),repeats=nx1,axis=1) )*tf_px1x2[None,:,:]
		
		mean_grad_p = grad_p - tf.reduce_mean(grad_p,axis=0)
		ss_p = gd.tfNaiveSS(pzcx1x2,-mean_grad_p,ss_init,ss_scale)
		if ss_p == 0:
			break
		new_pzcx1x2 = pzcx1x2 - ss_p * mean_grad_p

		# new error
		err_z = tf.reduce_sum(new_pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2)) - pz
		err_x1 = tf.reduce_sum(new_pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2) - pzcx1
		err_x2 = tf.reduce_sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# dual update
		dual_z += penalty * err_z
		dual_x1 += penalty * err_x1
		dual_x2 += penalty * err_x2

		# grad z, compute (1-gamma)H(Z)
		grad_z = -(1-gamma) * (tf.math.log(pz)+1) - dual_z - penalty * err_z
		mean_grad_z = grad_z - tf.reduce_mean(grad_z,axis=0)
		# grad x1, compute -gamma H(Z|X_1)
		grad_x1 = -gamma * (tf.math.log(pzcx1)+1)* px1[None,:] - dual_x1 - penalty * err_x1
		mean_grad_x1 = grad_x1 - tf.reduce_mean(grad_x1,axis=0)
		grad_x2 = -gamma * (tf.math.log(pzcx2)+1) * px2[None,:] - dual_x2 - penalty * err_x2
		mean_grad_x2 = grad_x2 - tf.reduce_mean(grad_x2,axis=0)
		ss_z = gd.tfNaiveSS(pz,-mean_grad_z,ss_init,ss_scale)
		if ss_z == 0:
			break
		ss_x1 = gd.tfNaiveSS(pzcx1,-mean_grad_x1, ss_z, ss_scale)
		if ss_x1 == 0:
			break
		ss_x2 = gd.tfNaiveSS(pzcx2,-mean_grad_x2,ss_x1,ss_scale)
		if ss_x2 == 0:
			break
		new_pz = pz - mean_grad_z * ss_x2
		new_pzcx1 = pzcx1 - mean_grad_x1 * ss_x2
		new_pzcx2 = pzcx2 - mean_grad_x2 * ss_x2

		# error
		err_z = tf.reduce_sum(new_pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2)) - new_pz
		err_x1 = tf.reduce_sum(new_pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2) - new_pzcx1
		err_x2 = tf.reduce_sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - new_pzcx2
		conv_z = 0.5 * tf.reduce_sum(tf.math.abs(err_z))
		conv_x1 = 0.5 * tf.reduce_sum(tf.math.abs(err_x1),axis=0)
		conv_x2 = 0.5 * tf.reduce_sum(tf.math.abs(err_x2),axis=0)
		if tf.reduce_all(conv_z < convthres) and tf.reduce_all(conv_x1 < convthres) and tf.reduce_all(conv_x2 < convthres):
			conv_flag = True
			break
		else:
			pzcx1x2 = new_pzcx1x2
			pz = new_pz
			pzcx1 = new_pzcx1
			pzcx2 = new_pzcx2

	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2.numpy(),
		"pz":pz.numpy(),"pzcx1":pzcx1.numpy(),"pzcx2":pzcx2.numpy(),
		"dual_z":dual_z.numpy(),"dual_x1":dual_x1.numpy(),"dual_x2":dual_x2.numpy(),}
'''
'''
def tfStoComAdmm(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale= kwargs['ss_scale']
	penalty = kwargs['penalty_coeff']
	#d_seed = None
	if kwargs["seed"]:
		rng = tf.random.Generator.from_seed(kwargs["seed"])
	else:
		rng = tf.random.Generator.from_non_deterministic_state()
	tf_px1x2 = tf.convert_to_tensor(px1x2,dtype=tf.float64)
	(nx1,nx2) = px1x2.shape
	#px2 = np.sum(px1x2,0)
	#px1 = np.sum(px1x2,1)
	px1 = tf.reduce_sum(tf_px1x2,axis=1)
	px2 = tf.reduce_sum(tf_px1x2,axis=0)
	px1cx2 = tf_px1x2 /px2[None,:]
	px2cx1 = tf.transpose(tf_px1x2/ px1[:,None])

	if "init_load" in kwargs.keys():
		pzcx1x2 = tf.convert_to_tensor(kwargs["pzcx1x2"],dtype=tf.float64)
		pz = tf.convert_to_tensor(kwargs["pz"],dtype=tf.float64)
		pzcx1 = tf.convert_to_tensor(kwargs["pzcx1"],dtype=tf.float64)
		pzcx2 = tf.convert_to_tensor(kwargs["pzcx2"],dtype=tf.float64)
		dual_z = tf.convert_to_tensor(kwargs["dual_z"],dtype=tf.float64)
		dual_x1 = tf.convert_to_tensor(kwargs["dual_x1"],dtype=tf.float64)
		dual_x2 = tf.convert_to_tensor(kwargs["dual_x2"],dtype=tf.float64)
	else:
		# random initialization
		pzcx1x2 = tf.nn.softmax(rng.normal(shape=(nz,nx1,nx2),dtype=tf.float64),axis=0)
		#pzcx1x2 = rng.random((nz,nx1,nx2))
		#pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

		# augmented variables
		pz = tf.reduce_sum(pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2))
		pzcx1 = tf.reduce_sum(pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2)
		pzcx2 = tf.reduce_sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
		#pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		#pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		#pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)

		# dual vars
		dual_z = tf.zeros(pz.shape,dtype=tf.float64)
		dual_x1= tf.zeros(pzcx1.shape,dtype=tf.float64)
		dual_x2= tf.zeros(pzcx2.shape,dtype=tf.float64)
	#print(pzcx1x2)
	#print(pzcx1.shape)
	#print(pzcx2.shape)
	#sys.exit()
	# masks
	mask_z = tf.ones(pz.shape,dtype=tf.float64)
	mask_x1 = tf.ones(pzcx1.shape,dtype=tf.float64)
	mask_x2 = tf.ones(pzcx2.shape,dtype=tf.float64)
	mask_p = tf.ones(pzcx1x2.shape,dtype=tf.float64)
	conv_flag = False
	itcnt = 0
	while itcnt < maxiter:
		itcnt += 1
		#err_z = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		err_z = tf.reduce_sum(pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2)) - pz
		#err_x1= np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		err_x1 = tf.reduce_sum(pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2) - pzcx1
		#err_x2= np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		err_x2 = tf.reduce_sum(pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# compute the gradient 

		#grad_p = ( (1+gamma) * (np.log(pzcx1x2)+1) \
		#		+ (dual_z + penalty * err_z)[:,None,None]\
		#		+ ((dual_x1 + penalty * err_x1)/px1[None,:])[:,:,None]\
		#		+np.repeat(np.expand_dims((dual_x2 + penalty * err_x2)/px2[None,:],axis=1),repeats=len(px1),axis=1) )*px1x2[None,:,:]
		grad_p = ( (1+gamma) * (tf.math.log(pzcx1x2)+1)\
				+ (dual_z + penalty * err_z)[:,None,None]\
				+ ((dual_x1 + penalty * err_x1)/px1[None,:])[:,:,None]\
				+ tf.repeat(tf.expand_dims((dual_x2 + penalty * err_x2)/px2[None,:],axis=1),repeats=len(px1),axis=1) )*tf_px1x2[None,:,:]

		#mean_grad_p = grad_p - np.mean(grad_p,axis=0)
		mean_grad_p = grad_p - tf.reduce_mean(grad_p,axis=0)
		#ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p,ss_init,ss_scale)
		ss_p = gd.tfNaiveSS(pzcx1x2,-mean_grad_p * mask_p,ss_init,ss_scale)
		if ss_p == 0:
			bad_fibers = tf.reduce_any(pzcx1x2 - mean_grad_p * 1e-9 >= 1.0, axis=0)
			pzcx1x2[:,bad_fibers] = 0
			#break
		new_pzcx1x2 = pzcx1x2 - mean_grad_p * ss_p * mask_p
		# new errors
		#err_z = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - pz
		#err_x1= np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - pzcx1
		#err_x2= np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		err_z = tf.reduce_sum(new_pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2)) - pz
		err_x1 = tf.reduce_sum(new_pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2) - pzcx1
		err_x2 = tf.reduce_sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - pzcx2
		# daul updates
		dual_z += penalty * err_z
		dual_x1+= penalty * err_x1
		dual_x2+= penalty * err_x2

		# compute the augmented gradients
		#grad_z = -(1-gamma) * (np.log(pz)+1) - dual_z - penalty*err_z
		#mean_grad_z = grad_z - np.mean(grad_z,axis=0)
		#ss_z = gd.naiveStepSize(pz,-mean_grad_z,ss_init,ss_scale)
		grad_z = -(1-gamma) * (tf.math.log(pz)+1) - dual_z - penalty * err_z
		mean_grad_z = grad_z - tf.reduce_mean(grad_z,axis=0)
		grad_x1 = -gamma * (tf.math.log(pzcx1)+1)*px1[None,:] - dual_x1 - penalty * err_x1
		mean_grad_x1 = grad_x1 - tf.reduce_mean(grad_x1,axis=0)
		grad_x2 = -gamma * (tf.math.log(pzcx2)+1)*px2[None,:] - dual_x2 - penalty * err_x2
		mean_grad_x2 = grad_x2 - tf.reduce_mean(grad_x2,axis=0)
		ss_z = gd.tfNaiveSS(pz,-mean_grad_z * mask_z,ss_init,ss_scale)
		if ss_z == 0:
			#break
			if tf.reduce_any(pz - mean_grad_z * 1e-9 >= 1.0):
				mask_z = tf.zeros(pz.shape)
			
		#grad_x1= -gamma  * (np.log(pzcx1)+1) - dual_x1 - penalty * err_x1
		#mean_grad_x1 = grad_x1 - np.mean(grad_x1,axis=0)
		#ss_x1 = gd.naiveStepSize(pzcx1,-mean_grad_x1,ss_z,ss_scale)
		ss_x1 = gd.tfNaiveSS(pzcx1,-mean_grad_x1 * mask_x1,ss_z,ss_scale)
		if ss_x1 == 0:
			#break
			bad_cols = tf.reduce_any(pzcx1 - mean_grad_x1 * 1e-9 >= 1.0)
			mask_x1[:,bad_cols] = 0 
		#grad_x2 = -gamma * (np.log(pzcx2)+1) - dual_x2 - penalty * err_x2
		#mean_grad_x2 = grad_x2 - np.mean(grad_x2,axis=0)
		#ss_x2 = gd.naiveStepSize(pzcx2,-mean_grad_x2,ss_x1,ss_scale)
		ss_x2 = gd.tfNaiveSS(pzcx2,-mean_grad_x2 * mask_x2,ss_x1,ss_scale)
		if ss_x2 == 0:
			#break
			bad_cols = tf.reduce_any(pzcx2 - mean_grad_x2 * 1e-9 >= 1.0)
			mask_x2[:,bad_cols] = 0
		new_pz = pz - mean_grad_z * ss_x2 * mask_z
		new_pzcx1 = pzcx1 - mean_grad_x1 * ss_x2 * mask_x1
		new_pzcx2 = pzcx2 - mean_grad_x2 * ss_x2 * mask_x2
		# new errors
		#err_z = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2)) - new_pz
		#err_x1= np.sum(new_pzcx1x2 * (px2cx1.T)[None,:,:],axis=2) - new_pzcx1
		#err_x2= np.sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - new_pzcx2
		err_z = tf.reduce_sum(new_pzcx1x2 * tf_px1x2[None,:,:],axis=(1,2)) - new_pz
		err_x1 = tf.reduce_sum(new_pzcx1x2 * tf.transpose(px2cx1)[None,:,:],axis=2) - new_pzcx1
		err_x2 = tf.reduce_sum(new_pzcx1x2 * px1cx2[None,:,:],axis=1) - new_pzcx2
		# convergence
		#conv_z = np.sum(np.abs(err_z))
		#conv_x1 = np.sum(np.abs(err_x1),axis=0)
		#conv_x2 = np.sum(np.abs(err_x2),axis=0)
		conv_z = 0.5*tf.reduce_sum(tf.math.abs(err_z * mask_z))
		conv_x1 = 0.5*tf.reduce_sum(tf.math.abs(err_x1 * mask_x1),axis=0)
		conv_x2 = 0.5*tf.reduce_sum(tf.math.abs(err_x2 * mask_x2),axis=0)
		if tf.reduce_all(conv_z< convthres) and tf.reduce_all(conv_x1<convthres) and tf.reduce_all(conv_x2<convthres):
			conv_flag = True
			break
		else:
			pzcx1x2 = new_pzcx1x2
			pz = new_pz
			pzcx1 = new_pzcx1
			pzcx2 = new_pzcx2

	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2.numpy(),
			"pz":pz.numpy(),"pzcx1":pzcx1.numpy(),"pzcx2":pzcx2.numpy(),
			"dual_z":dual_z.numpy(),"dual_x1":dual_x1.numpy(),"dual_x2":dual_x2.numpy()}
'''
'''
# logadmm for deterministic case
def detLogAdmm(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	#ss_init = kwargs["ss_init"]
	#ss_scale = kwargs["ss_scale"]
	ss_fixed = kwargs['ss_init']
	penalty = kwargs["penalty_coeff"]
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs["seed"]
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px2 = np.sum(px1x2,0)
	px1 = np.sum(px1x2,1)
	px1cx2 = px1x2 /px2[None,:]
	px2cx1 = (px1x2/ px1[:,None]).T

	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
		pz = kwargs['pz']
		pzcx1 = kwargs['pzcx1']
		pzcx2 = kwargs['pzcx2']
		dual_z = kwargs['dual_z']
		dual_x1 = kwargs['dual_x1']
		dual_x2 = kwargs['dual_x2']
	else:
		# random initialization
		pzcx1x2 = rng.random((nz,nx1,nx2),dtype="float64")
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

		# precomputed variables
		pz = np.sum(pzcx1x2 * px1x2[None,...],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,...],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,...],axis=1)

		# dual variables
		dual_z = np.zeros(pz.shape)
		dual_x1 = np.zeros(pzcx1.shape)
		dual_x2 = np.zeros(pzcx2.shape)
	mlog_pzcx1x2 = -np.log(pzcx1x2)
	mlog_pzcx1   = -np.log(pzcx1)
	mlog_pzcx2   = -np.log(pzcx2)
	mlog_pz      = -np.log(pz) 
	itcnt = 0
	conv_flag= False
	while itcnt < maxiter:
		itcnt +=1
		# precomputing error and exponentials
		exp_mlog_pz = np.exp(-mlog_pz)
		exp_mlog_pzcx1 = np.exp(-mlog_pzcx1)
		exp_mlog_pzcx2 = np.exp(-mlog_pzcx2)
		exp_mlog_pzcx1x2 = np.exp(-mlog_pzcx1x2)
		est_pzcx1 = np.sum(exp_mlog_pzcx1x2 * (px2cx1.T)[None,...],axis=2)
		est_pzcx2 = np.sum(exp_mlog_pzcx1x2 * px1cx2[None,...],axis=1)
		est_pz = np.sum(exp_mlog_pzcx1x2 * px1x2[None,...],axis=(1,2))

		# errors
		err_z = -np.log(est_pz)-mlog_pz
		err_x1 = -np.log(est_pzcx1)-mlog_pzcx1
		err_x2 = -np.log(est_pzcx2)-mlog_pzcx2

		# gradient of z|x1x2
		grad_p = -gamma*exp_mlog_pzcx1x2*(1-mlog_pzcx1x2)*px1x2[None,...] \
				+(exp_mlog_pzcx1x2 * (px2cx1.T)[None,...])*(( dual_x1 + penalty * err_x1)[...,None]/est_pzcx1[...,None]) \
				+(exp_mlog_pzcx1x2 * px1cx2[None,...])*np.repeat(np.expand_dims((dual_x2 + penalty* err_x2) /est_pzcx2,axis=1),repeats=nx1,axis=1)\
				+(exp_mlog_pzcx1x2*px1x2[None,...])*np.repeat(np.expand_dims(np.repeat(((dual_z + penalty* err_z) /est_pz)[:,None],repeats=nx1,axis=1),axis=2),repeats=nx2,axis=2)
		# projection, pzcx1x2
		raw_mlog_pzcx1x2 = mlog_pzcx1x2 - grad_p * ss_fixed
		raw_mlog_pzcx1x2 -= np.amin(raw_mlog_pzcx1x2,axis=0)
		raw_pzcx1x2 = np.exp(-raw_mlog_pzcx1x2) + 1e-9
		new_mlog_pzcx1x2 = -np.log(raw_pzcx1x2/np.sum(raw_pzcx1x2,axis=0,keepdims=True))

		# update temp vars
		exp_mlog_pzcx1x2 = np.exp(-new_mlog_pzcx1x2)
		est_pzcx1 = np.sum(exp_mlog_pzcx1x2 * (px2cx1.T)[None,...],axis=2)
		est_pzcx2 = np.sum(exp_mlog_pzcx1x2 * px1cx2[None,...],axis=1)
		est_pz = np.sum(exp_mlog_pzcx1x2 * px1x2[None,...],axis=(1,2))

		# update errors
		err_z = -np.log(est_pz)-mlog_pz
		err_x1 = -np.log(est_pzcx1)-mlog_pzcx1
		err_x2 = -np.log(est_pzcx2)-mlog_pzcx2

		# update dual variables
		dual_z += penalty * err_z
		dual_x1 += penalty * err_x1
		dual_x2 += penalty * err_x2

		# gradient of z
		grad_z = (1-gamma) * exp_mlog_pz*(1-mlog_pz) - (dual_z+penalty * err_z)
		# z projection
		raw_mlog_pz = mlog_pz - grad_z * ss_fixed
		raw_mlog_pz -= np.amin(raw_mlog_pz)
		raw_pz = np.exp(-raw_mlog_pz) + 1e-9
		new_pz = raw_pz/np.sum(raw_pz)
		new_mlog_pz = -np.log(new_pz)

		# gradient of x1
		grad_x1 = gamma * exp_mlog_pzcx1*(1-mlog_pzcx1) - (dual_x1 + penalty * err_x1)
		# x1 projection
		raw_mlog_pzcx1 = mlog_pzcx1 - grad_x1 * ss_fixed
		raw_mlog_pzcx1 -= np.amin(raw_mlog_pzcx1,axis=0)
		raw_pzcx1 = np.exp(-raw_mlog_pzcx1) + 1e-9
		new_pzcx1 = raw_pzcx1/np.sum(raw_pzcx1,axis=0,keepdims=True)
		new_mlog_pzcx1 = -np.log(new_pzcx1)

		# gradient of x2
		grad_x2 = gamma * exp_mlog_pzcx2*(1-mlog_pzcx2) - (dual_x2 + penalty*err_x2)
		# x2 projection
		raw_mlog_pzcx2 = mlog_pzcx2 = grad_x2 * ss_fixed
		raw_mlog_pzcx2 -= np.amin(raw_mlog_pzcx2,axis=0)
		raw_pzcx2 = np.exp(-raw_mlog_pzcx2) + 1e-9
		new_pzcx2 = raw_pzcx2 / np.sum(raw_pzcx2,axis=0,keepdims=True)
		new_mlog_pzcx2 = -np.log(new_pzcx2)

		# convergence # back to probability
		conv_z = 0.5 * np.sum(np.fabs(est_pz-new_pz))
		conv_x1 = 0.5 * np.sum(np.fabs(est_pzcx1-new_pzcx1),axis=0)
		conv_x2 = 0.5 * np.sum(np.fabs(est_pzcx2-new_pzcx2),axis=0)
		if np.all(conv_z < convthres) and np.all(conv_x1<convthres) and np.all(conv_x2 < convthres):
			conv_flag = True
			break
		else:
			mlog_pz = new_mlog_pz
			mlog_pzcx1 = new_mlog_pzcx1
			mlog_pzcx2 = new_mlog_pzcx2
			mlog_pzcx1x2 = new_mlog_pzcx1x2
	pzcx1x2 = np.exp(-mlog_pzcx1x2)
	pz = np.sum(pzcx1x2 * px1x2[None,...],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,...],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,...],axis=1)
	return {"conv":conv_flag,"niter":itcnt,
			"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,
			"dual_z":dual_z,"dual_x1":dual_x1,"dual_x2":dual_x2}
'''
# compared algorithms
# gradient based methods
# Sula, E.; Gastpar, M.C. Common Information Components Analysis. Entropy 2021, 23, 151.
# https://doi.org/10.3390/e23020151

# in its relaxed Wyner common information, the gradients are taken w.r.t. the (conditional) mutual information
# which can be equivalently expressed as derivative to a combination of entropy and conditional entropy functions.

def stoGradComp(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_init = kwargs['ss_init']
	ss_scale = kwargs['ss_scale']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,0)
	px2 = np.sum(px1x2,1)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)

	mask_pzcx1x2 = np.ones(pzcx1x2.shape)
	itcnt =0 
	conv_flag = False
	while itcnt < maxiter:
		itcnt += 1
		# auxiliary variables
		pz = np.sum(pzcx1x2*px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
		grad_p = ((1+gamma) * (np.log(pzcx1x2)+1)) * px1x2[None,:,:]\
					-(1-gamma) *( (np.log(pz) + 1)[:,None,None] * px1x2[None,:,:])\
					-gamma * (np.log(pzcx1) +1)[:,:,None] * px1x2[None,:,:] \
					-gamma * (np.log(np.repeat(np.expand_dims(pzcx2,axis=1),repeats=len(px1),axis=1))+1)*px1x2[None,:,:]
		mean_grad_p = grad_p - np.mean(grad_p,axis=0)
		ss_p = gd.naiveStepSize(pzcx1x2,-mean_grad_p * mask_pzcx1x2,ss_init,ss_scale)
		if ss_p == 0:
			break
		new_pzcx1x2 = pzcx1x2 - mean_grad_p * ss_p * mask_pzcx1x2
		# the compared method project the obtained encoder to the wyner setting
		new_pz = np.sum(new_pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		ent_z = -np.sum(new_pz*np.log(new_pz))
		ent_pzcx1x2 = -np.sum(new_pzcx1x2 * px1x2[None,:,:] * np.log(new_pzcx1x2))
		# the convergence criterion of the reference uses
		conv_z = 0.5 * np.sum(np.fabs(new_pzcx1x2 - pzcx1x2),axis=0) # total variation
		if np.all(conv_z<convthres):
			conv_flag=True
			break
		else:
			pzcx1x2 = new_pzcx1x2
	pz = np.sum(pzcx1x2*px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2}
'''
def stoLogGrad(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_fixed = kwargs['ss_init']
	d_seed = None
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,0)
	px2 = np.sum(px1x2,1)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
	mlog_pzcx1x2 = -np.log(pzcx1x2)
	itcnt =0
	conv_flag = False
	while itcnt < maxiter:
		itcnt +=1
		# auxiliary variables
		pzcx1x2 = np.exp(-mlog_pzcx1x2)
		pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
		pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
		pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)

		grad_p = -(1+gamma)*pzcx1x2 * px1x2[None,:,:] * (1-mlog_pzcx1x2) \
				 +gamma*pzcx1x2*px1x2[None,:,:]*(1+np.log(pzcx1)[:,:,None]) \
				 +gamma*pzcx1x2*px1x2[None,:,:]*(1+np.expand_dims(np.log(pzcx2),axis=1)) \
				 +(1-gamma)*pzcx1x2*px1x2[None,:,:]*(1+np.log(pz)[:,None,None])
		raw_mlog_pzcx1x2 = mlog_pzcx1x2 - ss_fixed * grad_p
		min_mlog_pzcx1x2 = np.amin(raw_mlog_pzcx1x2,axis=0)
		min_mlog_pzcx1x2 = np.where(min_mlog_pzcx1x2<0.0,min_mlog_pzcx1x2,np.zeros((nx1,nx2)))
		fixed_mlog_pzcx1x2 = raw_mlog_pzcx1x2 - min_mlog_pzcx1x2[None,:,:]
		raw_pzcx1x2 = np.exp(-fixed_mlog_pzcx1x2) + 1e-9
		new_pzcx1x2 = raw_pzcx1x2 / np.sum(raw_pzcx1x2,axis=0,keepdims=0)
		new_mlog_pzcx1x2 = -np.log(new_pzcx1x2)

		conv_z = np.sum(np.fabs(new_pzcx1x2 - pzcx1x2),axis=0)
		if np.all(conv_z<convthres):
			conv_flag = True
			break
		else:
			mlog_pzcx1x2 = new_mlog_pzcx1x2
	pzcx1x2 = np.exp(-mlog_pzcx1x2)
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,'pzcx1':pzcx1,"pzcx2":pzcx2}
'''
def stoLogDRS(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_fixed = kwargs['ss_init']
	d_seed = None
	penalty = kwargs['penalty_coeff']
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,0)
	px2 = np.sum(px1x2,1)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
		dual_p = kwargs['dual_p']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
	mlog_pzcx1x2 = -np.log(pzcx1x2)
	mlog_q = copy.deepcopy(mlog_pzcx1x2)
	dual_p = np.zeros((nz,nx1,nx2))

	itcnt =0
	conv_flag = False
	while itcnt < maxiter:
		itcnt +=1
		# p update
		# auxiliary variables for F
		aux_p = np.exp(-mlog_pzcx1x2)
		aux_pz = np.sum(aux_p * px1x2[None,:,:],axis=(1,2))
		#aux_pzcx1 = np.sum(aux_p * (px2cx1.T)[None,:,:],axis=2)
		#aux_pzcx2 = np.sum(aux_p * px1cx2[None,:,:],axis=1)
		err_p = mlog_pzcx1x2 - mlog_q

		grad_p = -(1+gamma) * aux_p * px1x2[None,:,:] * (1- mlog_pzcx1x2) \
				 +(1-gamma) * aux_p * px1x2[None,:,:] * (1+np.log(aux_pz)[:,None,None]) \
				 + dual_p + penalty * err_p
		raw_mlog_pzcx1x2 = mlog_pzcx1x2 - grad_p * ss_fixed
		# projection
		min_p = np.amin(raw_mlog_pzcx1x2,axis=0)
		min_p = np.where(min_p<0.0,min_p,np.zeros((nx1,nx2)))
		prj_mlog_pzcx1x2 = raw_mlog_pzcx1x2 - min_p
		raw_pzcx1x2 = np.exp(-prj_mlog_pzcx1x2) + 1e-9
		new_pzcx1x2 = raw_pzcx1x2 / np.sum(raw_pzcx1x2,axis=0,keepdims=True)
		new_mlog_pzcx1x2 = -np.log(new_pzcx1x2)

		# dual update
		err_p = new_mlog_pzcx1x2 - mlog_q
		dual_p += penalty * err_p

		# auxiliary variables for G
		q_prob = np.exp(-mlog_q)
		q_pzcx1 = np.sum(q_prob * (px2cx1.T)[None,:,:],axis=2)
		q_pzcx2 = np.sum(q_prob * px1cx2[None,:,:],axis=1)

		# FIXME: can be optimized for faster runtime, but keep as its separate gradients for clearness
		grad_q = gamma * q_prob * px1x2[None,:,:] * (1+np.log(q_pzcx1)[:,:,None]) \
				+gamma * q_prob * px1x2[None,:,:] * (1+np.expand_dims(np.log(q_pzcx2),axis=1)) \
				-dual_p - penalty * err_p

		raw_mlog_q = mlog_q - grad_q * ss_fixed
		min_q = np.amin(raw_mlog_q,axis=0)
		min_q = np.where(min_q<0,min_q,np.zeros((nx1,nx2)))
		prj_mlog_q = raw_mlog_q - min_q
		raw_q = np.exp(-prj_mlog_q) + 1e-9
		new_q = raw_q / np.sum(raw_q,axis=0,keepdims=True)
		new_mlog_q = -np.log(new_q)

		# convergence 
		#err_p = new_mlog_pzcx1x2 - new_mlog_q
		conv_p = np.sum(np.fabs(new_pzcx1x2-new_q),axis=0)
		if np.all(conv_p<convthres):
			conv_flag = True
			break
		else:
			mlog_pzcx1x2 = new_mlog_pzcx1x2
			mlog_q = new_mlog_q
	pzcx1x2 = np.exp(-mlog_pzcx1x2)
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,'pzcx1':pzcx1,"pzcx2":pzcx2,"dual_p":dual_p}

def stoLogDrsVar(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_fixed = kwargs['ss_init']
	d_seed = None
	penalty = kwargs['penalty_coeff']
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,0)
	px2 = np.sum(px1x2,1)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
		dual_p = kwargs['dual_p']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
	mlog_pzcx1x2 = -np.log(pzcx1x2)
	mlog_q = copy.deepcopy(mlog_pzcx1x2)
	dual_p = np.zeros((nz,nx1,nx2))

	itcnt =0
	conv_flag = False
	while itcnt < maxiter:
		itcnt +=1
		aux_p = np.exp(-mlog_pzcx1x2)
		err_p = mlog_pzcx1x2 - mlog_q
		#
		grad_p = -(1+gamma) * aux_p * px1x2[None,:,:] * (1-mlog_pzcx1x2) + dual_p + penalty * err_p
		raw_mlog_pzcx1x2 = mlog_pzcx1x2 - grad_p * ss_fixed
		min_p = np.amin(raw_mlog_pzcx1x2,axis=0)
		min_p = np.where(min_p<0.0,min_p,np.zeros((nx1,nx2)))
		prj_mlog_p = raw_mlog_pzcx1x2 - min_p
		raw_pzcx1x2 = np.exp(-prj_mlog_p) + 1e-9
		new_pzcx1x2 = raw_pzcx1x2 / np.sum(raw_pzcx1x2,axis=0,keepdims=True)
		new_mlog_pzcx1x2 = -np.log(new_pzcx1x2)
		# 
		err_p = new_mlog_pzcx1x2 - mlog_q
		#
		dual_p += penalty * err_p
		#
		aux_q = np.exp(-mlog_q)
		aux_qz = np.sum(aux_q * px1x2[None,:,:],axis=(1,2))
		aux_qzcx1 = np.sum(aux_q * (px2cx1.T)[None,:,:],axis=2)
		aux_qzcx2 = np.sum(aux_q * px1cx2[None,:,:],axis=1)
		
		grad_q = (1-gamma)*aux_q * px1x2[None,:,:] * (1+np.log(aux_qz)[:,None,None])\
				+gamma * aux_q * px1x2[None,:,:] * (1+np.log(aux_qzcx1)[:,:,None])\
				+gamma * aux_q * px1x2[None,:,:] * (1+np.expand_dims(np.log(aux_qzcx2),axis=1))\
				-dual_p - penalty * err_p
		raw_mlog_q = mlog_q - grad_q * ss_fixed
		min_q = np.amin(raw_mlog_q,axis=0)
		min_q = np.where(min_q<0.0,min_q,np.zeros((nx1,nx2)))
		prj_mlog_q = raw_mlog_q - min_q
		raw_q = np.exp(-prj_mlog_q) + 1e-9
		new_q = raw_q / np.sum(raw_q,axis=0,keepdims=True)
		new_mlog_q = -np.log(new_q)
		#
		err_p = new_mlog_pzcx1x2 - new_mlog_q
		dtv_p = 0.5 * np.sum(np.fabs(new_pzcx1x2-new_q),axis=0)
		if np.all(dtv_p<convthres):
			conv_flag = True
			break
		else:
			mlog_pzcx1x2 = new_mlog_pzcx1x2
			mlog_q = new_mlog_q
	pzcx1x2 = np.exp(-mlog_pzcx1x2)
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,'pzcx1':pzcx1,"pzcx2":pzcx2,"dual_p":dual_p}

def detLogDrs(px1x2,nz,gamma,maxiter,convthres,**kwargs):
	ss_fixed = kwargs['ss_init']
	d_seed = None
	penalty = kwargs['penalty_coeff']
	if kwargs.get("seed",False):
		d_seed = kwargs['seed']
	rng = np.random.default_rng(d_seed)
	(nx1,nx2) = px1x2.shape
	px1 = np.sum(px1x2,0)
	px2 = np.sum(px1x2,1)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = (px1x2/px1[:,None]).T
	if "init_load" in kwargs.keys():
		pzcx1x2 = kwargs['pzcx1x2']
		dual_p = kwargs['dual_p']
	else:
		pzcx1x2 = rng.random((nz,nx1,nx2))
		pzcx1x2 /= np.sum(pzcx1x2,axis=0,keepdims=True)
	mlog_pzcx1x2 = -np.log(pzcx1x2)
	mlog_q = copy.deepcopy(mlog_pzcx1x2)
	dual_p = np.zeros((nz,nx1,nx2))

	itcnt =0
	conv_flag = False
	while itcnt < maxiter:
		itcnt +=1
		# exact common information
		# loss = H(Z) + \gamma (I(Z;X1,X2) - I(Z;X1) - I(Z;X2) + I(X1;X2))
		# equivalent loss = (1-gamma)H(Z) -gamma H(Z|X1,X2) + gamma* H(Z|X1) + gamma* H(Z|X2)
		aux_p = np.exp(-mlog_pzcx1x2)
		err_p = mlog_pzcx1x2 - mlog_q
		# penalty <nu, mlogp - mlogq> + 0.5c * |mlogp - mlogq|^2
		grad_p = -gamma * aux_p * (1-mlog_pzcx1x2) * px1x2[None,:,:] + dual_p + err_p * penalty
		raw_mlog_pzcx1x2 = mlog_pzcx1x2 - ss_fixed * grad_p
		# projection
		min_p = np.amin(raw_mlog_pzcx1x2,axis=0)
		min_p = np.where(min_p<0.0,min_p,np.zeros((nx1,nx2)))
		prj_mlog_p = raw_mlog_pzcx1x2 - min_p
		raw_pzcx1x2 = np.exp(-prj_mlog_p) + 1e-9
		new_p = raw_pzcx1x2 / np.sum(raw_pzcx1x2,axis=0,keepdims=True)
		new_mlog_pzcx1x2 = -np.log(new_p)

		err_p = new_mlog_pzcx1x2 - mlog_q
		# dual ascend
		dual_p += penalty * err_p
		# q update
		aux_q = np.exp(-mlog_q)
		aux_qz = np.sum(aux_q * px1x2[None,:,:],axis=(1,2))
		aux_pzcx1 = np.sum(aux_q* (px2cx1.T)[None,:,:],axis=2)
		aux_pzcx2 = np.sum(aux_q* px1cx2[None,:,:],axis=1)
		grad_q = (1-gamma)* aux_q* px1x2[None,:,:] * (1+np.log(aux_qz)[:,None,None]) \
				 + gamma  * aux_q* px1x2[None,:,:] * (1+np.log(aux_pzcx1)[:,:,None]) \
				 + gamma  * aux_q* px1x2[None,:,:] * (1+np.expand_dims(np.log(aux_pzcx2),axis=1))\
				 - dual_p - penalty * err_p
		# projection for q
		raw_mlog_q = mlog_q - ss_fixed * grad_q
		min_q = np.amin(raw_mlog_q,axis=0)
		min_q = np.where(min_q<0.0,min_q,np.zeros((nx1,nx2)))
		prj_mlog_q = raw_mlog_q - min_q
		raw_q = np.exp(-prj_mlog_q) + 1e-9
		new_q = raw_q/np.sum(raw_q,axis=0,keepdims=True)
		new_mlog_q = -np.log(new_q)

		# convergence
		#err_p = new_mlog_pzcx1x2 - new_mlog_q
		# total variation
		dtv_p = 0.5 * np.sum(np.fabs(new_p-new_q),axis=0)
		#conv_p = np.sum(np.fabs(err_p),axis=0)
		if np.all(dtv_p<convthres):
			conv_flag = True
			break
		else:
			mlog_pzcx1x2 = new_mlog_pzcx1x2
			mlog_q = new_mlog_q
	pzcx1x2 = np.exp(-mlog_pzcx1x2)
	pz = np.sum(pzcx1x2 * px1x2[None,:,:],axis=(1,2))
	pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
	pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
	return {"conv":conv_flag,"niter":itcnt,"pzcx1x2":pzcx1x2,"pz":pz,"pzcx1":pzcx1,"pzcx2":pzcx2,"dual_p":dual_p,}
