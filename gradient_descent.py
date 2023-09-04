import numpy as np
import sys
import os
#import tensorflow as tf

def naiveStepSize(prob,update,ss_init,ss_scale):
	ssout = ss_init
	while np.any(prob + ssout*update <= 0.0) or np.any(prob+ssout*update>=1.0):
		ssout *= ss_scale
		if ssout < 1e-12:
			return 0
	return ssout

'''
def tfNaiveSS(tfprob,update,init_step,scale):
	stepsize = init_step
	while tf.reduce_any(tfprob+update * stepsize<=0.0 ) or tf.reduce_any(tfprob+update*stepsize>=1.0):
		stepsize*= scale
		if stepsize<1e-11:
			stepsize = 0
			break
	return stepsize
'''
# project to log probability so the range is R-
def logProb2DProj(log_p,grad_p,ss_step):
	epsilon = 1e-9 # smoothness coefficient 
	raw_log_p = log_p - ss_step * grad_p
	lmax = np.amax(raw_log_p,axis=0)
	lmax = np.where(lmax>0.0,lmax,np.zeros((log_p.shape[-1])))
	raw_log_p -= lmax[None,:]
	raw_p = np.exp(raw_log_p) + epsilon
	new_p = raw_p / np.sum(raw_p,axis=0,keepdims=True)
	return np.log(new_p)

def negLogProb2DProj(mlog_p,grad_p,ss_step):
	epsilon = 1e-9
	raw_mlog_p = mlog_p - ss_step * grad_p
	lmin = np.amin(raw_mlog_p,axis=0)
	lmin = np.where(lmin<0.0,lmin,np.zeros((log_p.shape[-1])))
	raw_mlog_p -= lmin[None,:]
	raw_p = np.exp(-raw_mlog_p) + epsilon
	new_p = raw_p / np.sum(raw_p,axis=0,keepdims=True)
	return -np.log(new_p)