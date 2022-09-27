import numpy as np
import sys
import os
import tensorflow as tf

def naiveStepSize(prob,update,ss_init,ss_scale):
	ssout = ss_init
	while np.any(prob + ssout*update <= 0.0) or np.any(prob+ssout*update>=1.0):
		ssout *= ss_scale
		if ssout < 1e-12:
			return 0
	return ssout

def tfNaiveSS(tfprob,update,init_step,scale):
	stepsize = init_step
	while tf.reduce_any(tfprob+update * stepsize<=0.0 ) or tf.reduce_any(tfprob+update*stepsize>=1.0):
		stepsize*= scale
		if stepsize<1e-11:
			stepsize = 0
			break
	return stepsize