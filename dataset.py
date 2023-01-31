import numpy as np
import os
import sys
import copy
import pickle

'''
def getDataset(stext,**kwargs):
	if stext == "syn2v":
		return syn2views()
	elif stext == "condindp2v":
		return syn2CondIndp()
	elif stext == "overlap2v":
		return syn2Overlap()
	elif stext == "toy2v":
		return synToy()
	elif stext == "toynv":
		return synToyNview(kwargs)
	else:
		sys.exit("ERROR: {:} does not match any defined dataset".format(stext))
'''
def syn2views():
	# data
	# Statistics
	# ----------------
	# mix1x2=0.6616
	# Hx1=1.0778
	# Hx2=1.0805
	# Hx1x2=1.4968
	# ----------------
	px1cx2 = np.array([
		[0.85, 0.04, 0.06],
		[0.07, 0.88, 0.02],
		[0.08, 0.08, 0.92]])
	px2 = np.array([0.25, 0.35,0.40])
	px12 = px1cx2 * px2[None,:]
	px1 = np.sum(px12,1)
	px2cx1 = (px12/px1[:,None]).T
	return {"p_joint":px12,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1]}


def syn2CondIndp():
	# data
	# Statistics
	# ----------------
	#
	# ----------------
	py = np.array([0.25,0.40,0.35])
	px1cy = np.array([
		[0.85, 0.04 , 0.06],
		[0.07, 0.40 , 0.60],
		[0.08, 0.56 , 0.34]])
	px1 = np.sum(px1cy* py[None,:],1)
	px2cy = np.array([
		[0.30, 0.08, 0.35],
		[0.40, 0.80, 0.15],
		[0.30, 0.12, 0.50]])
	px2 = np.sum(px2cy * py[None,:],1)
	px12 = np.zeros((px1cy.shape[0],px2cy.shape[0]))
	for ix1 in range(3):
		for ix2 in range(3):
			tmp_sum = 0
			for iy in range(3):
				tmp_sum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
			px12[ix1,ix2]= tmp_sum
	px1cx2 = px12 / px2[None,:]
	px2cx1 = (px12 / px1[:,None]).T
	return {"p_joint":px12,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}

def syn2Overlap():
	# data
	# Statistics
	# ----------------
	#
	# ----------------
	py = np.array([0.25,0.40,0.35])
	px1cy = np.array([
		[0.85, 0.04 , 0.06],
		[0.07, 0.86 , 0.04],
		[0.08, 0.10 , 0.90]])
	px1 = np.sum(px1cy* py[None,:],1)
	px2cy = np.array([
		[0.06, 0.84, 0.05],
		[0.06, 0.08, 0.90],
		[0.88, 0.08, 0.05]])
	px2 = np.sum(px2cy * py[None,:],1)
	px12 = np.zeros((px1cy.shape[0],px2cy.shape[0]))
	for ix1 in range(3):
		for ix2 in range(3):
			tmp_sum = 0
			for iy in range(3):
				tmp_sum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
			px12[ix1,ix2]= tmp_sum
	px1cx2 = px12 / px2[None,:]
	px2cx1 = (px12 / px1[:,None]).T
	return {"p_joint":px12,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}

def synToy():
	eps_min = 1e-2 # for smoothness
	# data
	# Statistics
	# ------------
	# 
	# ------------
	py = np.array([0.5,0.5])
	px1cy = np.array([
		[eps_min/2,0.5-eps_min/2],
		[0.5 - eps_min/2,eps_min/2],
		[0.5 - eps_min/2,eps_min/2],
		[eps_min/2,0.5-eps_min/2]])
	px2cy = np.array([
		[eps_min/2,0.5-eps_min/2],
		[0.5 - eps_min/2,eps_min/2],
		[0.5 - eps_min/2,eps_min/2],
		[eps_min/2,0.5-eps_min/2]])
	px1 = np.sum(px1cy * py[None,:],1)
	px2 = np.sum(px2cy * py[None,:],1)
	px12 = np.zeros((px1cy.shape[0],px2cy.shape[0]))
	for ix1 in range(4):
		for ix2 in range(4):
			tmp_sum = 0
			for iy in range(2):
				tmp_sum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
			px12[ix1,ix2] = tmp_sum
	px12 /= np.sum(px12)
	px1cx2 = px12 / px2[None,:]
	px2cx1 = (px12/px1[:,None]).T
	return {"p_joint":px12,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}

#TODO: nview dataset
#def synToyNview(ny):
#	return 
'''
def synExpandToy(ny,nx,corr):
	eps_min = 1e-7
	py = np.ones((ny,)) * (1/ny)
	first_vec_ycx = np.ones((nx,))*(1/nx) - corr/nx
	second_vec_ycx = np.ones((nx,)) * corr/nx
	block_ele = [first_vec_ycx,second_vec_ycx]
	#print(block_ele)
	for idx in range(0,ny-2):
		block_ele.append(np.zeros((nx,)))
	px1cy = np.zeros((nx*ny,ny))
	for cy in range(ny):
		for cb,cele in enumerate(block_ele):
			start_idx = ((cy + cb)%ny)*nx
			px1cy[start_idx:start_idx+nx,cy] = cele
	px2cy = copy.deepcopy(px1cy)
	px1x2 = np.zeros((px1cy.shape[0],px2cy.shape[0]))
	#print(px1cy)
	for ix1 in range(px1cy.shape[0]):
		for ix2 in range(px2cy.shape[0]):
			tmp_sum = 0
			for iy in range(ny):
				tmp_sum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
			px1x2[ix1,ix2] = tmp_sum
	# smoothness
	px1x2 += eps_min
	px1x2 /= np.sum(px1x2)
	px1 = np.sum(px1x2,axis=1)
	px2 = np.sum(px1x2,axis=0)
	px1cx2 = px1x2 / px2[None,:]
	px2cx1 = (px1x2 / px1[:,None]).T
	for ix2 in range(len(px2)):
		tmp_list = []
		for ix1 in range(len(px1)):
			tmp_list.append("{:.3f}".format(px1x2[ix1,ix2]))
		print(",".join(tmp_list))
	return {"p_joint":px1x2,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}
'''
def synExpandToy(ny,nb,corr):
	raw_eps = 1e-9
	# true cyclic correlated noise
	nx = ny * nb
	py = np.ones((ny,))/ny
	one_pxcy = np.zeros((nx,))
	one_pxcy[:nb] = (1-corr)/nb
	one_pxcy[nb:2*nb] = (corr)/nb
	px1cy = np.zeros((nx,ny))
	for iy in range(ny):
		pos_offset = nb*iy
		for ix in range(nb*ny):
			px1cy[ix,iy] = one_pxcy[-pos_offset + ix]
	#print(px1cy)
	#sys.exit()
	px2cy = copy.deepcopy(px1cy)
	px1x2 = np.zeros((nx,nx))
	for ix1 in range(nx):
		for ix2 in range(nx):
			tmpsum = 0
			for iy in range(ny):
				tmpsum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
			px1x2[ix1,ix2] = tmpsum
	# smoothing
	px1x2 += raw_eps
	px1x2 /= np.sum(px1x2,keepdims=True)
	'''
	for ix1 in range(nx):
		tmplist =[]
		for ix2 in range(nx):
			tmplist.append("{:.2f}".format(px1x2[ix1,ix2]))
		print(",".join(tmplist))
	'''
	#sys.exit()
	px1 = np.sum(px1x2,axis=1)
	px2 = np.sum(px1x2,axis=0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = px1x2.T/px1[None,:]
	return {"p_joint":px1x2,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}

def synExpandToyNonUnif(ny,nb,corr):
	raw_eps = 1e-9
	# true cyclic correlated noise
	nx = ny * nb
	#py = np.ones((ny,))/ny
	py = np.arange(1,ny+1,1)
	py /= np.sum(py,keepdims=True)
	one_pxcy = np.zeros((nx,))
	one_pxcy[:nb] = (1-corr)/nb
	one_pxcy[nb:2*nb] = (corr)/nb
	px1cy = np.zeros((nx,ny))
	for iy in range(ny):
		pos_offset = nb*iy
		for ix in range(nb*ny):
			px1cy[ix,iy] = one_pxcy[-pos_offset + ix]
	#print(px1cy)
	#sys.exit()
	px2cy = copy.deepcopy(px1cy)
	px1x2 = np.zeros((nx,nx))
	for ix1 in range(nx):
		for ix2 in range(nx):
			tmpsum = 0
			for iy in range(ny):
				tmpsum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
			px1x2[ix1,ix2] = tmpsum
	# smoothing
	px1x2 += raw_eps
	px1x2 /= np.sum(px1x2,keepdims=True)
	'''
	for ix1 in range(nx):
		tmplist =[]
		for ix2 in range(nx):
			tmplist.append("{:.2f}".format(px1x2[ix1,ix2]))
		print(",".join(tmplist))
	'''
	#sys.exit()
	px1 = np.sum(px1x2,axis=1)
	px2 = np.sum(px1x2,axis=0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = px1x2.T/px1[None,:]
	return {"p_joint":px1x2,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}

def synCorrUnif(ny,nb,corr):
	nx = ny * nb
	py = np.ones((ny,))/ny
	one_pxcy = np.ones((nx,)) * (corr/(nx-nb))
	one_pxcy[:nb] = (1-corr)/nb
	#print(one_pxcy)
	#print(np.sum(one_pxcy))
	px1cy = np.zeros((nx,ny))
	for iy in range(ny):
		pos_offset = nb * iy
		for ix in range(nb*ny):
			#print(one_pxcy[-pos_offset + ix])
			px1cy[ix,iy] = one_pxcy[-pos_offset + ix]
	#print(px1cy)
	#sys.exit()
	px2cy = copy.deepcopy(px1cy)
	px1x2 = np.zeros((nx,nx))
	for ix1 in range(nx):
		for ix2 in range(nx):
			tmpsum =0
			for iy in range(ny):
				tmpsum += py[iy] * px1cy[ix1,iy] * px2cy[ix2,iy]
			px1x2[ix1,ix2] = tmpsum
	px1x2 /= np.sum(px1x2,keepdims=True)
	px1 = np.sum(px1x2,axis=1)
	px2 = np.sum(px1x2,axis=0)
	px1cx2 = px1x2/px2[None,:]
	px2cx1 = px1x2.T/px1[None,:]
	return {"p_joint":px1x2,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}

def loadSampleData(dataset_path):
	with open(dataset_path,'rb') as fid:
		dataset_dict = pickle.load(fid)
	train_dict = dataset_dict['train_dict'] # ylabel, xsample, ny, nx1, nx2
	test_dict = dataset_dict['test_dict']
	# counting training set to have an estimate of px1x2,
	# send labels back too
	# the testing set simply copy what was loaded
	cnt_x1x2 = np.zeros((train_dict['nx1'],train_dict['nx2']))
	for idx in range(train_dict['xsample'].shape[0]):
		x1 = train_dict['xsample'][idx][0] # 1st view
		x2 = train_dict['xsample'][idx][1] # 2nd view
		cnt_x1x2[x1,x2] += 1
	est_px1x2 = cnt_x1x2 / np.sum(cnt_x1x2,keepdims=True)
	return {
		'ytrain':train_dict['ylabel'],'xtrain':train_dict['xsample'],
		"ytest":test_dict['ylabel'],"xtest":test_dict['xsample'],
		"ny":train_dict['ny'],'nx1':train_dict['nx1'],"nx2":train_dict['nx2'],
		"p_joint":est_px1x2,
	}