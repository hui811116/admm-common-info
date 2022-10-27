import numpy as np
import os
import sys
import copy

def getDataset(stext):
	if stext == "syn2v":
		return syn2views()
	elif stext == "condindp2v":
		return syn2CondIndp()
	elif stext == "overlap2v":
		return syn2Overlap()
	elif stext == "toy2v":
		return synToy()
	else:
		sys.exit("ERROR: {:} does not match any defined dataset".format(stext))

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

	return {"p_joint":px1x2,"px_list":[px1,px2],"p_cond":[px1cx2,px2cx1],"pycx_list":[px1cy,px2cy],"py":py}