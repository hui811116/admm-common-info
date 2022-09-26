import numpy as np
import os
import sys
import argparse
import pickle
import utils as ut
import algorithm as alg
import dataset as dt
import evaluation as ev
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("method",choices=["admmgd","tfadmmgd"])
parser.add_argument("--penalty",type=float,default=64.0,help="penalty coefficient of the ADMM solver")
parser.add_argument("--maxiter",type=int,default=40000,help="maximum iteration before termination")
parser.add_argument("--convthres",type=float,default=1e-6,help="convergence threshold")
parser.add_argument("--nrun",type=int,default=10,help="number of trail of each simulation")
parser.add_argument("--ss_init",type=float,default=2e-3,help="step size initialization")
parser.add_argument("--ss_scale",type=float,default=0.25,help="step size scaling")
parser.add_argument("--output",type=str,default="unsu_mv_output",help="output filename")
parser.add_argument("--seed",type=int,default=None,help="random seed for reproduction")
parser.add_argument("--ny",type=int,default=2,help="number of uniform hidden labels")
parser.add_argument("--nb",type=int,default=2,help="number of blocks for observations")
parser.add_argument("--corr",type=float,default=0,help="cyclic observation uncertainty given a label")
parser.add_argument("--gamma_min",type=float,default=0.1,help="minimum gamma value")
parser.add_argument("--gamma_max",type=float,default=2.0,help="maximum gamma value")
# the maximum value is always 1. otherwise a different problem
parser.add_argument("--gamma_num",type=int,default=10,help="number of gamma values")
parser.add_argument("--encoder",action="store_true",default=False,help="storing all the encoders found")

args = parser.parse_args()
argsdict = vars(args)

print(argsdict)

data_dict = dt.synExpandToy(args.ny,args.nb,args.corr)
#sys.exit()
prob_joint = data_dict["p_joint"]
px1 = np.sum(prob_joint,1)
px2 = np.sum(prob_joint,0)
mix1x2 = ut.calcMI(prob_joint)
print("mix1x2={:.4f}".format(mix1x2))
print("Hx1={:.4f}".format(ut.calcEnt(px1)))
print("Hx2={:.4f}".format(ut.calcEnt(px2)))
print("Hx1x2={:.4f}".format(ut.calcEnt(prob_joint)))
prob_cond = data_dict["p_cond"]
px1cx2 = prob_cond[0]
px2cx1 = prob_cond[1]
# TODO: what if |Z| neq |Y|
nz = len(data_dict["py"]) # FIXME: for "condindp" dataset, |Y|=3

#gamma_range = np.geomspace(1,args.gamma_max,num=args.gamma_num)
gamma_range = np.geomspace(args.gamma_min,args.gamma_max,num=args.gamma_num)

alg_dict = {
"penalty_coeff":args.penalty,
"ss_init":args.ss_init,
"ss_scale":args.ss_scale,
"seed":args.seed,
}

# algorithm selection
if args.method == "admmgd":
	algrun = alg.admmHighDim
elif args.method == "tfadmmgd":
	algrun = alg.tfStoComAdmm
else:
	sys.exit("undefined method {:}".format(args.method))

encoder_dict = {}

nz_set = [nz]
res_all = np.zeros((len(gamma_range)*args.nrun*len(nz_set),12)) # gamma, nidx, niter, conv,nz, entz, mizx1,mizx2,cmizx1cx2, cmizx2cx1, max_ele, loss
rec_idx = 0
for gidx ,gamma in enumerate(gamma_range):
	encoder_dict[gidx] = {}
	for nz in nz_set:
		min_loss = np.Inf
		for nn in range(args.nrun):
			#out_dict = alg.admmHighDim(prob_joint,nz,gamma,args.maxiter,args.convthres,**alg_dict)
			out_dict = algrun(prob_joint,nz,gamma,args.maxiter,args.convthres,**alg_dict)
			tmp_result = [gamma,nn,out_dict['niter'],int(out_dict["conv"]),nz]
			if out_dict['conv']:
				
				# convergence reached... do things afterward
				# calculate the mutual informations
				pz = out_dict["pz"]
				entz = ut.calcEnt(pz)
				pzcx1 = out_dict["pzcx1"]
				pzcx2 = out_dict["pzcx2"]
				# compute the joint encoder
				pzcx1x2 = out_dict['pzcx1x2'] # this might not be a valid prob, but is forced to be valid one
				if args.encoder:
					encoder_dict[gidx][nn] = pzcx1x2
				# take the maximum element
				max_ele = np.amax(pzcx1x2)
				mizx1 = ut.calcMI(pzcx1 * px1[None,:])
				mizx2 = ut.calcMI(pzcx2 * px2[None,:])
				cmizx1cx2 = ut.calcMIcond(pzcx1x2 * prob_joint[None,:,:])
				# calculate other direction
				cmizx2cx1 = ut.calcMIcond(np.transpose(pzcx1x2 * prob_joint[None,:,:],(0,2,1)))
				# loss calculation
				joint_mi = ut.calcMI(np.reshape(pzcx1x2 * prob_joint[None,:,:],(nz,np.prod(prob_joint.shape))))
				tmp_loss = (1+gamma)*joint_mi - gamma * mizx1 - gamma * mizx2
				tmp_result += [entz,mizx1,mizx2,cmizx1cx2,cmizx2cx1,max_ele,tmp_loss]
				if tmp_loss < min_loss:
					min_loss = tmp_loss
			else:
				# failed in convergence, either maximum iteration reached or no available step size
				max_ele = -1
				tmp_result += [0,0,0,0,0,max_ele,np.Inf]
				tmp_loss = np.Inf
				
			res_all[rec_idx,:] = np.array(tmp_result)
			print("gamma:{:.4f}, ntrial:{:}, nz:{:}, convergence:{:}, niter:{:}, max_element:{:.6f}, tmp_loss:{:.5f}, min_loss:{:.5f}".format(gamma,nn,nz,out_dict["conv"],out_dict["niter"],max_ele,tmp_loss,min_loss))
			rec_idx += 1

		#print("simulated gamma:{:.4f}, nz:{:}, conv_rate:{:.4f}, max_element:{:.4f}, min_loss:{:.5f}".format(gamma,nz,conv_cnt/args.nrun,avg_test,min_loss))

timenow= datetime.datetime.now()
ymd_tex = "{:04}{:02}{:02}".format(timenow.year,timenow.month,timenow.day)
hm_tex  = "h{:02}m{:02}".format(timenow.hour,timenow.minute)
md_dir_name = "{:}_{:}_{:}".format(args.output,ymd_tex,hm_tex)
# saving the results, numpy array
with open(args.output+".npy","wb") as fid:
	np.save(fid,res_all)
# saving the encoders
if args.encoder:
	with open(args.output+"_encoders.pkl","wb") as fid:
		pickle.dump(encoder_dict,fid)
# saving the configuration in case of error
with open(args.output+"_config.pkl","wb") as fid:
	pickle.dump(argsdict,fid)