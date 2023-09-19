import numpy as np
import os
import sys
import argparse
import pickle
import utils as ut
import algorithm as alg
import dataset as dt
import evaluation as ev
import copy
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("method",choices=["admmgd","logdrs"])
parser.add_argument("--penalty",type=float,default=256.0,help="penalty coefficient of the ADMM solver")
parser.add_argument("--maxiter",type=int,default=100000,help="maximum iteration before termination")
parser.add_argument("--convthres",type=float,default=1e-6,help="convergence threshold")
parser.add_argument("--nrun",type=int,default=10,help="number of trail of each simulation")
parser.add_argument("--ss_init",type=float,default=4e-3,help="step size initialization")
parser.add_argument("--ss_scale",type=float,default=0.25,help="step size scaling")
parser.add_argument("--output_dir",type=str,default="unsu_det_test_results",help="output filename")
parser.add_argument("--seed",type=int,default=None,help="random seed for reproduction")
parser.add_argument("--ny",type=int,default=2,help="number of uniform hidden labels")
parser.add_argument("--nb",type=int,default=2,help="number of blocks for observations")
parser.add_argument("--corr",type=float,default=0,help="cyclic observation uncertainty given a label")
parser.add_argument("--gamma_min",type=float,default=1.0,help="minimum gamma value")
parser.add_argument("--gamma_max",type=float,default=10.0,help="maximum gamma value")
# the maximum value is always 1. otherwise a different problem
parser.add_argument("--gamma_num",type=int,default=10,help="number of gamma values")
parser.add_argument("--test",action="store_true",default=True,help="enabling testing for synthetic clustering")
parser.add_argument("--zsearch",action="store_true",default=False,help="no knowledge of the number of clusters")
#parser.add_argument("--encoder",action="store_true",default=False,help="storing all the encoders found")

args = parser.parse_args()
argsdict = vars(args)

print(argsdict)

data_dict = dt.synExpandToy(args.ny,args.nb,args.corr)
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

gamma_range = np.geomspace(args.gamma_min,args.gamma_max,num=args.gamma_num)

alg_dict = {
"penalty_coeff":args.penalty,
"ss_init":args.ss_init,
"ss_scale":args.ss_scale,
"seed":args.seed,
}

# algorithm selection
if args.method == "admmgd":
	algrun = alg.detComAdmm
#elif args.method == "tfadmmgd":
#	algrun = alg.tfDetComAdmm
#elif args.method == "logadmm":
#	algrun = alg.detLogAdmm
elif args.method == "logdrs":
	algrun = alg.detLogDrs
else:
	sys.exit("{:} is an undefined method".format(args.method))

encoder_list = []
if args.zsearch:
	nz_set = np.arange(2,len(px1)*len(px2)+1,1)
else:
	nz_set = np.array([args.ny]) # FIXME: assume knowing the cardinality of Z
res_all = np.zeros((len(gamma_range)*args.nrun*len(nz_set),11)) # gamma,nn, nidx, niter, conv,nz, entz, mizx1,mizx2,joint_mi, loss, cmix1x2cz
rec_idx = 0
for gidx ,gamma in enumerate(gamma_range):
	for nz in nz_set:
		for nn in range(args.nrun):
			out_dict = algrun(prob_joint,nz,gamma,args.maxiter,args.convthres,**alg_dict)
			tmp_result = [gamma,nn,out_dict['niter'],int(out_dict["conv"]),nz]
			# convergence reached... do things afterward
			# calculate the mutual informations
			# compute the joint encoder
			pzcx1x2 = out_dict['pzcx1x2'] # this might not be a valid prob, but is forced to be valid one
			pz = np.sum(pzcx1x2 * prob_joint[None,:,:],axis=(1,2))
			pzcx1 = np.sum(pzcx1x2 * (px2cx1.T)[None,:,:],axis=2)
			pzcx2 = np.sum(pzcx1x2 * px1cx2[None,:,:],axis=1)
			if args.test:
				encoder_list.append(pzcx1x2)
			# take the maximum element
			pzx1x2 = pzcx1x2 * prob_joint[None,:,:]

			#cmix1x2cz = ut.calcMIcond(np.transpose(pzx1x2,axes=[1,2,0]))
			entzcx1x2 = np.sum(-pzx1x2 * np.log(pzcx1x2))
			entz = ut.calcEnt(pz)
			mizx1 = ut.calcMI(pzcx1 * px1[None,:])
			mizx2 = ut.calcMI(pzcx2 * px2[None,:])
			joint_mi = entz - entzcx1x2
			cmix1x2cz = joint_mi - (mizx1 + mizx2) + mix1x2
			# loss calculation
			
			tmp_loss = entz +gamma * cmix1x2cz
			tmp_result += [entz,mizx1,mizx2,joint_mi,tmp_loss,cmix1x2cz]
				
			res_all[rec_idx,:] = np.array(tmp_result)
			print("gamma,{:.3f},nidx,{:},nz,{:},conv,{:},nit,{:},H(Z),{:.5f},loss,{:.4f},I(X1;X2|Z),{:.5f}".format(gamma,nn,nz,int(out_dict["conv"]),out_dict["niter"],entz,tmp_loss,cmix1x2cz))
			rec_idx += 1

d_save_dir = os.path.join(os.getcwd(),args.output_dir)
os.makedirs(d_save_dir,exist_ok=True)
tnow = datetime.datetime.now()
repeat_cnt = 0
safe_savename_base = "det_{:}_y{:}b{:}_cr{:.4f}_nzall_{:}".format(args.method,args.ny,args.nb,args.corr,tnow.strftime("%Y%m%d"))
safe_savename = copy.copy(safe_savename_base)
while os.path.isfile(os.path.join(d_save_dir,safe_savename)):
	repeat_cnt+=1
	safe_savename = "{:}_{:}".format(safe_savename_base,repeat_cnt)

# saving the results, numpy array
with open(os.path.join(d_save_dir,safe_savename+".npy"),"wb") as fid:
	np.save(fid,res_all)
# saving the encoders
if args.encoder:
	with open(os.path.join(d_save_dir,safe_savename+"_encoders.pkl"),"wb") as fid:
		pickle.dump(encoder_list,fid)
# saving the configuration in case of error
with open(os.path.join(d_save_dir,safe_savename+"_config.pkl"),"wb") as fid:
	pickle.dump(argsdict,fid)