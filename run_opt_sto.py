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
import copy

parser = argparse.ArgumentParser()
parser.add_argument("method",choices=["admmgd",'gddrs',"gdbaseline","loggd",'logdrs','logdrsvar'])
parser.add_argument("--penalty",type=float,default=128.0,help="penalty coefficient of the ADMM solver")
parser.add_argument("--maxiter",type=int,default=300000,help="maximum iteration before termination")
parser.add_argument("--convthres",type=float,default=1e-6,help="convergence threshold")
parser.add_argument("--nrun",type=int,default=5,help="number of trail of each simulation")
parser.add_argument("--ss_init",type=float,default=1e-2,help="step size initialization")
parser.add_argument("--ss_scale",type=float,default=0.25,help="step size scaling")
parser.add_argument("--output_dir",type=str,default="unsu_sto_test_results",help="output filename")
parser.add_argument("--seed",type=int,default=None,help="random seed for reproduction")
parser.add_argument("--ny",type=int,default=8,help="number of uniform hidden labels")
parser.add_argument("--nb",type=int,default=2,help="number of blocks for observations")
parser.add_argument("--corr",type=float,default=0,help="cyclic observation uncertainty given a label")
parser.add_argument("--gamma_min",type=float,default=1.0,help="minimum gamma value")
parser.add_argument("--gamma_max",type=float,default=50.0,help="maximum gamma value")
# the maximum value is always 1. otherwise a different problem
parser.add_argument("--gamma_num",type=int,default=20,help="number of gamma values")
#parser.add_argument("--encoder",action="store_true",default=False,help="storing all the encoders found")
#parser.add_argument("--test",action="store_true",default=True,help="enabling testing for synthetic clustering")
parser.add_argument("--zsearch",action="store_true",default=False,help="no knowledge of the number of clusters")

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
	algrun = alg.admmHighDim
elif args.method == "gddrs":
	algrun = alg.stoGdDrs
#elif args.method == "tfadmmgd":
#	algrun = alg.tfStoComAdmm
#elif args.method == "logadmm":
#	algrun = alg.stoLogAdmm
elif args.method == "gdbaseline":
	algrun = alg.stoGradComp
#elif args.method == "loggd":
#	algrun = alg.stoLogGrad
elif args.method == "logdrs":
	algrun = alg.stoLogDRS
elif args.method == "logdrsvar":
	algrun = alg.stoLogDrsVar
else:
	sys.exit("undefined method {:}".format(args.method))

# loading testing data # FIXME: manually loaded
# FIXME: only holds for ny=8, nb=2, corr=0, or corr=0.2
#with open("test_opt_y8b2_cr0.0000e+00_obs.npy",'rb') as fid:
#	test_obs_y8b2_c00 = np.load(fid)
#with open("test_opt_y8b2_cr0.0000e+00_label.npy",'rb') as fid:
#	test_label_y8b2_c00 = np.load(fid)

# saving the encoders 
#encoder_dict = {}
encoder_list =[]

if args.zsearch:
	nz_set = np.arange(2,len(px1)*len(px2)+1,1)
else:
	nz_set = np.array([args.ny]) # FIXME: assume knowing the cardinality of Z
res_all = np.zeros((len(gamma_range)*args.nrun*len(nz_set),11)) # gamma, nidx, niter, conv,nz, entz, mizx1,mizx2,joint_MI, loss, cmix1x2cz
#res_all = np.zeros((len(gamma_range)*args.nrun*len(nz_set),12)) # gamma, nidx, niter, conv,nz, entz, mizx1,mizx2,joint_MI, loss, cmix1x2cz, best_acc
rec_idx = 0
for gidx ,gamma in enumerate(gamma_range):
	#encoder_dict[gidx] = {}
	for nz in nz_set:
		for nn in range(args.nrun):
			out_dict = algrun(prob_joint,nz,gamma,args.maxiter,args.convthres,**alg_dict)
			tmp_result = [gamma,nn,out_dict['niter'],int(out_dict["conv"]),nz]
				
			# convergence reached... do things afterward
			# calculate the mutual informations
			pz = out_dict["pz"]
			entz = ut.calcEnt(pz)
			# calculate ent_z|x1x2
			pzcx1 = out_dict["pzcx1"]
			pzcx2 = out_dict["pzcx2"]
			# compute the joint encoder
			pzcx1x2 = out_dict['pzcx1x2'] # this might not be a valid prob, but is forced to be valid one

			pzx1x2 = pzcx1x2 * prob_joint[None,:,:]
			entzcx1x2 = -np.sum(pzx1x2 * np.log(pzcx1x2))
			joint_mi = entz - entzcx1x2
			# take the maximum element
			mizx1 = ut.calcMI(pzcx1 * px1[None,:])
			mizx2 = ut.calcMI(pzcx2 * px2[None,:])
			cmix1x2cz = joint_mi - (mizx1 + mizx2) + mix1x2 # by definition
			# loss calculation
			
			tmp_loss = (1+gamma)*joint_mi - gamma * mizx1 - gamma * mizx2
			# clustering accuracy
			tmp_result += [entz,mizx1,mizx2,joint_mi,tmp_loss,cmix1x2cz]
			res_all[rec_idx,:] = np.array(tmp_result)
			print("gamma,{:.3f},nidx,{:},nz,{:},conv,{:},nit,{:},I(X1,X2;Z),{:.5f},H(Z),{:.5f},loss,{:.4f},I(X1;X2|Z),{:.5f}".format(gamma,nn,nz,int(out_dict["conv"]),out_dict["niter"],joint_mi,entz,tmp_loss,cmix1x2cz))
			rec_idx += 1

timenow= datetime.datetime.now()
# result directory
d_cwd = os.getcwd()
d_save_dir = os.path.join(d_cwd,args.output_dir)
os.makedirs(d_save_dir,exist_ok=True)

repeat_cnt = 0
safe_savename_base = "sto_{:}_y{:}b{:}_cr{:.4f}_nzall_{:}".format(args.method,args.ny,args.nb,args.corr,timenow.strftime("%Y%m%d"))
safe_savename = copy.copy(safe_savename_base)
while os.path.isfile(os.path.join(d_save_dir,safe_savename)):
	repeat_cnt +=1
	safe_savename = "{:}_{:}".format(safe_savename_base,repeat_cnt)

# saving the results, numpy array
with open(os.path.join(d_save_dir,safe_savename+".npy"),"wb") as fid:
	np.save(fid,res_all)
# saving the encoders
# saving the configuration in case of error
with open(os.path.join(d_save_dir,safe_savename+"_config.pkl"),"wb") as fid:
	pickle.dump(argsdict,fid)