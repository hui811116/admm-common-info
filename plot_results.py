import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import utils as ut
import dataset as dt
import pickle

argv = sys.argv

filename = os.path.join(os.getcwd(),argv[1])
print_enc = False
for item in os.listdir(filename):
	if ".npy" in item:
		with open(os.path.join(filename,item),"rb") as fid:
			res_load = np.load(fid)
	elif "_config.pkl" in item:
		with open(os.path.join(filename,item),"rb") as fid:
			res_config = pickle.load(fid)
	elif "_encoders.pkl" in item:
		print_enc =True
		with open(os.path.join(filename,item),'rb') as fid:
			encoder_dict = pickle.load(fid)

print(res_config)
# default dataset

datadict = dt.synExpandToy(res_config["ny"],res_config["nb"],res_config["corr"])
px1x2 = datadict['p_joint']
mix1x2 = ut.calcMI(px1x2)
#px1 = datadict['px_list'][0]
#px2 = datadict['px_list'][1]
# headers
# [gamma, nrun, niter, conv, nz, entz, mizx1, mizx2, cmizx1cx2, cimzx2cx1, test, loss]

mix1x2cz = res_load[:,8] - res_load[:,6] + mix1x2
# or equivalently
# mix1x2cz = res_load[8] - res_load[6] + mix1x2
sel_idx = res_load[:,3] == 1

conv_test = res_load[sel_idx,10]
conv_loss = res_load[sel_idx,11]
# configurations
mk_size = 24
fs_lab  = 18
fs_leg  = 16
fs_tick = 16

fig, (ax,ax2,ax3) = plt.subplots(1,3)
ax.grid("on")

clean_mix1x2cz_vs_hz, mivshz_index_list = ut.aggregateResults(res_load[sel_idx,5],mix1x2cz[sel_idx],"min",2)
#print(mivshz_index_list[:,1].astype("int"))
# load the encoders
if print_enc:
	gamma_range = np.geomspace(res_config["gamma_min"],res_config["gamma_max"],num=res_config["gamma_num"])
	with open(os.path.join(filename,"encoder_figure_details.txt"),'w') as fid:

		for tt, idx in enumerate(mivshz_index_list[:,1].astype("int")):
			gamma = res_load[sel_idx,0][idx]
			nn    = res_load[sel_idx,1][idx]
			gidx = list(gamma_range).index(gamma)
			fid.write("list H(Z)={:.6f}".format(clean_mix1x2cz_vs_hz[tt,0])+"\n")
			#print(clean_mix1x2cz_vs_hz[tt,0])
			encoder = encoder_dict[gidx][nn]
			for iz in range(encoder.shape[0]):
				#print("z={:}".format(iz))
				fid.write("z={:}".format(iz)+"\n")
				for ix1 in range(encoder.shape[1]):
					tmp_list= []
					for ix2 in range(encoder.shape[2]):
						tmp_list.append("{:.4f}".format(encoder[iz,ix1,ix2]))
					#print(",".join(tmp_list))
					fid.write(",".join(tmp_list)+"\n")
			#print(encoder)
			double_check_pz = np.sum(encoder * px1x2[None,:,:],axis=(1,2))
			double_check_hz = np.sum(- double_check_pz * np.log(double_check_pz))
			#print("HZ={:.4f}".format(double_check_hz))
			#print('-'*10)
			fid.write("HZ={:.4f}".format(double_check_hz) + "\n")
			fid.write("-"*10 + "\n")
	fid.close()

ax.scatter(clean_mix1x2cz_vs_hz[:,0],clean_mix1x2cz_vs_hz[:,1],mk_size,marker="o",color="tab:red",label=r"Solutions",facecolor="none")

#ax.set_ylim([0,1])
ax.set_ylabel(r"$I(X_1;X_2|Z)$",fontsize=fs_lab)
ax.set_xlabel(r"$H(Z)$",fontsize=fs_lab)
ax.legend(loc="best",fontsize=fs_leg)
ax.tick_params(axis="both",labelsize=fs_tick)

# plot I(X1;X2|Z) versus gamma
#fig, ax2 = plt.subplots(1,3,2)
ax2.grid("on")
ax2.scatter(res_load[sel_idx,0],mix1x2cz[sel_idx],mk_size,marker="o",color="tab:red",label=r"Solutions",facecolor="none")
#ax2.set_ylim([0,1])
ax2.set_xlabel(r"$\gamma$",fontsize=fs_lab)
ax2.set_ylabel(r"$I(X_1;X_2|Z)$",fontsize=fs_lab)
ax2.legend(loc="best",fontsize=fs_leg)
ax2.tick_params(axis="both",labelsize=fs_tick)
# plot Hz versus I(X1;X2|Z)

#fig, ax3 = plt.subplots(1,3,3)
ax3.grid("on")
ax3.scatter(res_load[sel_idx,0],res_load[sel_idx,5],mk_size,marker="o",color="tab:red",label=r"Solutions",facecolor="none")
#ax3.set_ylim([0,1])
ax3.set_xlabel(r"$\gamma$",fontsize=fs_lab)
ax3.set_ylabel(r"$H(Z)$",fontsize=fs_lab)
ax3.legend(loc="best",fontsize=fs_leg)
ax3.tick_params(axis="both",labelsize=fs_tick)

fig, ax = plt.subplots()
ax.grid("on")
ax.scatter(clean_mix1x2cz_vs_hz[:,0],clean_mix1x2cz_vs_hz[:,1],mk_size,marker="o",color="tab:red",label=r"Solutions",facecolor="none")
#ax.set_ylim([0,1])
ax.set_ylabel(r"$I(X_1;X_2|Z)$",fontsize=fs_lab)
ax.set_xlabel(r"$H(Z)$",fontsize=fs_lab)
ax.legend(loc="best",fontsize=fs_leg)
ax.tick_params(axis="both",labelsize=fs_tick)
plt.tight_layout()
#plt.savefig('_'.join(argv[1].split("."))+".eps",format="eps")


## TESTING accuracy
'''
clean_test_vs_gamma = ut.aggregateResults(res_load[sel_idx,0],res_load[sel_idx,10],"max",2)

fig, ax = plt.subplots()
ax.grid("on")
ax.scatter(clean_test_vs_gamma[:,0],clean_test_vs_gamma[:,1],mk_size,marker="o",color="tab:red",label=r"Solutions",facecolor="none")
#ax.scatter(res_load[sel_idx,0],conv_test,mk_size,marker="o",color="tab:red",label=r"Solutions",facecolor="none")
ax.set_ylabel(r"Testing Accuracy",fontsize=fs_lab)
ax.set_xlabel(r"$\gamma$",fontsize=fs_lab)
ax.legend(loc="best",fontsize=fs_leg)
ax.tick_params(axis="both",labelsize=fs_tick)
plt.tight_layout()
plt.savefig("_".join(argv[1].split("."))+"_acc.eps",format="eps")
'''
plt.show()