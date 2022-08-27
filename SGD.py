#base tensorflow, openmm

import numpy as np 
import mdtraj as md
import matplotlib.pyplot as plt
# import tensorflow as tf 
import os

import torch
import torch.optim as optim
from torch.autograd import Variable


def SGD(diff_vec, eigenvecs, Print = False, PrintRound = 1000, Iter = 100000):
        """
        do SGD for given diff_vec and eigenvectors to find to optimal collective motion
        
        :param diff_vec: the differential vector of two conformations, shape N
        :param eigenvecs: the tICs that are used to compose the OCM, shape N * m
        :param Print: whether to print loss during fitting
        :param PrintRound: the interval of each two print
        :param Iter: iterations of the SGD
        :return vec_combine: the normalized OCM we get based on given tICs
        :return alpha: the weight on each tICs to compose the OCM
        :return loss: the final loss, aka the cosine distance of the diffvec and the OCM
        """
        N = eigenvecs.shape[1]
        diff_vec_var = Variable(torch.DoubleTensor(diff_vec), requires_grad = False)
        diff_vec_var = diff_vec_var / torch.linalg.norm(diff_vec_var)
        eigenvec_var = Variable(torch.DoubleTensor(eigenvecs), requires_grad = False)
        alpha_var = Variable(torch.rand(N, 1, dtype = torch.double), requires_grad = True)

        optimizer = optim.Adam([alpha_var], lr=10, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        for i in range(Iter):
            optimizer.zero_grad()
            vec_combine = torch.matmul(eigenvec_var, alpha_var)
            loss = -torch.matmul(vec_combine.T, diff_vec_var) / torch.linalg.norm(vec_combine)
            if i % PrintRound == 0 and Print:
                print(loss.data)
            loss.backward()
            optimizer.step()
        alpha_var = alpha_var / torch.linalg.norm(vec_combine)
        alpha = alpha_var.detach().numpy()
        vec_combine = eigenvecs.dot(alpha)
        np.savetxt('./OCM.txt', vec_combine)
        print("loss=",loss.data.detach().numpy()[0])
        return vec_combine, alpha, loss.data.detach().numpy()[0]


'''
def SGD():
	data = np.loadtxt('./tICs.txt')
	f_pc1 = np.loadtxt('./PC1.txt')
	l = []
	f_tic = data[0:1500]
	n = len(f_tic)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False
	sess = tf.Session(config=config)

	w0 = tf.Variable(tf.expand_dims(np.random.randn(n),0))
	tic = tf.constant(f_tic)
	pc1 = tf.expand_dims(f_pc1,0)
	vec = tf.matmul(w0,tic)

	LOSS = -(tf.reduce_sum(tf.multiply(vec,pc1))/tf.sqrt(tf.reduce_sum(tf.multiply(vec,vec))))

	X = tf.placeholder(tf.float32, shape=[None,None])
	#X2 = tf.placeholder(tf.float32,shape=[None])
	Y = tf.placeholder(tf.float32, shape=[None])
	tf.summary.scalar('loss_linear', LOSS)

	#Minimize
	learning_rate = tf.train.exponential_decay(0.2, 100000, 25000, 0.9, staircase=True)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train = optimizer.minimize(LOSS)
	merged = tf.summary.merge_all()

	sess.run(tf.global_variables_initializer())
	#write=tf.summary.FileWriter('log',sess.graph)

	for step in range(100001):
	    loss_val, merg, W_val,_ = sess.run([LOSS,merged,w0,train],
	                                         feed_dict={X:f_tic,
	                                                    Y:f_pc1})


	l.append([n,loss_val, W_val])
	LOSS = loss_val
	assert LOSS < 0, 'Illegal combined vector with obtuse angle'
	weighted_tics = np.dot(W_val, f_tic)
	np.savetxt('./OCM.txt', weighted_tics)
'''



def proj(parm):
    data = np.loadtxt('./eq.txt')
    OCM = np.loadtxt('./OCM.txt')
    data = data - data[0]
    proj = np.dot(data, OCM)
    
    gap_low = np.percentile(proj, 5)
    gap_high = np.percentile(proj, 95)
    print(gap_low)
    print(gap_high)
    print(gap_high-gap_low)
    
    plt.scatter(range(len(data)), proj, s=10)
    plt.axhline(gap_high)
    plt.axhline(gap_low)
    plt.xlabel('frame')
    plt.ylabel('projection')
    plt.savefig('./proj_eq.png')
    plt.close()
    
    proj = proj.tolist()
    print(max(proj))
    print(min(proj))
    print(proj.index(max(proj)))
    print(proj.index(min(proj)))
    max_index = proj.index(max(proj))+1
    
    
    _load = md.load('./eq.dcd', top=parm)
    #_load.autoimage()
    _load.image_molecules()
    _load[max_index].save_amberrst7('./restart.rst7')

    return max(proj)-min(proj)
    #return gap_high-gap_low

parm1 = 'fixed.close.prmtop'
parm2 = 'fixed.open.prmtop'

'''
os.chdir('./close/')
SGD()
close_p = proj(parm=parm1)
print('projection of close:', close_p)
os.chdir('../open/')
SGD()
open_p = proj(parm=parm2)
print('projection of open', open_p)
'''

os.chdir('./close/')
diff_vec = np.loadtxt('./PC1.txt') 
eigenvecs = np.loadtxt('./tICs.txt').T 
SGD(diff_vec, eigenvecs)
close_p = proj(parm=parm1)
print('projection of close:', close_p)
os.chdir('../open/')
diff_vec = np.loadtxt('./PC1.txt') 
eigenvecs = np.loadtxt('./tICs.txt').T 
SGD(diff_vec, eigenvecs)
open_p = proj(parm=parm2)
print('projection of open', open_p)


os.chdir('../')
close_r  = md.load('./close/eq.rst7', top=parm1)
close_load_ca = md.load('./close/eq.rst7', top=parm1, atom_indices=close_r.topology.select('protein and name CA'))

open_r = md.load('./open/eq.rst7', top=parm2)
open_load_ca = md.load('./open/eq.rst7', top=parm2, atom_indices=open_r.topology.select('protein and name CA'))
close_load_ca.superpose(reference=open_load_ca, atom_indices=close_load_ca.topology.select('(resid 15 to 23 or resid 249 to 279 or resid 289 to 292 or resid 501 to 518 or resid 528 to 541 or resid 549 to 553) and name CA'))

RMSD = md.rmsd(close_load_ca, open_load_ca, 0)*10   #unit in Angstrom
print('RMSD: ',RMSD)

scale_s = 0.05*RMSD**2
scale_l = 0.09*RMSD**2
ss_close = 0.1*RMSD**2*close_p
ss_open = 0.1*RMSD**2*open_p
print('scaling factor: ', scale_s, scale_l)
print('step size: ', ss_close, ss_open)

import sampling
from sampling import run_bias
from sampling import run_eq

os.chdir('./close/')
run_bias(parm=parm1, crd='./restart.rst7', OCM_file='./OCM.txt', scaling=float(scale_s), ss=ss_close)

os.chdir('../open/')
run_bias(parm=parm2, crd='./restart.rst7', OCM_file='./OCM.txt', scaling=float(scale_s), ss=ss_open)



def load_bias1(parm, ss):
	ref = md.load('restart.rst7', top=parm)
	ca = ref.topology.select('protein and name CA')
	ref = md.load('restart.rst7', top=parm, atom_indices=ca)
	load = md.load('bias.dcd', top=parm, atom_indices=ca)
	load.superpose(reference=ref, atom_indices=ref.topology.select('(resid 15 to 23 or resid 249 to 279 or resid 289 to 292 or resid 501 to 518 or resid 528 to 541 or resid 549 to 553) and name CA'))
	data = load.xyz
	x,y,z = data.shape
	data = data.reshape(x, y*z)

	vec = np.loadtxt('./OCM.txt')
	xyz = ref.xyz[0].reshape(len(ca)*3,)
	target = xyz + ss*vec
	print(target.shape)
	data = data - target
	data = np.sqrt(np.dot(data, np.transpose(data)))
	dis = [data[i,i] for i in range(len(data))]
	sel = dis.index(min(dis))
	print('bias select: ', sel)

	_load = md.load('bias.dcd', top=parm)
	_load[sel].save_amberrst7('./bias_select.rst7')

'''
def load_bias2(parm, ss):
	ref = md.load('restart.rst7', top=parm)
	ca = ref.topology.select('protein and name CA')
	ref = md.load('restart.rst7', top=parm, atom_indices=ca)
	load = md.load('bias.dcd', top=parm, atom_indices=ca)
	load.superpose(reference=ref)
	data = load.xyz
	x,y,z = data.shape
	data = data.reshape(x, y*z)

	vec = np.loadtxt('./OCM.txt')
	xyz = ref.xyz[0].reshape(len(ca)*3,)
	data = data[:,i] - data[0,i]
	proj = np.dot(data, vec)

	dis = proj.tolist()
	sel = dis.index(min(dis))

	_load = md.load('bias.dcd', top=parm)
	_load[sel].save_amberrst7('./bias_select.rst7')
'''

def load_bias3(parm):
	_load = md.load('bias.dcd', top=parm)
	_load[-1].save_amberrst7('./bias_select.rst7')



os.chdir('../close/')
#_load = md.load('bias.dcd', top=parm1)
#_load[-1].save_amberrst7('./bias_select.rst7')
#load_bias1(parm=parm1, ss=close_p)
load_bias3(parm=parm1)

os.chdir('../open/')
#_load = md.load('bias.dcd', top=parm2)
#_load[-1].save_amberrst7('./bias_select.rst7')
#load_bias1(parm=parm2, ss=open_p)
load_bias3(parm=parm2)








