from __future__ import print_function
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import mdtraj as md
from msmbuilder.decomposition import tICA
import matplotlib.pyplot as plt
import tensorflow as tf



class load_data(object):
	def __init__(self, PARM, REF, ID):
		self.topfile = PARM
		self.reference= REF
		self.round = ID

	# write coordinates of C alpha
	def write_xyz(self):
		traj = md.load('eq.nc', top=self.topfile)
		ref = md.load(self.reference, top=self.topfile)
		top = ref.topology
		ca = top.select('protein and name CA')
		traj.superpose(ref, frame=0, atom_indices=ca,ref_atom_indices=ca)
		traj.autoimage()
		n_traj = traj.xyz[:, ca, :]
		traj_2d = n_traj.reshape(n_traj.n_frames,len(ca)*3)
		np.savetxt('eq.txt',traj_2d)



	# PC1
	def Diff_v(self):
		local = md.load('eq.rst7', self.topfile)
		top = local.topology
		ca = top.select('protein and name CA')
		remote = md.load('../../state_B/round_'+str(self.round)+'/eq.rst7', self.topfile)
		local.superpose(remote, frame=0, atom_indices=ca, ref_atom_indices=ca)
		new_p = local[ca]
		p_2d = new_p.xyz.reshape(local.n_frames, len(ca)*3)
		new_r = remote[ca]
		r_2d = new_r.xyz.reshape(remote.n_frames, len(ca)*3)
		diff = p_2d - r_2d
	 	np.savetxt('PC1.txt', -diff[0]/np.sqrt(np.dot(diff[0],diff[0])))
	 	np.savetxt('../../state_B/round_'+str(self.round)+'PC1.txt', 
	 		       -diff[0]/np.sqrt(np.dot(diff[0],diff[0])))



	# perform tICA
	def eq_tICA(self):
		data = np.genfromtxt('eq.txt')
		tica = tICA(lag_time=10)
		reduced = tica.fit_transform(data)
		plt.plot(range(len(reduced.eigenvalues)), reduced.eigenvalues)
		plt.xlabel('tICs')
		plt.ylabel('eigenvalues')
		plt.savefig('tIC_eigenv.png')
		np.savetxt('tICs.txt', np.transpose(reduced.eigenvectors), fmt='%.8f')
		plt.close()




	# SGD
	def SGD(self):
		l = []
		for n in range(10, len(ceil(len(data)/10)/2)*10, 10):
		    f_tic = np.loadtxt('tICs.txt')[0:n]
		    f_pc1 = np.loadtxt('PC1.txt')

		    config = tf.ConfigProto()
		    config.gpu_options.allow_growth = True
		    config.allow_soft_placement = True
		    config.log_device_placement = False
		    sess = tf.Session(config=config)

		    w0 = tf.Variable(tf.expand_dims(np.random.randn(n),0))
		    tic = tf.constant(f_tic)
		    pc1 = tf.expand_dims(f_pc1,0)
		    #vec = tf.reduce_sum(tf.matmul(w0,tic),0)
		    vec = tf.matmul(w0,tic)
		    #LOSS = tf.convert_to_tensor(-pearson(vec))

		    LOSS = -(tf.reduce_sum(tf.multiply(vec,pc1))/tf.sqrt(tf.reduce_sum(tf.multiply(vec,vec))))

		    X = tf.placeholder(tf.float32,shape=[None,None])
		    #X2 = tf.placeholder(tf.float32,shape=[None])
		    Y = tf.placeholder(tf.float32,shape=[None])
		    tf.summary.scalar('loss_linear',LOSS)

		    #Minimize
		    learning_rate = tf.train.exponential_decay(0.2, 120000, 30000, 0.9, staircase=True)
		    optimizer=tf.train.GradientDescentOptimizer(learning_rate)
		    train=optimizer.minimize(LOSS)
		    merged=tf.summary.merge_all()

		    sess.run(tf.global_variables_initializer())
		    #write=tf.summary.FileWriter('log',sess.graph)

		    for step in range(120001):
		        loss_val,merg,W_val,_=sess.run([LOSS,merged,w0,train],
		                                             feed_dict={X:f_tic,
		                                                        Y:f_pc1})


		    l.append([n,loss_val,W_val])
		    #write.close()

		np.savetxt('loss.txt',[l[i][1] for i in range(len(l))])

		tic = np.loadtxt('tICs.txt')
		weighted_tics = []
		for i in range(len(ceil(len(data)/10)/2)):
		    s_tic = tic[0:10+10*i]
		    weighted_tics.append(np.dot(l[i][2],s_tic))
		weighted_tics = np.array(weighted_tics)
		np.savetxt('combined_tics.txt',weighted_tics[:,0])


if __name__ == "__main__":
    LOAD = load_data(PARM, REF, ID)
    load_data.write_xyz()
    load_data.eq_tICA()
    load_data.Diff_v()
    load_data.SGD()
