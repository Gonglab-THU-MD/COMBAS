import numpy as np
import pytraj as pt
import os
import matplotlib
matplotlib.use('Agg')
import pylab as plt


# projection
traj = pt.load('round_ID.dcd','../../r.prmtop')
ref = pt.load('eq_restart.rst7',top='../../r.prmtop')
pt.superpose(traj, ref=ref, mask=':1-86@CA')
traj.autoimage()
n_traj = traj[':1-86@CA']
traj_2d = n_traj.xyz.reshape(n_traj.n_frames,86*3)
np.savetxt('round_ID.txt',traj_2d)
combined_vec = np.loadtxt('combined_150tics.txt')

for i in range(len(traj_2d[0])):
        traj_2d[:,i] = traj_2d[:,i]-traj_2d[0,i]

proj_on_vec = np.dot(traj_2d,combined_vec)
np.savetxt('proj_forced.txt',proj_on_vec)
plt.figure()
plt.plot(range(len(traj_2d)),proj_on_vec)
plt.xlabel('frame')
plt.ylabel('projection')
plt.savefig('proj_forced.png')
plt.close()

# rmsd
refo = pt.load('../../r/round_ID/eq_restart.rst7', top='../../r.prmtop')
data = pt.rmsd(traj, ref=refo, mask=':1-86@CA')
np.savetxt('rmsd_forced.txt', data)
plt.figure()
plt.plot(range(len(data)), data)
plt.xlabel('frame')
plt.ylabel('RMSD')
plt.savefig('rmsd_forced.png')
plt.close()

# select frame
data = data.tolist()
min_index = data.index(min(data[int(len(traj_2d)/2):-1]))+1
print('minimum rmsd of forced:', min(data[int(len(traj_2d)/2):-1]))
min_list = np.array([1, min_index])
pt.write_traj('round_ID_restart.rst7', traj, frame_indices=min_list, overwrite=True)
os.rename('round_ID_restart.rst7.2', 'round_ID_restart.rst7')
os.remove('round_ID_restart.rst7.1')
