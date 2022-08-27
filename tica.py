#py35, pyemma

import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from pyemma.coordinates import tica
import os


def load_eq(parm):
        ref = md.load_pdb('ref.pdb')
        topo = ref.topology
        ca = topo.select('protein and name CA')
        # ca = topo.select('(not (resid 15 to 23 or resid 249 to 279 or resid 289 to 292 or resid 501 to 518 or resid 528 to 541 or resid 549 to 553)) and name CA')
        data_load = md.load('eq.dcd', top=parm, atom_indices=ca)
        ref_ca = md.load_pdb('ref.pdb', atom_indices=ca)
        data_load.superpose(reference=ref_ca, 
            atom_indices=ref_ca.topology.select('(resid 15 to 23 or resid 249 to 279 or resid 289 to 292 or resid 501 to 518 or resid 528 to 541 or resid 549 to 553) and name CA'))
        # data = data_load.xyz
        all_ca = [n for n in range(len(ca))]
        const_ca = [n for n in range(15,24)] + [n for n in range(249,280)] + [n for n in range(284,288)] + [n for n in range(496,514)] + [n for n in range(523,537)] + [n for n in range(544,549)]
        tica_ca = [i for i in all_ca if i not in const_ca]
        # data = data_load.xyz[:, tica_ca, :]
        data = data_load.xyz
        dx, dy, dz = data.shape
        for i in range(dy):
            if i in const_ca:
                data[:, i] = [0,0,0]
        data = data.reshape(dx, dy*dz)

        reduced = tica(data, lag=20, var_cutoff=1)
        plt.plot(range(len(reduced.eigenvalues)), reduced.eigenvalues)
        plt.xlabel('tICs')
        plt.ylabel('eigenvalues')
        plt.savefig('./tIC_eigenv.png')
        plt.close()

        np.savetxt('./eq.txt', np.array(data))
        np.savetxt('./tICs.txt', np.transpose(reduced.eigenvectors), fmt='%.8f')
        np.savetxt('./last_ca.txt', data[-1], fmt='%.8f')
        _load = md.load('eq.dcd', top=parm)
        _load.image_molecules()
        _load[-1].save_amberrst7('./eq.rst7')


def PC1():
        open_ca = np.loadtxt('./open/last_ca.txt')
        close_ca = np.loadtxt('./close/last_ca.txt')
        diff = open_ca - close_ca
        np.savetxt('./close/PC1.txt', diff/np.sqrt(np.dot(diff,diff)))
        np.savetxt('./open/PC1.txt', -diff/np.sqrt(np.dot(diff,diff)))


os.chdir('./close')
load_eq(parm='../../fixed.close.prmtop')
os.chdir('../open/')
load_eq(parm='../../fixed.open.prmtop')
os.chdir('../')
PC1()



