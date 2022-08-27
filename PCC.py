import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import mdtraj as md
import pytraj as pt
import sys
import re


def plot_group(self):

    l = np.loadtxt('loss.txt')
    print(l.shape)
    plt.figure()
    plt.plot(range(1, len(l)+1), l)
    plt.xlabel('tICs group')
    plt.ylabel('loss')
    plt.savefig('loss.png')
    plt.close()

    c = np.loadtxt('combined_tics.txt')
    print(c.shape)
    plt.figure()
    pcc_to_m1 = []
    for i in range(1,len(c)):
        pcc_to_m1.append(pearsonr(c[i].tolist(), c[i-1].tolist())[0])
    plt.plot(range(2, len(c)+1), pcc_to_m1, label='n vs. n-1')

    pcc_to_m4 = []
    for i in range(4,len(c)):
        pcc_to_m4.append(pearsonr(c[i].tolist(), c[i-4].tolist())[0])
    plt.plot(range(5, len(c)+1), pcc_to_m4, label='n vs. n-4')

    pcc_to_1 = []
    for i in range(1,len(c)):
        print(i,pearsonr(c[i].tolist(), c[0].tolist())[0])
        pcc_to_1.append(pearsonr(c[i].tolist(), c[0].tolist())[0])
    plt.plot(range(2, len(c)+1), pcc_to_1,label='n vs. 1st')
    plt.legend()
    plt.xlabel('tICs group')
    plt.ylabel('PCC')
    plt.savefig('PCC.png')
    plt.close()



# optimal collective motion

def OCM(self):
    # convergence check
    def theta(vec):
          vec = np.array(vec)
          theta = vec/np.sqrt(np.dot(vec,vec))
          return theta

    for i in range(0,len(pcc_to_m4)):
          if abs((pcc_to_m1[i+3]-pcc_to_m1[i+5])/20)<0.0005 and abs((pcc_to_m1[i+5]-pcc_to_m1[i+7])/20)<0.0005:
                  if abs((pcc_to_m4[i]-pcc_to_m4[i+2])/20)<0.0005 and abs((pcc_to_m4[i+2]-pcc_to_m4[i+4])/20)<0.0005:
                          if abs((pcc_to_1[i+3]-pcc_to_1[i+5])/20)<0.0005 and abs((pcc_to_1[i+5]-pcc_to_1[i+7])/20)<0.0005:
                                  n_c = c[i+4]
                                  break

    # write xyz file
    combined_vec = n_c/np.sqrt(np.dot(n_c,n_c))
    np.savetxt('combined_tics.txt', combined_vec)
    np.savetxt('combined_tics_3f.txt', combined_vec, fmt='%.3f')

    pc1 = np.loadtxt('PC1.txt')
    print('pcc with PC1:', np.dot(pc1,combined_vec))

    file = 'combined_tics'
    vec = np.genfromtxt(file+'.txt')
    vec = vec.reshape(int(len(c[1])/3),3)
    np.savetxt(file+'_reshape.txt',vec)

    w = open(file+'.xyz','w')
    w.writelines(str(int(len(c[1])/3))+'\n')
    w.close()
    w = open(file+'.xyz','a')
    w.writelines('C alpha'+'\n')

    f1 = open(file+'_reshape.txt','r')
    lines = f1.readlines()
    for i in range(0,int(len(c[1])/3)):
            newline = 'CA' + '   ' + lines[i]
        # l = 'CA' + '   ' + str(vec[3*i]) + ' ' + str(vec[3*i+1]) + ' ' + str(vec[3*i+2]) + '\n'
            with open(file + '.xyz', 'a') as w:
                    w.write(newline)


# select seed structure from equilibration

def select_seed(self):
    # projection of eq on the combined vector
    raw_data = np.loadtxt('eq.txt')
    for i in range(len(raw_data[0])):
        raw_data[:,i] = raw_data[:,i]-raw_data[0,i]
    print(raw_data.shape)
    proj_on_vec_p = np.dot(raw_data, combined_vec)
    plt.scatter(range(len(raw_data)), proj_on_vec_p, s=10)
    plt.xlabel('frame')
    plt.ylabel('projection')
    # plt.title('projections on 190 tICs-combined vector')
    plt.savefig('proj_eq.png')

    proj_on_vec = proj_on_vec_p.tolist()
    print(max(proj_on_vec))
    print(min(proj_on_vec))
    print(proj_on_vec.index(max(proj_on_vec)))
    print(proj_on_vec.index(min(proj_on_vec)))
    max_index = proj_on_vec.index(max(proj_on_vec))+1

    # select eq frame with largest projection
    traj = pt.load('eq.nc', self.topfile)
    ref = pt.load(self.reference, self.topfile)
    top = ref.topology
    ca = top.select('protein and name CA')
    traj.superpose(ref, frame=0, mask=ca)
    traj.autoimage()
    # pt.write_traj('eq_restart.rst7', traj, frame_indices=max_index, overwrite=True)
    # pt.write_traj('eq_restart.pdb', traj, frame_indices=max_index, overwrite=True)
    max_list = np.array([1, max_index])
    print(max_list)
    pt.write_traj('eq_restart.rst7', traj, frame_indices=max_list, overwrite=True)
    os.rename('eq_restart.rst7.2', 'eq_restart.rst7')
    os.remove('eq_restart.rst7.1')
    trajf = pt.load('eq_restart.rst7','../../r.prmtop')
    pt.write_traj('eq_restart.pdb', trajf, overwrite=True)



def step_size(self):
    # write conf files
    # step size(center)
    # k*RMSD^2*(max-min)
    def bubbleSort(arr):
        n = len(arr)
        for i in range(n):
            # Last i elements are already in place
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr

    sort_proj = bubbleSort(proj_on_vec_p)
    top10 = int(len(sort_proj)*0.1)
    m_top10 = np.median(np.array(sort_proj[0:top10]))
    last10 = int(len(sort_proj)*0.9)
    m_last10 = np.median(np.array(sort_proj[last10:-1]))

    k = 0.15
    p_struc = pt.load('eq.rst7', self.topfile)
    r_struc = pt.load('../../state_B/round_'+str(self.round)+'/eq.rst7', self.topfile)
    ca = p_struc.select('protein and name CA')
    RMSD = pt.rmsd(p_struc, ref=r_struc, mask=ca)
    print('RMSD:', RMSD[0])
    print(abs(m_last10-m_top10))
    center = k*RMSD[0]**2*abs(m_last10-m_top10)
    print('step size of state A:', center)

    # prepare reference pdb file
    count = 0
    ini = 'eq_restart.pdb'
    ref = 'eq_restart_ref.pdb'

    f = open(ini,'r')
    fo = open(ref, 'w')
    l1 = f.readlines()[0]
    fo.write(l1)
    fo.close()

    f = open(ini,'r')
    for line in f.readlines():
            if line[0:4] == 'ATOM':
                    if line[13:15] == 'CA':
                            line = line[0:56]+'1'+line[57:81]
                            count = count +1
                            with open(ref,'a') as m:
                                    m.write(line)
                                    m.close()
                    else:
                            line = line[0:56]+'0'+line[57:81]
                            with open(ref,'a') as m:
                                    m.write(line)
                                    m.close()


    with open(ref,'a') as m:
            m.write('END')
            m.close()
    print('number of CA:',count)


    # write colvar file
    f = open('../../colvar.in', 'r+')
    fr = open('round_'+str(self.round)+'.in', 'w+')
    str1 = r'NUMBER'
    str2 = r'CENTER'
    for line in f.readlines():
            line = re.sub(str1,str(number),line)
            line = re.sub(str2,str(center),line)
            fr.write(line)
    f.close()
    fr.close()

    # write conf file
    fc = open('../../namd.conf', 'r+')
    fcr = open('round_'+str(self.round)+'.conf', 'w+')
    str3 = r'ROUND'
    for line in fc.readlines():
            line = re.sub(str3,str(ID),line)
            fcr.writelines(line)
    fc.close()
    fcr.close()

