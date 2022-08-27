import numpy as np
import mdtraj as md

parm1 = 'fixed.close.prmtop'
parm2 = 'fixed.open.prmtop'
ref = md.load_pdb('ref.pdb')
ref_p = md.load_pdb('ref.pdb', atom_indices=ref.topology.select('protein'))


# ref1 = md.load('/export/disk4/ly/chignolin/unf/unf.rst7', top=parm1)
# ref_p1 = md.load('/export/disk4/ly/chignolin/unf/unf.rst7', top=parm1, 
#       atom_indices=ref1.topology.select('protein'))

file1 = []
file2 = []
for i in range(1,5):
        # file1.append('/export/disk4/ly/chignolin/run2/round_'+str(i)+'/unf/eq.dcd')
        # file2.append('/export/disk4/ly/chignolin/run2/round_'+str(i)+'/f/eq.dcd')
        file1.append('./round_'+str(i)+'/close/eq.dcd')
        file2.append('./round_'+str(i)+'/open/eq.dcd')

print(file1)

traj1 = md.load(file1, top=parm1, atom_indices=ref.topology.select('protein'))
traj1.superpose(reference=ref_p, atom_indices=ref_p.topology.select('(resid 15 to 19 or resid 289 to 292 or resid 501 to 518 or resid 528 to 541 or resid 549 to 553) and name CA'))

traj1.save('close_1-5_0.05_ss.dcd')

#m1 = traj1[0]
#for i in range(1, int(len(traj1)/10)):
#       m1 = m1.join(traj1[i*10])
#m1.save('unf_p.dcd')

traj2 = md.load(file2, top=parm2, atom_indices=ref.topology.select('protein'))
traj2.superpose(reference=ref_p, atom_indices=ref_p.topology.select('(resid 15 to 19 or resid 289 to 292 or resid 501 to 518 or resid 528 to 541 or resid 549 to 553) and name CA'))

traj2.save('open_1-5_0.05ss.dcd')




