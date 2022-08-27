import sampling
from sampling import run_eq
import os

#os.chdir('./unf/')
#run_eq(parm='/export/disk4/ly/chignolin/unfolded/unf.prmtop', crd='bias_select.rst7', DeviceIndex=0)
os.chdir('./open/')
run_eq(parm='../../fixed.open.prmtop', crd='bias.rst7')
#run_eq(parm='unf.prmtop', crd='unf.rst7', DeviceIndex=0)





