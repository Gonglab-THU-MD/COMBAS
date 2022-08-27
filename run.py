import sampling
from sampling import run_eq
import os

os.chdir('./close/')
run_eq(parm='../../fixed.close.prmtop', crd='bias.rst7')
#os.chdir('../f/')
#run_eq(parm='/export/disk4/ly/chignolin/5awl.prmtop', crd='bias_select.rst7', DeviceIndex=1)
#run_eq(parm='unf.prmtop', crd='unf.rst7', DeviceIndex=0)


