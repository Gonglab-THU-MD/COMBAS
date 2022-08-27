from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from sys import stdout
import numpy as np

# Input Files

rid = int(self.round)-1
prmtop = AmberPrmtopFile(self.topfile)
inpcrd = AmberInpcrdFile('round_'+str(self.round)+'_restart.rst7')

# System Configuration

nonbondedMethod = PME
nonbondedCutoff = 1.0*nanometers  #10 Angstroms
ewaldErrorTolerance = 0.00001
constraints = HBonds  #ntc=2,ntf=2
rigidWater = True     #rigid water model like TIPnP

# Integration Options

dt = 0.002*picoseconds
temperature = 300*kelvin
friction = 2.0/picosecond     #gamma_ln=2.0
integrator = LangevinIntegrator(temperature, friction, dt)

# Simulation Options

nsteps = 5000000
platform = Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '0', 'Precision': 'double'}
dcdReporter = DCDReporter('eq.dcd', 5000)

# Prepare the Simulation

topology = prmtop.topology
positions = inpcrd.positions
system = prmtop.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
    constraints=constraints, rigidWater=rigidWater, ewaldErrorTolerance=ewaldErrorTolerance)
# print('total atoms in system:',system.getNumParticles())

# Simulation Properties

# simulation = Simulation(topology, system, integrator, platform, properties)
simulation = Simulation(topology, system, integrator)
simulation.context.setPositions(positions)
# simulation.context.setVelocitiesToTemperature(300*kelvin)
if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

# Minimize and Equilibrate

print('Performing energy minimization...')
simulation.minimizeEnergy()

print('Equilibrating...')
simulation.reporters.append(StateDataReporter('data.csv', 10000, step=True, time=True, progress=True, potentialEnergy=True, temperature=True, remainingTime=True, speed=True, totalSteps=5000000, separator='\t'))

print('Simulating...')
simulation.currentStep = 0
simulation.step(nsteps)
