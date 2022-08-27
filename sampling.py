from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from sys import stdout
import numpy as np
import mdtraj as md



class ForceReporter(object):

    def __init__(self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(kilojoules/mole/nanometer)
        for f in forces:
            self._out.write('%g %g %g\n' % (f[0], f[1], f[2]))




# System Configuration
nonbondedMethod = PME
nonbondedCutoff = 1.0*nanometers  #10 Angstroms
ewaldErrorTolerance = 0.00001
constraints = HBonds  #ntc=2,ntf=2
rigidWater = True     #rigid water model like TIPnP
constraintTolerance = 0.00001

# Integration Options
dt = 0.002*picoseconds
temperature = 310*kelvin
friction = 1.0/picosecond     #gamma_ln=2.0



def add_restraints(
        system: openmm.System,
        crd,
        parm,
        OCM_file,
        scaling: float,
        ss: float):
        """Adds a harmonic potential that restrains the system to a structure."""

        force = openmm.CustomExternalForce("0.5*k*periodicdistance(x, y, z, tx, ty, tz)^2")
        force.addGlobalParameter("k", float(scaling))
        for p in ["tx", "ty", "tz"]:
                force.addPerParticleParameter(p)

        t = md.load(crd, top=parm)
        ca = t.topology.select('protein and name CA')
        # ca_coor = t.xyz[0, ca, :]
        coor = t.xyz[0]
        vec = np.loadtxt(OCM_file)
        count = 0
        for i, atom in list(enumerate(t.topology.atoms)):
                if atom.name == 'CA':
                        # x0,y0,z0 = ca_coor[i,0],ca_coor[i,1],ca_coor[i,2]
                        x0,y0,z0 = coor[i,0],coor[i,1],coor[i,2]
                        a,b,c = vec[count*3],vec[count*3+1],vec[count*3+2]
                        tx,ty,tz = x0+ss*a, y0+ss*b, z0+ss*c
                        # tx,ty,tz = x0+a, y0+b, z0+c
                        force.addParticle(i, [tx,ty,tz])
                        count = count+1
        assert count == len(ca), 'Inconsistent number of CA'
        print("Restraining %d particles."%force.getNumParticles())
        #logging.info("Restraining %d particles.",
        #                  force.getNumParticles())
        system.addForce(force)



def run_bias(parm, crd, OCM_file, scaling: float, ss: float):
        """Minimize energy via openmm."""

        prmtop = AmberPrmtopFile(parm)
        inpcrd = AmberInpcrdFile(crd)

        system = prmtop.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
    constraints=constraints, rigidWater=rigidWater)

        add_restraints(system=system, parm=parm, crd=crd, OCM_file=OCM_file, scaling=float(scaling), ss=ss)

        integrator = LangevinIntegrator(temperature, friction, dt)
        #integrator.setConstraintTolerance(constraintTolerance)
        platform = Platform.getPlatformByName("CUDA")
        #properties = {'DeviceIndex': str(DeviceIndex), 'Precision': 'double'}
        simulation = Simulation(prmtop.topology, system, integrator, platform)
        simulation.context.setPositions(inpcrd.positions)

        if inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

        print('Performing energy minimization...')
        simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(310*kelvin)

        simulation.reporters.append(ForceReporter('force.txt', 50000))
        simulation.reporters.append(DCDReporter('bias.dcd', 2500))
        simulation.reporters.append(StateDataReporter('biasout.csv', 50000, step=True, time=True, progress=True, potentialEnergy=True, temperature=True, remainingTime=True, speed=True, totalSteps=200000, separator='\t'))

        # Simulate
        simulation.currentStep = 0
        simulation.step(200000)



def run_eq(parm, crd):
        parm = AmberPrmtopFile(parm)
        crd = AmberInpcrdFile(crd)

        system = parm.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
    constraints=constraints, rigidWater=rigidWater)

        integrator = LangevinIntegrator(temperature, friction, dt)
        # integrator.setConstraintTolerance(constraintTolerance)
        platform = Platform.getPlatformByName('CUDA')
        #properties = {'DeviceIndex': str(DeviceIndex), 'Precision': 'double'}
        simulation = Simulation(parm.topology, system, integrator, platform)
        simulation.context.setPositions(crd.positions)

        if crd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*crd.boxVectors)

        print('Performing energy minimization...')
        simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(310*kelvin)

        print('Equilibrating...')
        simulation.reporters.append(DCDReporter('eq.dcd', 10000))
        simulation.reporters.append(StateDataReporter('eqout.csv', 100000, step=True, time=True, progress=True, potentialEnergy=True, temperature=True, remainingTime=True, speed=True, totalSteps=2500000, separator='\t'))

        print('Simulating...')
        simulation.currentStep = 0
        simulation.step(2500000)







