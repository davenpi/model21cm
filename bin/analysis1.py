#!/bin/python

###############################################################################
########################################################################

# This script replicates the operations documented in the 
#`tutorial.ipynb` jupyter notebook distributed with the `model21cm` 
# python package. Consult that document for further documentation and 
# explanation of this script.

import os
import emcee
import corner 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import model21cm as m21

#ncpu_cores = os.environ["SLURM_STEP_NUM_TASKS"]
ncpu_cores = 8

# Snip weight 0 elements from EDGES data
m21.edgesdata.snip_data('Weight', 1)

# Setup priors on model parameters
a0 = m21.Parameter('a0', 1530., 1650.)
a1 = m21.Parameter('a1', 450., 900.)
a2 = m21.Parameter('a2', -1400., -800.)
a3 = m21.Parameter('a3', 450., 900.)
a4 = m21.Parameter('a4', -225., -125.)
A = m21.Parameter('A', 0.3, 1.0)
tau = m21.Parameter('tau', 4., 12.)
nu0 = m21.Parameter('nu0', 76., 80.)
w = m21.Parameter('w', 17., 24.)
parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w]
parameters = m21.Parameters(parameter_list=parameterlist)

# Instantiate the standard 21cm model
simulatedmodel = m21.Model()
simulatedmodel.add_parameters(parameterlist)
simulatedmodel.parameters

# Simulate a dataset 
Freq = m21.edgesdata.frequencies
Temp = m21.forwardmodel(Freq, (1565, 650, -1200, 700, -174, 0.43, 6.8, 78.6, 20.0))
dat = {'Frequencies': Freq, 'Temperature': Temp}
df = pd.DataFrame(data=dat)

# Add dataset to model
simulatedmodel.add_data(df)

# Configure MCMC
walkers = 100
steps = 10000
burn = 1000

# Run MCMC
sampler = simulatedmodel.MCMC(nwalkers=walkers, nsteps=steps, ncpu=ncpu_cores)

# Remove burn-in points
_, frame = m21.trim_to_frame(sampler, burn)

# Save Results
datapath = os.environ['STORAGE_DIR']

frame.quantile([0.16, 0.50, 0.84], axis = 0).to_csv(datapath+"/normalmodel_quantile.csv", index = None, header=True)
sampler.to_csv(datapath+"/normalmodel_chains.csv", index = None, header=True)
fig = corner.corner(traces.T, labels= ['a0', 'a1', 'a2', 'a3', 'a4', 'A', 'tau','nu0', 'w'])
fig.savefig(datapath+"/normalmodel_cornerplot.png")

# Repeat for Dark Model 
darkburn=1000
xi = m21.Parameter('xi', -10, 0)
parameterlist = [a0, a1, a2, a3, a4, A, tau, nu0, w, xi]
parameters = m21.Parameters(parameter_list=parameterlist)
simulateddarkmodel = m21.DarkModel()
simulateddarkmodel.add_parameters(parameterlist)
simulateddarkmodel.parameters
Freq = m21.edgesdata.frequencies
Temp = m21.forwardmodel(Freq, (1565, 650, -1200, 700, -174, 0.43, 6.8, 78.6, 20.0, -0.1),True)
dat = {'Frequencies': Freq, 'Temperature': Temp}
df = pd.DataFrame(data=dat)
simulateddarkmodel.add_data(df)
sampler = simulateddarkmodel.MCMC(nwalkers=walkers, nsteps=steps, ncpu=ncpu_cores)
_, frame = m21.trim_to_frame(sampler, darkburn, True)
frame.quantile([0.16, 0.50, 0.84], axis = 0).to_csv(datapath+"/darkmodel_quantile.csv", index = None, header=True)
sampler.to_csv(datapath+"/darkmodel_chains.csv", index = None, header=True)
fig = corner.corner(traces.T, labels= ['a0', 'a1', 'a2', 'a3', 'a4', 'A', 'tau','nu0', 'w','xi'])
fig.savefig(datapath+"/darkmodel_cornerplot.png")





