# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:30:43 2022

@author: kpadur

"""
import os

timestep = 1
nExperiments = 1
nTimesteps = 10000
nAgents = 1000

# scenarios - change these to change the experiment
cyberattack = False
misinformation = False
coordinated_attack = False

if cyberattack or misinformation or coordinated_attack:
    nMalAgents = int(0.05*nAgents)
else: # no malicious users are introduced if no attack is selected
    nMalAgents = 0
regagents = nAgents - nMalAgents
nProviders = 3
warm_up_time = 2000

# Parameters
alpha = 0.05 # alpha - learning rate
d = 0.80 # delta - satisfaction threshold
eps = 0.10 # epsilon - exploration parameter
misinfo_detect_rate = 0.10 # misinfo detection rate eta_misinfo
cyber_detect_rate = 0.80 # cyber attack detection rate eta_cyber
kNearest = 6 # kappa - number of neighbours
# Attack order (future addition)
same_transition = True
if same_transition:
    Z = [0.03,0.03,0.03] # endpoint
    X = [0.25,0.25,0.25] # endpoint
    P = [0.01,0.01,0.01] # central
    Q = [0.30,0.30,0.30] # central
else:
    P = [0.01,0.04,0.07]
    Q = [0.30,0.16,0.12]
    Z = [0.03,0.05,0.07]
    X = [0.25,0.21,0.20]
pos_initialisation = 100 # initialisation of the position of the malicious agents
rewProb = 0.01 # rho - rewiring probability
tau = 0.90 # tau - experience weight

# Save figures
save_fig = True
output_dir = os.path.join(os.getcwd(), "output_fig")
