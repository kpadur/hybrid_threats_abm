#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:39:56 2022

@author: kartpadur
"""
import random
import numpy as np

class ServiceProvidersBehaviour:
    """Service providers' behaviour"""

    def __init__(self, providers, agents, P, Q, Z, X, nTimesteps):
        """A two-state Markov model is used to model the central service states and
        end-point service levels of each provider. Each provider has different transition
        probabilities for central service states and end-point service levels"""
        self.providers = providers
        self.agents = agents
        self.P = P
        self.Q = Q
        self.Z = Z
        self.X = X
        self.nTimesteps = nTimesteps
        self.center = np.full((self.nTimesteps+1, len(self.providers)), fill_value = 1, dtype = int)
        self.endpoint = np.full((len(self.agents), self.nTimesteps+1, len(self.providers)),
                                fill_value = 1, dtype = int)
        self.serviceStates = [1,0]
        self.serviceLevels = [1,0]
        self.center_vector = np.zeros((self.nTimesteps, len(self.providers), 2), dtype = float)
        self.endpoint_vector = np.zeros((self.nTimesteps, len(self.providers), 2), dtype = float)

    def show_init_central_state(self):
        """
        center : array
            Central service states.
        """
        return self.center

    def central_state(self, provider, target, attack_method, attack_decision, timestep):
        """
        center : array
            Central service states modelled with two-state Markov model.
        """
        if timestep == 1:
                cState = 1
        elif attack_decision==1 and target == provider and (attack_method == 0 or attack_method == 2): # 0 - cyberattack, 2 - cyber & misinfo
            # Note: should be probabilistic - there is some probability that the central state will
            # 'survive' an attack and not go unavailable.
            cState = 0
        elif timestep == 1:
            cState = 1
        elif self.center[timestep-1,provider] == 1:
            cState = random.choices(self.serviceStates,\
                            weights = ((1-self.P[provider]), self.P[provider]), k = 1)[0]
        else:
            cState = random.choices(self.serviceStates,\
                            weights = (self.Q[provider], (1-self.Q[provider])), k = 1)[0]
        self.center[timestep,provider]=cState
        return self.center

    def endpoint_level(self, provider, center, timestep):
        """
        endpoint : array
            Endpoint service states modelled with two-state Markov model.
        """
        for agent in self.agents:
            if timestep == 1:
                    eLevel = 1
            elif center[timestep,provider]==1 and self.endpoint[agent,timestep-1,provider] == 1:
                eLevel = random.choices(self.serviceLevels,\
                            weights = ((1-self.Z[provider]), self.Z[provider]), k = 1)[0]
            elif center[timestep,provider]==1 and self.endpoint[agent,timestep-1,provider] == 0:
                eLevel = random.choices(self.serviceLevels,\
                            weights = (self.X[provider], (1-self.X[provider])), k = 1)[0]
            else:
                eLevel = 0
            self.endpoint[agent,timestep,provider] = eLevel
        return self.endpoint
    