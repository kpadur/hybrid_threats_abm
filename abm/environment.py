#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:34:56 2022

@author: kartpadur
"""

import numpy as np
import networkx as nx
from networkx.generators.random_graphs import watts_strogatz_graph

np.random.seed(0)

class Environment:
    """Environment"""

    def __init__(self, nAgents, nMalAgents, nProviders, nTimesteps):
        """Creates the environment that consists of a three types of agents: regular agents,
        malicious agents, and service providers. Regular agents and malicious agents are
        connected with a social network graph."""
        self.nAgents = nAgents
        self.nMalAgents = nMalAgents
        self.nProviders = nProviders
        self.nTimesteps = nTimesteps
        self.network = nx.Graph()
        self.neighbours = dict
        self.agents, self.regagents, self.malagents, self.providers = [], [], [], []

    def form_network(self, kNearest, rewProb):
        """
        network : graph
            Graph that represents a social network, including agents and their
            connections to each other.
        neighbours : dictionary
            Agents who are connected to each other on a graph.
        """
        graph = watts_strogatz_graph(self.nAgents, kNearest, rewProb)
        self.network.add_nodes_from(graph)
        self.network.add_edges_from(graph.edges)
        self.neighbours = nx.to_dict_of_lists(graph)
        return self.network, self.neighbours

    def create_agents(self):
        """
        agents : list
        regagents : list
        malagents : list
        """
        for agent in self.neighbours:
            self.agents.append(agent)
        while len(self.malagents) < self.nMalAgents:
            malagent = np.random.choice(self.agents)
            if malagent not in self.malagents:
                self.malagents.append(malagent)
        self.regagents = [regagent for regagent in self.agents\
                          if regagent not in self.malagents]
        return self.agents, self.regagents, self.malagents

    def create_providers(self):
        """
        providers : list
        """
        for provider in range(self.nProviders):
            self.providers.append(provider)
        return self.providers

    def create_attributes(self):
        """
        Agents' attributes
        """
        for agent in self.agents:
            self.network.nodes[agent]["C"] = self.neighbours[agent] # neighbours
            self.network.nodes[agent]["zeta"] = np.zeros(self.nTimesteps+1, dtype = int) # communication partners
            self.network.nodes[agent]["a"] = np.full(self.nTimesteps+1, fill_value = -1, dtype=int) # actions
            self.network.nodes[agent]["n(a)"] = np.zeros(self.nProviders, dtype=int) # count of actions
            self.network.nodes[agent]["r(a)"] = np.zeros(self.nTimesteps+1, dtype=int) # rewards
            self.network.nodes[agent]["n(r)"] = np.zeros(self.nProviders, dtype=int) # count of positive rewards
            self.network.nodes[agent]["Q(a)"] = np.zeros(self.nProviders, dtype=float) # action-value estimates
            self.network.nodes[agent]["Phi(o)"] = np.zeros((2, self.nProviders), dtype = float) # opinion-value estimates
           