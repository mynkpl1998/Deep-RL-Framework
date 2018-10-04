import numpy as np
import torch
import pandas as pd
from collections import Counter


class annealed_ep_greedy():

    def __init__(self):

        self.name = "annealed_ep_greedy"
        self.params = {
            "current_epsilon": None,
            "intital_epsilon": None,
            "final_epsilon": None,
            "episodes_to_anneal": None,
        }
        self.actions = None
        self.policy_net = None
        self.device = None
        self.actions_stat = pd.DataFrame(columns=["Random", "Greedy"])
        self.last_episode_stat = []
        self.latest_episode_count = 0

    def getHyperParams(self):
        print("===================================================================")
        print("Hyper Paramters (%s)\t\t\tCurrent Value"%(self.name))
        print('-------------------------------------------------------------------')
        for i,key in enumerate(self.params.keys()):
            print("%d. %s\t\t\t\t\t%s"%(i+1,key,self.params[key]))
        print("===================================================================")

    def setHyperParams(self,initial_epsilon,final_epsilon,episodes_to_anneal):
        self.params["intital_epsilon"] = float(initial_epsilon)
        self.params["final_epsilon"] = float(final_epsilon)
        self.params["episodes_to_anneal"] = int(episodes_to_anneal)
        self.params["current_epsilon"] = float(initial_epsilon)

    def exploreAction(self,state,curr_epsiode):

        type = None
        if np.random.rand() <= self.params["current_epsilon"]:
            action = np.random.randint(0,high=self.actions)
            type = "Random"
        else:
            with torch.no_grad():
                torch_x = torch.from_numpy(state).to(self.device).float()
                out = self.policy_net.forward(torch_x,bsize=1)
                value,act = out.max(dim=1)
            action = int(act[0])
            type = "Greedy"

        # Save statistics
        if self.latest_episode_count == curr_epsiode:
            self.last_episode_stat.append(type)
        else:
            counts = Counter(self.last_episode_stat)
            self.actions_stat = self.actions_stat.append(counts, ignore_index=True)
            self.actions_stat.fillna(0, inplace=True)
            self.last_episode_stat [:] = []
            self.latest_episode_count = curr_epsiode
            self.last_episode_stat.append(type)

        return action

    def updateEpsilon(self):

        if self.params["current_epsilon"] > self.params["final_epsilon"]:
            self.params["current_epsilon"] -= (self.params["intital_epsilon"] - self.params["final_epsilon"])/self.params["episodes_to_anneal"]
