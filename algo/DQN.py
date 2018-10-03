import torch
import torch.nn as nn
import pandas as pd

class DQN():

    def __init__(self):
        self.name = "DQN"
        self.params = {
            'batch_size': None,
            'gamma': None,
            'freeze_steps': None,
            'max_steps': None,
            'max_episodes': None,
            'optimizer': None,
            "learn rate": None,
            "loss func": None
        }
        self.replay = None
        self.target_model = None
        self.main_model = None
        self.device = None
        self.optimizer = None
        self.loss = None
        self.loss_stat = pd.DataFrame(columns=["loss"])
        self.total_episode_reward = pd.DataFrame(columns=["total reward"])


    def getHyperParams(self):
        print("===================================================================")
        print("Hyper Paramters (%s)\t\t\t\t\tCurrent Value"%(self.name))
        print('-------------------------------------------------------------------')
        for i,key in enumerate(self.params.keys()):
            print("%d. %s\t\t\t\t\t\t\t%s"%(i+1,key,self.params[key]))
        print("===================================================================")

    def setHyperParams(self,batch_size,gamma,freeze_steps,max_steps,max_episodes,optimizer,lr,loss):
        self.params["batch_size"] = int(batch_size)
        self.params["gamma"] = float(gamma)
        self.params["freeze_steps"] = int(freeze_steps)
        self.params["max_steps"] = int(max_steps)
        self.params["max_episodes"] = int(max_episodes)
        self.params["learn rate"] = float(lr)

        if optimizer == "adam":
            self.params["optimizer"] = "adam"
            self.optimizer = torch.optim.Adam(self.main_model.parameters(),lr=self.params["learn rate"])
        elif optimizer == "rmsprop":
            self.params["optimizer"] = "rmsprop"
            self.optimizer = torch.optim.RMSprop(self.main_model.parameters(),lr=self.params["learn rate"])
        else:
            raise ValueError("%s is not a valid optimizer. Use adam or rmsprop"%(optimizer))

        if loss == "absolute":
            self.params["loss func"] = "m. absolute"
            self.loss = nn.L1Loss()
        elif loss == "huber":
            self.params["loss func"] = "huber"
            self.loss = nn.SmoothL1Loss()
        elif loss == "mse":
            self.params["loss func"] = "m. squarred"
            self.loss = nn.MSELoss()
        else:
            raise ValueError("%s is not a valid loss. Use mse,absolute,huber."%(loss))

    def store_episode_reward(self, value):
        self.total_episode_reward = self.total_episode_reward.append({'total reward':value}, ignore_index=True)

    def update(self):

        batch = self.replay.get_batch(size=self.params["batch_size"])
        current_states, actions, rewards, next_states, game_over = [],[],[],[],[]
        for sample in batch:
            current_states.append(sample.prev_state)
            actions.append(sample.action)
            rewards.append(sample.reward)
            next_states.append(sample.next_state)
            game_over.append(sample.done)

        # mask game_over
        masked_game_over = [not i for i in game_over]

        current_states_torch = torch.tensor(current_states).to(self.device).float()
        actions_torch = torch.tensor(actions).to(self.device)
        rewards_torch = torch.tensor(rewards).to(self.device).float()
        next_states_torch = torch.tensor(next_states).to(self.device).float()
        game_over_torch = torch.tensor(masked_game_over).to(self.device).float()

        # calculate target values
        Q_next_target = self.target_model.forward(next_states_torch,bsize=self.params["batch_size"]).detach()
        next_state_values, _ = Q_next_target.max(dim=1)
        next_state_values = torch.mul(game_over_torch,next_state_values)
        target_values = rewards_torch + (self.params["gamma"]*next_state_values)

        # calculate current estimate
        Q_s = self.main_model.forward(current_states_torch, bsize=self.params["batch_size"])
        Q_s_a = Q_s.gather(dim=1, index= actions_torch.unsqueeze(dim=1)).squeeze(dim=1)

        # make previus grad zero
        self.optimizer.zero_grad()

        loss = self.loss(Q_s_a,target_values)

        # do backpropogate
        loss.backward()

        # update Paramters
        self.optimizer.step()

        self.loss_stat = self.loss_stat.append({"loss":loss.item()}, ignore_index=True)

    def copytoTarget(self):
        self.target_model.load_state_dict(self.main_model.state_dict())
