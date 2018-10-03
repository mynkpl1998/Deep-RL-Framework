from collections import deque
import random


class uniform_sampling():

    def __init__(self):

        self.name = "uniform-sampling"
        self.params = {
            "capacity":None
        }

    def getHyperParams(self):
        print("===================================================================")
        print("Hyper Paramters (%s)\t\t\tCurrent Value"%(self.name))
        print('-------------------------------------------------------------------')
        for i,key in enumerate(self.params.keys()):
            print("%d. %s\t\t\t\t\t\t%s"%(i+1,key,self.params[key]))
        print("===================================================================")

    def setHyperParams(self, capacity):
        self.params["capacity"] = int(capacity)
        self.memory = deque(maxlen=self.params["capacity"])

    def add_sample(self, sample):
        self.memory.append(sample)

    def get_batch(self, size):
        return random.sample(self.memory, k=size)
