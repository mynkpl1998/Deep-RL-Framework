class dumpData():

    def __init__(self,exp_name,explore_policy,algorithm_object):
        self.exp_name = exp_name
        self.explore_policy_object = explore_policy
        self.algorithm_object = algorithm_object

    def export_csv(self):
        save_string = str(self.exp_name)
        self.explore_policy_object.actions_stat.to_csv(r'../data/%s_actions_stat.csv'%(save_string))
        self.algorithm_object.loss_stat.to_csv(r'../data/%s_loss_stat.csv'%(save_string))
        self.algorithm_object.total_episode_reward.to_csv(r'../data/%s_total_reward_stat.csv'%(save_string))
