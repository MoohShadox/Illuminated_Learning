from typing import Sequence
import yaml
import numpy as np
import gym, gym_fastsim
import pandas as pd
from .Genotype import *


class Experience(object):

    def __init__(self, *args, **kwargs):
        super(Experience, self).__init__(*args)
        self.env = kwargs["env"] if "env" in kwargs else None
        self.gym_env = None
        self.read("conf/conf.yaml")
    
    def read(self,file_path : str):
        """Construct an evaluator from a YAML file.
        Args:
            file_path (str): path to the yaml file.
        """
        self.env = {}
        with open(file_path) as f:
            docs = yaml.load_all(f, Loader=yaml.FullLoader)
            self.env.update(list(docs)[0]["conf"])
    

    def prepareEnv(self):
        env_name = self.env['gym_name']
        self.gym_env = gym.make(self.env['gym_name'], **self.env['env_params'])

    def get_criterion_from_logs(self, logs : pd.DataFrame, criterion = "dist_obj"):
        L = []      
        for i in np.unique(logs.epoch):
            c = np.array(logs.loc[logs["epoch"] == i, criterion].tail(1))[0]
            L.append(c)
        mean = np.mean(np.array(L))
        std = np.std(np.array(L))
        return mean,std

    def get_multi_criterions_from_logs(self, logs : pd.DataFrame, criterions = ["dist_obj"]):
        results = {}
        for criterion in criterions:
            L = []
            for i in np.unique(logs.epoch):
                c = np.array(logs.loc[logs["epoch"] == i, criterion].tail(1))[0]
                L.append(c)
            mean = np.mean(np.array(L))
            std = np.std(np.array(L))
            results[criterion] = (mean, std)
        return results

    def simple_evaluation(self, genotype : Genotype, criterion = "dist_obj",max_steps = 1000, epochs = 10,render=True):
        logs = self.get_logs(genotype, max_steps = max_steps, epochs = epochs,render=render)
        L = []      
        for i in np.unique(logs.epoch):
            c = np.array(logs.loc[logs["epoch"] == i, criterion].tail(1))[0]
            L.append(c)
        mean = np.mean(np.array(L))
        std = np.std(np.array(L))
        return mean,std


    def mo_evaluation(self, genotype : Genotype, criterions = ["dist_obj"],max_steps = 1000, epochs = 10,render=True):
        logs = self.get_logs(genotype, max_steps = max_steps, epochs = epochs,render=render)
        return self.get_multi_criterions_from_logs(logs,criterions), logs

    def get_logs(self,genotype: Genotype, max_steps = 1000, epochs = 10,render=True):
        if(not self.gym_env):
            self.prepareEnv()
        observation = self.gym_env.reset()
        spec = genotype.get_spec()
        observation, reward, done, info = self.gym_env.step([0]*spec["nb_output"])
        action_scale_factor = self.gym_env.action_space.high
        episode_reward=0
        episode_bd=None
        self.gym_env.enable_display()
        logs = {}
        for epoch in range(epochs):
            episode_log={}
            observation = self.gym_env.reset()
            observation, reward, done, info = self.gym_env.step([0]*spec["nb_output"])
            for t in range(max_steps):
                if render:
                    self.gym_env.render()
                action=genotype.get_action(observation)
                action=action_scale_factor*np.array(action)
                observation, reward, done, info = self.gym_env.step(action)

                if (self.env["episode_reward_kind"] == "cumul"):
                    episode_reward+=reward

                for k in self.env["episode_log"].keys():
                    if (self.env["episode_log"][k] == "cumul"):
                        episode_log[k] = episode_log.get(k,[]) + [episode_log.get(k,[0])[-1] + info[k]]
                    else:
                        episode_log[k] = episode_log.get(k,[]) + [info[k]]
                if(done):
                    break
            logs[epoch] = episode_log
        records = []
        for epoch,log in logs.items():
            arr = np.array(pd.DataFrame(log))
            for r in arr:
                records.append((epoch,*r))
        df = pd.DataFrame.from_records(records,columns=["epoch"]+list(episode_log.keys()))
        return df






    

