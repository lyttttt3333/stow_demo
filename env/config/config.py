
import numpy as np
import torch
from env.config.PATH import *
# todo 
# add new hyperparameters


class Config:
    def __init__(self,kwargs:dict=None) -> None:
        self.robot_config=RobotConfig()

        if kwargs is not None:
            self.update(kwargs)

    def __getitem__(self,key):
        return getattr(self,key)
    
    def update(self,kwargs):
        for key in kwargs:
            self[key].update(kwargs[key])

class RobotConfig:
    def __init__(self):
        self.type="franka"
        self.num=1
        self.position=torch.tensor([[0.06037,-1.49523,0.63282]])
        self.orientation=[None,None]
    
class EpisodeConfig():
    def __init__(self,length:int=100,contain_task:bool=False,pick:bool=False,place:bool=False):
        self.length=length
        self.contain_task=contain_task
        self.pick=pick
        self.place=place
        if contain_task is False:
            self.task_params=None

    def add_task(self,task_params:list=None):
        self.task_params=task_params if self.contain_task is True else None


ep_sequence=[]
ep00=EpisodeConfig(contain_task=True,length=200)
ep00.add_task(
    [
        [torch.tensor([0.06037,-1.49523,0.63282]),None],
        [torch.tensor([0.39933,-1.51923,0.4523]),3],     
    ]
)
ep001=EpisodeConfig(contain_task=True,length=200,pick=True,place=False)
ep001.add_task(
    [
        [torch.tensor([0.39933,-1.51923,0.4523]),None],     
        [torch.tensor([0.19933,-1.51923,0.9523]),3],    
    ]
)
ep10=EpisodeConfig(contain_task=True,length=200,pick=True,place=False)
ep10.add_task(
    [
        [torch.tensor([0.19933,-1.51923,0.9523]),None],
        [torch.tensor([0.01957,-2.36048,1.24274]),3],
    ]
)
ep_sequence.append([ep00])
ep_sequence.append([ep001])
ep_sequence.append([ep10])


GarmentConfig={
            "path":CLOTH_TO_HANG,
            "position":np.array([-0.65481,-1.27712,0.54132]),
            "orientation":np.array([0.47366,-0.32437,-0.46264,-0.67557]),
            "scale":np.array([0.0075, 0.0075, 0.0075]),
        }

import torch
grasp_offset=torch.tensor([0.,0.,0.045])