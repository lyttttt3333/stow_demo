
import numpy as np
import torch
from env.config.PATH import *
from omni.isaac.core.utils.rotations import euler_angles_to_quat
# todo 
# add new hyperparameters


place_offset=torch.tensor([[-0.3,0.0,0.],[0.7,0.,0.]])
#torch.tensor([[[-0.5,0.,0.2]]])

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
        self.num=2
        self.position=place_offset
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



robot_initial_position1=torch.tensor([0.06037,-1.49523,0.63282])+place_offset.squeeze(0)
robot_initial_position1=torch.tensor([0.06037,0.,0.63282])
robot_initial_position2=torch.tensor([0.46037,0.,0.63282])#+torch.tensor([-0.5,0,0])
robot_initial_position_list=[robot_initial_position1,robot_initial_position2]
ep_sequence=[]
ep00=EpisodeConfig(contain_task=True,length=200)
ep00.add_task(
    [
        [robot_initial_position1,None],
        [torch.tensor([-0.12009,-0.11572,0.03467]),3],   
            
    ]
)
ep01=EpisodeConfig(contain_task=True,length=200,pick=True,place=False)
ep01.add_task(
    [
        [torch.tensor([-0.12009,-0.11572,0.03467]),3],   
        [torch.tensor([-0.19731,-0.26433,1.7833]),3],   
            
    ]
)
ep10=EpisodeConfig(contain_task=True,length=200)
ep10.add_task(
    [
        [robot_initial_position1,None],
        [torch.tensor([0.23985,-0.22354,0.03467]),3],    
    ]
)
ep11=EpisodeConfig(contain_task=True,length=200,pick=True,place=False)
ep11.add_task(
    [
        [torch.tensor([0.23985,-0.22354,0.03467]),3],    
        [torch.tensor([0.20364,-0.27814,1.7833]),3],    
    ]
)
ep_sequence.append([ep00,ep10])
ep_sequence.append([ep01,ep11])

GarmentConfig={
            "path":Fling_dress,
            "position":np.array([0.64835,0.43807,0.18831]),
            "orientation":np.array([0.20007,-0.27577,0.43207,0.83501]),
            "scale":np.array([0.0075, 0.0075, 0.0075]),
        }

RigidConfig={
            "path":BED,
            "position":np.array([0.15618, 0.06288, -0.13856]),
            "orientation":euler_angles_to_quat(np.array([0.,0.,np.pi/2])),
            "scale":np.array([0.004,0.006,0.004]),
        }

import torch
grasp_offset=torch.tensor([0.,0.,0.045])