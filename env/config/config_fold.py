
import numpy as np
import torch
from env.config.PATH import *
from omni.isaac.core.utils.rotations import euler_angles_to_quat
# todo 
# add new hyperparameters


place_offset=torch.tensor([[-0.4,-1.7,0.3]])
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
        self.num=1
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



robot_initial_position1=torch.tensor([0.46765,0.00516,0.37345])+place_offset.squeeze(0)

ep_sequence=[]
ep00=EpisodeConfig(contain_task=True,length=200)
ep_sequence=[]

###########################
ep00.add_task(
    [
        [robot_initial_position1,None],
        [torch.tensor([0.16989,0.4424,0.62284]),50],     
    ]
)
ep001=EpisodeConfig(contain_task=True,length=200)
ep001.add_task(
    [
        [torch.tensor([0.16989,0.4424,0.62284]),None],    
        [torch.tensor([0.16989,0.4424,0.42284]),50],   
    ]
)
ep10=EpisodeConfig(contain_task=True,length=200,pick=True,place=False)
ep10.add_task(
    [
        [torch.tensor([0.16989,0.4424,0.42284]),None],
        [torch.tensor([0.16989,0.2811,0.70848]),50],
    ]
)
ep101=EpisodeConfig(contain_task=True,length=200,pick=False,place=True)
ep101.add_task(
    [
        [torch.tensor([0.16989,0.2811,0.70848]),None],
        [torch.tensor([0.10289,0.04736,0.43604]),50],
    ]
)

ep102=EpisodeConfig(contain_task=True,length=200,pick=False,place=False)
ep102.add_task(
    [
        [torch.tensor([0.10289,0.04736,0.43604]),None],
        [robot_initial_position1,30],
    ]
)

###########################
ep200=EpisodeConfig(contain_task=True,length=200)
ep200.add_task(
    [
        [robot_initial_position1,None],
        [torch.tensor([-0.06015,-0.13247,0.73604]),30],
    ]
)
ep201=EpisodeConfig(contain_task=True,length=200,)
ep201.add_task(
    [
        [torch.tensor([-0.06015,-0.13247,0.73604]),None],
        [torch.tensor([-0.06015,-0.13247,0.42304]),50],
    ]
)


ep202=EpisodeConfig(contain_task=True,length=200,pick=True,place=False)
ep202.add_task(
    [
        [torch.tensor([-0.06015,-0.13247,0.42304]),None],
        [torch.tensor([0.0455,0.00627,0.58405]),100],
    ]
)
ep203=EpisodeConfig(contain_task=True,length=200,pick=False,place=True)
ep203.add_task(
    [
        [torch.tensor([0.0455,0.00627,0.58405]),None],
        [torch.tensor([0.10499,0.13657,0.43604]),100],
    ]
)   

ep30=EpisodeConfig(contain_task=True,length=200)
ep30.add_task(
    [
        [torch.tensor([0.10499,0.13657,0.43604]),None],
        [torch.tensor([-0.01998,0.33428,0.62512]),100],
    ]
)   
ep300=EpisodeConfig(contain_task=True,length=200)
ep300.add_task(
    [
        [torch.tensor([-0.01998,0.33428,0.62512]),None],
        [torch.tensor([-0.01998,0.33428,0.41951]),100],
    ]
)  
ep301=EpisodeConfig(contain_task=True,length=200,pick=True,place=False)
ep301.add_task(
    [
        [torch.tensor([-0.01998,0.33428,0.41951]),None],
        [torch.tensor([0.13383,0.26981,0.65432]),100],
    ]
)  
ep302=EpisodeConfig(contain_task=True,length=200,pick=False,place=True)
ep302.add_task(
    [
        [torch.tensor([0.13383,0.26981,0.65432]),None],
        [torch.tensor([0.20869,0.22656,0.5922]),50],
        [torch.tensor([0.29869,0.21656,0.4797]),50],
    ]
)  
ep40=EpisodeConfig(contain_task=True,length=200)
ep40.add_task(
    [
        [torch.tensor([0.29869,0.21656,0.4797]),None],
        [torch.tensor([-0.04752,0.20484,0.71992]),50],
        [torch.tensor([-0.02636,0.18378,0.41498]),50],
    ]
)  
ep400=EpisodeConfig(contain_task=True,length=200,pick=True,place=True)
ep400.add_task(
    [
        [torch.tensor([-0.02636,0.18378,0.41498]),None],
        [torch.tensor([0.04947,0.04216,0.6562]),50],
        [torch.tensor([0.0919,0.01741,0.5992]),50],
        [torch.tensor([0.18731,-0.05501,0.47992]),50],
    ]
)   
ep_sequence.append([ep00])
ep_sequence.append([ep001])
ep_sequence.append([ep10])
ep_sequence.append([ep101])
ep_sequence.append([ep102])
ep_sequence.append([ep200])
ep_sequence.append([ep201])
ep_sequence.append([ep202])
ep_sequence.append([ep203])
ep_sequence.append([ep30])
ep_sequence.append([ep300])
ep_sequence.append([ep301])
ep_sequence.append([ep302])        
ep_sequence.append([ep40])
ep_sequence.append([ep400])

GarmentConfig={
            "path":TOP2,
            "position":np.array([-0.59244,0.38773,0.54132]),
            "orientation":euler_angles_to_quat(np.array([0.,0.,-109.931/180*np.pi])),
            "scale":np.array([0.006, 0.006, 0.006]),
        }
        
RigidConfig={
            "path":BED,
            "position":np.array([0.15618, 0.06288, 0.23628]),
            "orientation":euler_angles_to_quat(np.array([0.,0.,np.pi/2])),
            "scale":np.array([0.004,0.006,0.004]),
        }

import torch
grasp_offset=torch.tensor([0.,0.,0.045])