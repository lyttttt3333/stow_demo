
import numpy as np
import torch
from env.config.PATH import *
from omni.isaac.core.utils.rotations import euler_angles_to_quat
# todo 
# add new hyperparameters


place_offset=torch.tensor([[-0.3,0.0,0.]])
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



robot_initial_position1=torch.tensor([0.06037,-1.49523,0.63282])+place_offset.squeeze(0)
ep_sequence=[]
ep00=EpisodeConfig(contain_task=True,length=200)
ep00.add_task(
    [
        [robot_initial_position1,None],
        [torch.tensor([0.10138,-0.45458,0.03467]),100],   
            
    ]
)
ep01=EpisodeConfig(contain_task=True,length=200,pick=True,place=False)
ep01.add_task(
    [
        [torch.tensor([0.10138,-0.45458,0.03467]),3],   
        [torch.tensor([0.39135,0.47799,1.56418]),3],   
            
    ]
)
ep_sequence.append([ep00])
ep_sequence.append([ep01])


garment_configs=[]
GarmentConfig={
            "path":Fling_dress,
            "position":np.array([0.84139,0.2338,0.34144]),
            "orientation":np.array([0.23764,-0.24413,0.30751,0.88846]),
            "scale":np.array([0.0075, 0.0075, 0.0075]),
        }
garment_configs.append(GarmentConfig)
GarmentConfig1={
            "path":pile1,
            "position":np.array([0.84139,-0.07035,0.34144]),
            "orientation":np.array([-0.10213,0.32503,-0.67489,-0.65456]),
            "scale":np.array([0.0075, 0.0075, 0.0075]),
        }
garment_configs.append(GarmentConfig1)
GarmentConfig2={
            "path":pile2,
            "position":np.array([0.84139,0.73239,0.34144]),
            "orientation":np.array([0.23764,-0.24413,0.30751,0.88846]),
            "scale":np.array([0.0075, 0.0075, 0.0075]),
        }
garment_configs.append(GarmentConfig2)
GarmentConfig3={
            "path":pile3,
            "position":np.array([0.84139,1.51593,0.34144]),
            "orientation":np.array([0.23764,-0.24413,0.30751,0.88846]),
            "scale":np.array([0.0075, 0.0075, 0.0075]),
        }
#garment_configs.append(GarmentConfig3)
GarmentConfig4={
            "path":pile4,
            "position":np.array([0.84139,0.689,0.34144]),
            "orientation":np.array([0.23764,-0.24413,0.30751,0.88846]),
            "scale":np.array([0.0075, 0.0075, 0.0075]),
        }
garment_configs.append(GarmentConfig4)



RigidConfig={
            "path":BED,
            "position":np.array([0.15618, 0.06288, -0.13856]),
            "orientation":euler_angles_to_quat(np.array([0.,0.,np.pi/2])),
            "scale":np.array([0.004,0.006,0.004]),
        }

import torch
grasp_offset=torch.tensor([0.,0.,0.045])