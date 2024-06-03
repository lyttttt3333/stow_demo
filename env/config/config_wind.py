
import numpy as np
import torch
from env.config.PATH import *
from omni.isaac.core.utils.rotations import euler_angles_to_quat
# todo 
# add new hyperparameters


place_offset=torch.tensor([[0.5,0.5,0.0],[-0.5,0.5,0.0]])
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
        self.orientation=[euler_angles_to_quat(np.array([0.,0.,np.pi])),None]
    
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



ATTACH_PLACE=np.array([[0.08448,-0.02413,0.79431],
                       [-0.08997,-0.02413,0.79431]])

robot_initial_position1=torch.tensor([0.06037,-1.49523,0.63282])+place_offset.squeeze(0)
ep_sequence=[]
ep00=EpisodeConfig(contain_task=True,length=200,pick=True,place=False)
ep00.add_task(
    [
        torch.tensor([-0.60816,0.48735,1.2089]),
        torch.tensor([-0.02073,0.02366,1.82951]), 
        torch.tensor([-0.02888,0.00871,1.55991]),  
        torch.tensor([-0.02888,0.00871,1.55991]), 
    ]
)
ep01=EpisodeConfig(contain_task=True,length=200,pick=True,place=False)
ep01.add_task(
    [
        [torch.tensor([-0.58176,0.47086,1.05648]),3],   
        [torch.tensor([-0.02909,0.00965,1.54965]),3],   
            
    ]
)
ep_sequence.append([ep00])
ep_sequence.append([ep01])


garment_configs=[]
GarmentConfig={
            "path":WIND_DRESS,
            "position":np.array([0.0,0.0,-0.1472]),
            "orientation":np.array([0.70711,0.70711,0.0,0.0]),
            "scale":np.array([0.0075, 0.0075, 0.0075]),
        }
garment_configs.append(GarmentConfig)



RigidConfig={
            "path":BED,
            "position":np.array([0.0, 0.0, -0.1718]),
            "orientation":euler_angles_to_quat(np.array([0.,0.,np.pi/2])),
            "scale":np.array([0.004,0.006,0.004]),
        }

import torch
grasp_offset=torch.tensor([0.,0.,0.045])