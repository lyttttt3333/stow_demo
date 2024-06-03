
import numpy as np
import torch
from env.config.PATH import *
from omni.isaac.core.utils.rotations import euler_angles_to_quat
# todo 
# add new hyperparameters


place_offset=torch.tensor([[-0.65,0.,0.5]])

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
        self.orientation=[None]
    
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

ep00=[
        [torch.tensor([-0.44393,-0.3,1.55823]),None,None],  
        [torch.tensor([-0.34393,-0.0,1.55823]),None,None],  
        [torch.tensor([-0.34393,-0.0,1.4054]),None,None], 
    ]

ep11=[
        [None,torch.tensor([-0.48333,0.82575,1.79849]),torch.tensor([-0.78477,0.48995,1.79849]),],
        [None,torch.tensor([0.65775,0.2003,1.66116]),torch.tensor([-0.80282,0.3686,1.65262]),],  
        [None,torch.tensor([0.35775,0.2003,0.87173]),torch.tensor([-0.74086,-0.58022,1.47973]),], 
        [None,torch.tensor([0.35775,0.2003,0.87173]),torch.tensor([-0.31494,-0.28949,1.42068]),], 
        [None,torch.tensor([0.35775,0.2003,0.87173]),torch.tensor([-0.00227,-0.42359,1.36109]),], 
        [None,torch.tensor([0.35775,0.2003,0.87173]),torch.tensor([-0.06306,-0.17359,1.02131]),], 
        #[None,torch.tensor([0.00775,0.2003,1.07173]),torch.tensor([-0.74086,-0.58022,1.47973]),], 
    ]

ep12=[
        [None,torch.tensor([-0.18266,-0.27408,1.42152]),None,],
        [None,torch.tensor([-0.47344,-0.16961,1.42152]),None],  
        [None,torch.tensor([-0.50344,0.27617,1.42152]),None], 
        [None,torch.tensor([-0.63344,0.42617,1.42152]),None], 
        [None,torch.tensor([-0.05975,0.00617,1.0145]),None], 
    ]

rest_position=[None,None,torch.tensor([-0.64065,-0.86618,1.6235])]
rest_position2=[torch.tensor([-0.34393,-0.0,1.6054]),None,None,]


RigidConfig={
            "path":BED,
            "position":np.array([-0.61832, 0.79868, 0.82689]),
            "orientation":euler_angles_to_quat(np.array([0.,0.,np.pi/2])),
            "scale":np.array([0.014,0.006,0.004]),
        }

import torch
grasp_offset=torch.tensor([0.,0.,0.045])


book_list=[]

BOOKConfig={
            "path":BOOK1,
            "position":np.array([-0.0182, 0.22416, 0.96991]),
            "orientation":np.array([-67.396/180*np.pi,0.,0.]),
            "scale":np.array([1.,0.8,2])*0.01,
        }
book_list.append(BOOKConfig)

BOOKConfig={
            "path":BOOK2,
            "position":np.array([-0.0182, 0.21429, 0.89063]),
            "orientation":np.array([-67.62/180*np.pi,0.,0.]),
            "scale":np.array([1.,0.8,3.])*0.01,
        }
book_list.append(BOOKConfig)

BOOKConfig={
            "path":BOOK3,
            "position":np.array([-0.0182, 0.29397, 0.95972]),
            "orientation":np.array([-60.668/180*np.pi,0.,0.]),
            "scale":np.array([1.,1.,1.])*0.01,
        }
book_list.append(BOOKConfig)

BOOKConfig={
            "path":BOOK4,
            "position":np.array([-0.0182, 0.365, 0.95599]),
            "orientation":np.array([-46.099/180*np.pi,0.,0.]),
            "scale":np.array([1.,1.,1.])*0.01,
        }
book_list.append(BOOKConfig)


config_list=[]
RigidConfig={
            "path":DESK,
            "position":np.array([0., 0.03477, 0.]),
            "orientation":euler_angles_to_quat(np.array([0.,0.,0.])),
            "scale":np.array([1.4,0.6,1.])*0.01,
        }
config_list.append(RigidConfig)
RigidConfig={
            "path":BOOK_STOP,
            "position":np.array([-0.04272, 0.17297, 0.87243]),
            "orientation":euler_angles_to_quat(np.array([0.5*np.pi,0.,0.])),
            "scale":np.array([1.,1.,0.3])*0.01,
        }
config_list.append(RigidConfig)