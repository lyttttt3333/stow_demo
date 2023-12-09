
import numpy as np
import torch


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
        self.position=np.array([[0,0,0],])
    def update(self,kwargs):
        for key in kwargs:
            setattr(self,key,kwargs[key])
    def __getitem__(self,key):
        return getattr(self,key)
    def __str__(self):
        return str(self.__dict__)
    def reset(self):
        self.__init__()
    def __call__(self):
        return self.__dict__