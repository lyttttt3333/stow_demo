from omni.isaac.core.utils.types import ArticulationAction
from env.config.config import RobotConfig
from env.Robot.Franka.franka import WrapFranka
from omni.isaac.core import World
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
import torch
# write pick and place primitive
# write movep function
# write fling function
# write hang function


class Robot:
    def __init__(self,world:World,robot_config:RobotConfig,init_position:list,num):
        self.robot_config=robot_config
        self.world=world
        self.init_position=init_position
        self.robot_num=num
        self._robot=[]
        self.load_robot()

    def initialize(self):
        for i in range(self.robot_num):
            self._robot[i].initialize()
            self._robot[i].open()
    
    
    def load_robot(self):
        if self.robot_config.type=="franka":
            for i in range(self.robot_config.num):
                self._robot.append(WrapFranka(world=self.world,Position=self.robot_config.position[i],orientation=self.robot_config.orientation[i],prim_path=f"/World/franka_{i}"))
    
    def pick_and_place(self,pick:list,place:list):
        assert len(pick)==self.robot_config.num, "pick position num not equal to robot num"
        assert len(place)==self.robot_config.num, "place position num not equal to robot num"
        for i in range(self.robot_config.num):
            if pick[i] is not None and place[i] is not None:
                self._robot[i].pick_and_place(pick[i],place[i])

    def move_to_next_position(self,position,orientation,index):
        self._robot[index].move(position,orientation)
    
    def get_current_position(self,index):
        return self._robot[index].get_current_position()

    def open(self,flag_list):
        for i in range(self.robot_config.num):
            if flag_list[i] == True:
                self._robot[i].open()
    
    def close(self,flag_list):
        for i in range(self.robot_config.num):
            if flag_list[i] == True:
                self._robot[i].close()