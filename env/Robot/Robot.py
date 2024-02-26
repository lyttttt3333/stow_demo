from omni.isaac.core.utils.types import ArticulationAction
from env.config.config import RobotConfig
from env.Robot.Franka.franka import WrapFranka
from omni.isaac.core import World
# write pick and place primitive
# write movep function
# write fling function
# write hang function


class Robot:
    def __init__(self,world:World,robot_config:RobotConfig):
        self.robot_config=robot_config
        self.world=world
        self._robot=[]
        self.load_robot()
    
    
    def load_robot(self):
        if len(self.robot_config.position)!=self.robot_config.num:
            assert False, "robot position num not equal to robot num"
        if self.robot_config.type=="franka":
            for i in range(self.robot_config.num):
                self._robot.append(WrapFranka(self.world,self.robot_config.position[i]))
    
    def pick_and_place(self,pick:list,place:list):
        assert len(pick)==self.robot_config.num, "pick position num not equal to robot num"
        assert len(place)==self.robot_config.num, "place position num not equal to robot num"
        for i in range(self.robot_config.num):
            if pick[i] is not None and place[i] is not None:
                self._robot[i].pick_and_place(pick[i],place[i])

    def movep(self,end_pos):
        assert len(end_pos)==self.robot_config.num, "pick position num not equal to robot num"
        for i in range(self.robot_config.num):
            if end_pos[i] is not None:
                self._robot[i].movep(end_pos[i],speed=2)
    def open(self):
        for i in range(self.robot_config.num):
            self._robot[i].open()
    
    def close(self):
        for i in range(self.robot_config.num):
            self._robot[i].close()