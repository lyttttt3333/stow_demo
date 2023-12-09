import sys
import os
import numpy as np
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+"/env")
from omni.isaac.kit import SimulationApp
simulation_app=SimulationApp({"headless":False})
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from env.Robot.Robot import Robot
from env.config.config import Config
from env.utils.isaac_utils import add_workspace
from env.mesh.garment.garment import Garment


class DeformEnv:
    def __init__(self,config:Config):
        self.world = World(stage_units_in_meters=1.0,backend="torch",device="cuda:0")
        self.world.scene.add_default_ground_plane()
        self.config=config
        self.robot=Robot(self.world,self.config.robot_config)
        add_workspace(self.world)
        self.garment=Garment(self.world,"/home/isaac/garmentIsaac/TCLC_002_obj.usd")
        self.world.reset()
        for i in range(100):
            self.world.step(render=True)

    
    def pick_and_place_test(self):
        # self.robot.pick_and_place([np.array([0.5,0.4,0.2])],[np.array([0.5,-0.3,0.15])])
        self.robot.pick_and_place([np.array([0.3,0.3,0.15])],[np.array([0.5,-0.3,0.15])])
        for i in range(500):
            self.world.step(render=True)
        
if __name__=="__main__":
    config=Config()
    env=DeformEnv(config)
    env.pick_and_place_test()

    while simulation_app.is_running():
        env.world.step(render=True)