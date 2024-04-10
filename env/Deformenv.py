import sys
import os
import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+"/env")
from omni.isaac.kit import SimulationApp
simulation_app=SimulationApp({"headless":False})
from omni.isaac.core import World, SimulationContext, PhysicsContext
from omni.isaac.core.utils.types import ArticulationAction

from env.Robot.Robot import Robot
from env.config.config import *
from env.config.PATH import *
from env.utils.isaac_utils import add_workspace
from env.Robot.control_module import *
from env.mesh.garment.garment import Garment, ParticleSamplerDemo,Rigid, Rigid2, ParticleCloth, AttachmentBlock, WayPoint, ParticleCloth

from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
from omni.isaac.franka import Franka
import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.physx.scripts import physicsUtils, deformableUtils, particleUtils
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat,quat_to_euler_angles
from omni.isaac.sensor import Camera



class BaseEnv:
    def __init__(self,load_scene=False,load_waypoint=False,real_robot=False,block_visual=True,scene_path=LIVING_ROOM,rigid_path=BED):
        self.config=Config()
        self.unit=0.1
        self.world = World(stage_units_in_meters=self.unit,backend="torch",device="cuda:0")
        self.stage=self.world.stage
        self.robot_num=self.config.robot_config.num
        self.load_scene=load_scene
        self.real_robot=real_robot
        Scene=self.world.get_physics_context() 
        self.scene=Scene._physics_scene
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        self.scene.CreateGravityMagnitudeAttr().Set(0.98)
        self.device="cuda:0"
        self.sim_dt=1/60
        
        if load_scene:    
            self.load_room(scene_path)
        else:
            self.rigid=Rigid("/World/Rigid",rigid_path)

        if load_waypoint:
            self.way_point=WayPoint(root_path="/Way_point", section_num=3)

        self.dynamcis=DynamicsModule(self.world,robot_initial_position=self.config.robot_config.position,robot_num=self.robot_num,real_robot=real_robot,load_scene=load_scene)
        
        self.garment=Garment(self.world, self.stage,self.scene,GarmentConfig)
        init_params={
            "q":GarmentConfig["orientation"],
            "r":GarmentConfig["position"],
            "scale":1/GarmentConfig["scale"][0],
        }
        self.dynamcis.register_env_object("garment",init_params,self.garment.garment_mesh._cloth_prim_view)
        # all objects (garment,robot,rigid...) which will move during manipulation need adding here
        # non-moving objects ground\table... dont need adding
        # moving block and robots have already been added
        # TODO add rigid body

        self.dynamcis.create_collsion_group()

        
    def load_room(self,env_path):
        from omni.isaac.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
        from omni.isaac.core.utils.string import find_unique_string_name
        from omni.isaac.core.utils.prims import is_prim_path_valid
        self.room_prim_path=find_unique_string_name("/Room",is_unique_fn=lambda x: not is_prim_path_valid(x))
        add_reference_to_stage(env_path,self.room_prim_path)
        self.room_prim=XFormPrim(self.room_prim_path,name="Room",scale=[0.8,0.8,0.8],position=[0.7,0.5,0],orientation=euler_angles_to_quat([0,0,-np.pi]))

    def warmup(self):
        self.dynamcis.warmup()

    def episode(self,ep_seq):
        self.dynamcis.episode(ep_seq)

    def test(self):
        while simulation_app.is_running():
            simulation_app.update()


if __name__=="__main__":

    #ep_sequence=keep_action_consistency(ep_sequence,)
    env=BaseEnv(load_scene=True,load_waypoint=True,real_robot=True)
    env.test()
    env.warmup()
    env.episode(ep_seq=ep_sequence)


