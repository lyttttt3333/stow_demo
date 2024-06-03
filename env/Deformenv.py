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
from omni.isaac.core.materials.physics_material import  PhysicsMaterial
from omni.isaac.quadruped.controllers import A1QPController
from env.Robot.Robot import Robot
from env.config.config import *
from env.config.PATH import *
from env.utils.isaac_utils import add_workspace
from env.Robot.control_module import *
from env.mesh.garment.garment import Garment, Rigid, WayPoint, Human, Deformable, Collider

from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
from omni.isaac.franka import Franka
import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.physx.scripts import physicsUtils, deformableUtils, particleUtils
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat,quat_to_euler_angles
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid



class BaseEnv:
    def __init__(self,load_scene=False,load_waypoint=False,real_robot=False,block_visual=True,scene_path=BED_ROOM,rigid_path=BED):
        self.config=Config()
        self.unit=0.1
        self.world = World(stage_units_in_meters=self.unit,backend="torch",device="cuda:1")
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
        
        if load_waypoint:
            self.way_point=WayPoint(root_path="/Way_point", section_num=3)

        for idx, config in enumerate(config_list):
            self.rigid=Collider(f"/World/Rigid_{idx}",config,f"collider_{idx}")

        for idx, config in enumerate(book_list):
            self.rigid=Rigid(f"/World/BOOK_{idx}",config,f"book_{idx}")

        self.dynamcis=DynamicsModule(self.world,robot_initial_position=self.config.robot_config.position,robot_num=self.robot_num,real_robot=real_robot,load_scene=load_scene)

        self.dynamcis.create_collsion_group()

        prim = DynamicCuboid(prim_path="/World/Target", color=np.array([1.0, 1.0, 1.0]),
                name="target",
                position=np.array([-0.32862,0.57002,0.79554]),
                scale=np.array([0.12984, 0.18875, 0.01501]),
                mass=None,
                visible=True)
        prim._rigid_prim_view.disable_gravities()
        self.physics_material=PhysicsMaterial(prim_path="/World/Target/mate", dynamic_friction=1.99,static_friction=1.99)
        #prim.apply_physics_material(self.physics_material)

    def warmup(self):
        self.dynamcis.warmup()

    def episode(self,ep_seq):
        self.dynamcis.episode(ep_seq)

    def test(self):
        while simulation_app.is_running():
            simulation_app.update()


if __name__=="__main__":

    env=BaseEnv(load_scene=True,load_waypoint=False,real_robot=True,scene_path=LIVING_ROOM)
    env.test()
    env.warmup()
    env.episode(ep_seq=ep_sequence)
    for i in range(10000):
        simulation_app.update()


