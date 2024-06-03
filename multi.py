from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
import torch.nn as nn

import omni
import numpy as np
import os
import torch
import yaml
import carb
import sys
import matplotlib.pyplot as plt
from PIL import Image
from config import *
import random
import json



from omni.isaac.core import World, PhysicsContext, SimulationContext
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
from omni.isaac.core.prims import GeometryPrim,RigidPrim,XFormPrim
from omni.isaac.core.objects import DynamicCuboid,VisualCuboid,FixedCuboid
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import GeometryPrim,RigidPrim,XFormPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim, create_prim
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage, open_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.cloner import GridCloner
from pxr import UsdGeom, Gf
from omni.isaac.core.utils.semantics import add_update_semantics, get_semantics
from omni.isaac.franka.tasks import PickPlace
from omni.isaac.manipulators.grippers import ParallelGripper, SurfaceGripper
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.sensor import Camera
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim, create_prim, get_prim_at_path,get_all_matching_child_prims,get_first_matching_child_prim,get_prim_type_name,get_prim_path

DEVICE="cuda:1"

def qua_mult(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return torch.tensor(
            [
                -x2 * x1 - y2 * y1 - z2 * z1 + w2 * w1,
                x2 * w1 + y2 * z1 - z2 * y1 + w2 * x1,
                -x2 * z1 + y2 * w1 + z2 * x1 + w2 * y1,
                x2 * y1 - y2 * x1 + z2 * w1 + w2 * z1,
            ]
            ,device=DEVICE
        ).cpu()

def set_init_pose(my_world,state_list,envs_num):
    for i in range(envs_num):
        x_index=i%ENV_line
        y_index=i//ENV_line
        step=torch.tensor([-2*y_index+0.1, -2*x_index-0.36, 0.]).to(DEVICE)
        for state in state_list:
            name="xform"+"_"+f"{i}"+"_"+state[0].split("_")[-1]
            globals()[name].set_world_pose(position=torch.tensor(state[1]).to(DEVICE)+step,orientation=torch.tensor(state[2]).to(DEVICE))
    for i in range(20):
        my_world.step(render=True)

def add_robot(my_world,asset_path,index,axis_list):
    add_reference_to_stage(usd_path=asset_path, prim_path=f"/World/Franka{index}")

    gripper = ParallelGripper(
        end_effector_prim_path=f"/World/Franka{index}/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=torch.tensor([0.08, 0.08]),
        joint_closed_positions=torch.tensor([0.005, 0.005]),
    )
    my_franka = my_world.scene.add(
        SingleManipulator(
            prim_path=f"/World/Franka{index}", name=f"my_franka{index}", end_effector_prim_name="panda_rightfinger", gripper=gripper
        )
    )
    my_franka.set_enabled_self_collisions(False)
    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
    gripper.enable_rigid_body_physics()
    my_controller = PickPlaceController(
        name="pick_place_controller", gripper=my_franka.gripper, robot_articulation=my_franka, end_effector_initial_height=0.25
    )
    articulation_controller = my_franka.get_articulation_controller()
    init_pos=np.array([0., 0., 0.])
    x_index=index%ENV_line
    y_index=index//ENV_line
    robotframe=XFormPrim(prim_path=f"/World/Franka{index}",name=f"robotframe{index}")
    robotframe.set_world_pose(position=[-2*y_index+init_pos[0],-2*x_index+init_pos[1],init_pos[2]],)
                              #orientation=[0.92388,0.0,0.0,0.38268])
    
    return my_franka,my_controller,articulation_controller

import argparse
import time
start_time= time.time()
argparser = argparse.ArgumentParser()
argparser.add_argument("--indice", type=int, default=62020)
argparser.add_argument("--start", type=int, default=0)
argparser.add_argument("--target_name", type=str, default=0)
args= argparser.parse_args()

my_world= World(stage_units_in_meters=1.0, backend="torch",device=DEVICE)
my_world.scene.add_default_ground_plane()
my_world.set_simulation_dt(physics_dt=1/20,rendering_dt=1/20)
phy=my_world.get_physics_context()
phy.set_gravity(-9.81)

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

rigid_label_list=[]

ENV_line=2
env_num=ENV_line**2
start=args.start

asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
robot_list=[]
controller_list=[]
articulation_controller_list=[]
orientation=[]

succ_list=[False]*env_num
axis_list=[]
result_list=[[False,False,False,False]]*env_num
for env in range(env_num):
    my_franka,my_controller,articulation_controller=add_robot(my_world,asset_path,env,axis_list)
    robot_list.append(my_franka)
    controller_list.append(my_controller)
    articulation_controller_list.append(articulation_controller)

if True:
    while simulation_app.is_running():
        simulation_app.update()

for iter in range(4):
    print("begin")
    my_world.reset()
    for robot_index in range(len(robot_list)):
        controller_list[robot_index].reset()
    print("###################### new direction ########################")
    for i in range(8000):
        my_world.step(render=True)
        for robot_index in range(len(robot_list)):
            #step=np.array([0.5+0.1*random.random(),0.0+0.1*random.random(),0.0])
            target=np.array([0.5,0.0,0.5])
            actions = controller_list[robot_index].forward(
                picking_position=target,
                placing_position=np.array([0., 0.6, 1.5]),
                current_joint_positions=robot_list[robot_index].get_joint_positions(),
                #end_effector_offset=np.array([0.0, 0.0, -0.028]),
            ) 
            articulation_controller_list[robot_index].apply_action(actions)
simulation_app.close()


#./python.sh /home/sim/.local/share/ov/pkg/isaac_sim-2023.1.0/standalone_examples/api/omni.isaac.kit/gen_dataset/m-grasp-gpu.py