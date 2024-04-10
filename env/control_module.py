import sys
import os
import numpy as np
import torch
from PATH import *
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World, SimulationContext, PhysicsContext
from omni.isaac.core.utils.types import ArticulationAction
from env.Robot.Robot import Robot
from env.config.config import Config
from env.utils.isaac_utils import add_workspace
from env.mesh.garment.garment import Garment, ParticleSamplerDemo,Rigid, Rigid2, ParticleCloth, AttachmentBlock, WayPoint, ParticleCloth
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
from omni.isaac.franka import Franka
import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.physx.scripts import physicsUtils, deformableUtils, particleUtils
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat,quat_to_euler_angles
from omni.isaac.sensor import Camera

class trajectory_transformer():
    def __init__(self,robot_config) -> None:
        self.robot_num=robot_config.num
        self.position=robot_config.position
        self.orientation=robot_config.orientation
    
    def compute(self,position,index):
        position=position-self.position[index]
        position=self.apply_rotation(position,index)
        return position
    
    def apply_rotation(self,position,index):
        rotation=torch.tensor([1.,0.,0.,0.]) if self.orientation[index] is None else self.orientation[index]
        q0=rotation[0].item()
        q1=-rotation[1].item()
        q2=-rotation[2].item()
        q3=-rotation[3].item()
        R=torch.tensor(
            [
                [1-2*q2**2-2*q3**2,2*q1*q2-2*q0*q3,2*q1*q3+2*q0*q2],
                [2*q1*q2+2*q0*q3,1-2*q1**2-2*q3**2,2*q2*q3-2*q0*q1],
                [2*q1*q3-2*q0*q2,2*q2*q3+2*q0*q1,1-2*q1**2-2*q2**2],
            ]
        )
        position=torch.mm(position.unsqueeze(0),R.transpose(1,0))
        return position.squeeze(0)


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

class DeformEnvOld:
    def __init__(self,robot_initial_position,robot_num,load_scene=False,load_waypoint=False,real_robot=False,block_visual=True):
        self.config=Config()
    
        self.unit=0.1
        self.world = World(stage_units_in_meters=self.unit,backend="torch",device="cuda:0")
        self.stage=self.world.stage
        self.robot_num=robot_num
        self.load_scene=load_scene
        self.real_robot=real_robot
        self.robot=Robot(self.world,self.config.robot_config,robot_initial_position,robot_num)
        Scene=self.world.get_physics_context() 
        self.scene=Scene._physics_scene
        self.default_length=36
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        self.scene.CreateGravityMagnitudeAttr().Set(0.98)

        if load_scene:        
            self.load_room(LIVING_ROOM)
        else:
            self.rigid=Rigid("/World/Rigid",BED)

        if load_waypoint:
            self.way_point=WayPoint(root_path="/Way_point", section_num=3)

        self.trans=trajectory_transformer(self.config.robot_config)

        garment_params={
            "path":CLOTH_TO_HANG,
            "position":np.array([-0.65481,-1.27712,0.54132]),
            "orientation":np.array([0.47366,-0.32437,-0.46264,-0.67557]),
            "scale":np.array([0.0075, 0.0075, 0.0075]),
        }

        self.garment=Garment(self.stage,self.scene,garment_params)
        self.world.scene.add(self.garment.garment_mesh)
        self.info=self.garment.garment_mesh._cloth_prim_view 

        self.create_collsion_group()
        self.set_camera()
        

        self.device="cuda:0"
        self.sim_dt=1/60
        
        self.attach=AttachmentBlock(self.world, self.stage,"/World/attach","/World/Garment/garment/mesh",robot_initial_position_list,self.robot_num)
        self.attach.create(block_visual)
        move_block_list=self.attach.block_prim_list
        self.move_block_controller_list=[move_block._rigid_prim_view for move_block in move_block_list]
        for move_block in self.move_block_controller_list:
            move_block.disable_gravities()

        self.objects=[]
        init_params={
            "q":garment_params["orientation"],
            "r":garment_params["position"],
            "scale":1/garment_params["scale"][0],
        }
        object1={
            "controller":self.garment.garment_mesh._cloth_prim_view,
            "params":init_params,
            "state":None,
        }
        robot1={
            "controller":self.robot._robot[0]._robot,
            "params":None,
            "state":None,
        }
        self.objects.append(object1)
        self.objects.append(robot1)
        for i in range(self.robot_num):
            attachment={
                "controller":move_block_list[i],
                "params":None,
                "state":None,
            }
            self.objects.append(attachment)

    def load_room(self,env_path):
        from omni.isaac.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
        from omni.isaac.core.utils.string import find_unique_string_name
        from omni.isaac.core.utils.prims import is_prim_path_valid
        self.room_prim_path=find_unique_string_name("/Room",is_unique_fn=lambda x: not is_prim_path_valid(x))
        add_reference_to_stage(env_path,self.room_prim_path)
        self.room_prim=XFormPrim(self.room_prim_path,name="Room",scale=[0.8,0.8,0.8],position=[0.7,0.5,0],orientation=euler_angles_to_quat([0,0,-np.pi]))

    def set_camera(self):
        rotation=np.array([0.37207,0.20084,0.43454,0.79524])
        self.camera = Camera(
            prim_path="/Camera",
            position=np.array([9.0, 7.0, 7.5]),
            frequency=20,
            resolution=(512, 512),
            orientation=rotation
        )
        self.camera.initialize()

    def test_camera(self):
        self.world.reset()
        for i in range(100):
            self.world.step(render=True)
            photo=self.camera.get_rgb()
            print(photo)


    def create_collsion_group(self):
        robot_group_list=[]
        robot_group_path_list=[]
        attach_group_list=[]
        attach_group_path_list=[]
        for i in range(self.robot_num):
            robot_group_path=f"/World/robot_group_{i}"
            robot_group_path_list.append(robot_group_path)
            robot_group = UsdPhysics.CollisionGroup.Define(self.stage, robot_group_path)
            robot_group_list.append(robot_group)
            attach_group_path=f"/World/attach_group_{i}"
            attach_group_path_list.append(attach_group_path)
            attach_group = UsdPhysics.CollisionGroup.Define(self.stage, attach_group_path)
            attach_group_list.append(attach_group)
        garment_group_path="/World/Garment_group"
        garment_group = UsdPhysics.CollisionGroup.Define(self.stage, garment_group_path)
        rigid_group_path="/World/Rigid_group"
        rigid_group = UsdPhysics.CollisionGroup.Define(self.stage, rigid_group_path)

        filter_garment = garment_group.CreateFilteredGroupsRel()
        filter_rigid = rigid_group.CreateFilteredGroupsRel()
        filter_robot_list=[]
        filter_attach_list=[]
        for i in range(self.robot_num):
            filter_robot = robot_group_list[i].CreateFilteredGroupsRel()
            filter_robot_list.append(filter_robot)
            filter_attach = attach_group_list[i].CreateFilteredGroupsRel()
            filter_attach_list.append(filter_attach)

        for i in range(self.robot_num):
            filter_robot_list[i].AddTarget(garment_group_path)
            filter_robot_list[i].AddTarget(rigid_group_path)
            for j in range(self.robot_num):
                filter_robot_list[i].AddTarget(attach_group_path_list[j])
            filter_attach_list[i].AddTarget(rigid_group_path)
            filter_attach_list[i].AddTarget(garment_group_path)
            for j in range(i+1,self.robot_num):
                filter_robot_list[i].AddTarget(robot_group_path_list[j])
                filter_attach_list[i].AddTarget(attach_group_path_list[j])

        for i in range(self.robot_num):
            collectionAPI_robot = Usd.CollectionAPI.Apply(filter_robot_list[i].GetPrim(), "colliders")
            collectionAPI_robot.CreateIncludesRel().AddTarget(f"/World/franka_{i}")
            collectionAPI_attach = Usd.CollectionAPI.Apply(filter_attach_list[i].GetPrim(), "colliders")
            collectionAPI_attach.CreateIncludesRel().AddTarget(f"/World/attach/attach_{i}")
        collectionAPI_garment = Usd.CollectionAPI.Apply(filter_garment.GetPrim(), "colliders")
        collectionAPI_garment.CreateIncludesRel().AddTarget("/World/Garment")
        collectionAPI_rigid = Usd.CollectionAPI.Apply(filter_rigid.GetPrim(), "colliders")
        if self.load_scene:
            collectionAPI_rigid.CreateIncludesRel().AddTarget("/Room")
        else:
            collectionAPI_rigid.CreateIncludesRel().AddTarget("/World/Rigid")

    def warmup(self):
        ep=EpisodeConfig(contain_task=False,length=60)
        self.sub_episode(ep)

    def episode(self,ep_seq):
        self.ep_sequence=ep_seq
        for ep in self.ep_sequence:
            self.sub_episode(ep)
        self.world.reset()
        

    def sub_episode(self,sub_episode_config):
        if isinstance(sub_episode_config,list):
            contain_task=True
            assert len(sub_episode_config) == self.robot_num
        else:
            if not isinstance(sub_episode_config, EpisodeConfig):
                raise ValueError("Unknow Type")
            contain_task=sub_episode_config.contain_task
            if contain_task is True:
                raise ValueError("Task-contained episode must be list type")
        if not contain_task:
            self.simulation(params=sub_episode_config)
        else:
            self.action(params_list=sub_episode_config)


    def log_final_scene(self):
        for i in range(len(self.objects)):
            object=self.objects[i]
            if i == 0:
                object["state"]=object["controller"].get_world_positions()
                print(object["state"].shape)
            if i == 1:     
                object["state"]=object["controller"].get_joint_positions()
            if i == 2:
                position,_=object["controller"].get_world_pose()
                object["state"]=position

    def reload_final_scene(self,phase1:bool,phase2:bool):
        for i in range(len(self.objects)):
            object=self.objects[i]
            if i == 0 and phase1:
                params=object["params"]
                q=params["q"]
                r=params["r"]
                scale=params["scale"]
                next_state=self.trans_points(q,r,scale,object["state"])
                object["controller"].set_world_positions(next_state.unsqueeze(0),force_set=True)
            if i == 1 and phase2:
                next_state=object["state"]
                if next_state is not None:
                    object["controller"].set_joint_positions(next_state)
            if i == 2:
                next_state=object["state"]
                object["controller"].set_world_pose(next_state)

    def trans_points(self,q,r,scale,points):
        q0=q[0]
        q1=-q[1]
        q2=-q[2]
        q3=-q[3]
        R=torch.tensor(
            [
                [1-2*q2**2-2*q3**2,2*q1*q2-2*q0*q3,2*q1*q3+2*q0*q2],
                [2*q1*q2+2*q0*q3,1-2*q1**2-2*q3**2,2*q2*q3-2*q0*q1],
                [2*q1*q3-2*q0*q2,2*q2*q3+2*q0*q1,1-2*q1**2-2*q2**2],
            ]
        )
        points=points[0]
        points=points-r
        points=points*scale
        points=torch.mm(points,R.transpose(1,0))
        return points

    def Rotation(self,q,vector):
        q0=q[0].item()
        q1=-q[1].item()
        q2=-q[2].item()
        q3=-q[3].item()
        R=torch.tensor(
            [
                [1-2*q2**2-2*q3**2,2*q1*q2-2*q0*q3,2*q1*q3+2*q0*q2],
                [2*q1*q2+2*q0*q3,1-2*q1**2-2*q3**2,2*q2*q3-2*q0*q1],
                [2*q1*q3-2*q0*q2,2*q2*q3+2*q0*q1,1-2*q1**2-2*q2**2],
            ]
        )
        vector=torch.mm(vector.unsqueeze(0),R.transpose(1,0))
        return vector.squeeze(0)
        

    def simulation(self,params:EpisodeConfig):
        length=params.length
        self.start()
        for i in range(length):
            self.world.step(render=True)
        self.pause()

    def action(self,params_list:list):
        trajectory_list=[]
        pick_list=[]
        place_list=[]
        for i in range(len(params_list)):
            params=params_list[i].task_params
            pick_list.append(params_list[i].pick)
            place_list.append(params_list[i].place)
            way_point=[sub_list[0].unsqueeze(0) for sub_list in params]
            way_point=torch.cat(way_point,dim=0)
            trajectory_list.append(way_point)
            length=0
            for j in range(len(params)):
                if j == 0:
                    pass
                else:
                    length+=params[j][-1]
            if i == 0:
                last_length=length
            else:
                assert last_length == length
        length=self.default_length if not self.real_robot else length
        trajectory_mat=self.decouple_trajectory(trajectory_list=trajectory_list,length=length)
        pick_position=trajectory_mat[0]
        self.pick(pick_position,flag_list=pick_list)
        self.start()
        for i in range(6):
            print(i)
            self.world.step(render=True)
        for i in range(length):
            next_position=trajectory_mat[i]
            self.move_follow_gripper(next_position)
        if self.real_robot:
            while self.check_error_gripper(next_position):
                self.move_follow_gripper(next_position)
        else:
            self.move_follow_gripper(next_position)
        self.pause()
        self.place(place_list)

    def check_error_block(self,target_position):
        import torch.nn.functional as F
        error_list=[]
        for i in range(self.robot_num):
            position,_=self.move_block_controller_list[i].get_world_poses()
            current_position=position[0].cpu()
            error=F.mse_loss(current_position,target_position[3*i:3*i+3]).item()
            error_list.append(error)
        max_error=max(error_list)
        if max_error >=5e-6:
            return True
        else:
            return False


    def check_error_gripper(self,target_position):
        import torch.nn.functional as F
        error_list=[]
        for i in range(self.robot_num):
            if True:
                gripper_position,gripper_orientation=self.robot.get_current_position(index=i)
                current_position=gripper_position.cpu()
                print(gripper_orientation)
                error=F.mse_loss(current_position,target_position[3*i:3*i+3]).item()
                error_list.append(error)
        max_error=max(error_list)
        print(max_error)
        if max_error >=7e-3:
            return True
        else:
            return False
    
    def decouple_trajectory(self,trajectory_list:list, length:int):
        import torch.nn.functional as F
        for i in range(len(trajectory_list)):
            tra=trajectory_list[i].unsqueeze(0).unsqueeze(0)
            trajectory_list[i] = F.interpolate(tra,size=(length,3),mode="bilinear",align_corners=True).squeeze(0).squeeze(0)
        trajectory_mat=torch.cat(trajectory_list,dim=-1)
        return trajectory_mat

    def pick(self, grasp_position, grasp_orientation=None,flag_list=None):
        if grasp_orientation is None:
            grasp_orientation=np.array([0.,0.,1.,0.])
        grasp_position=grasp_position.reshape(-1,3)
        self.attach.set_position(grasp_point_list=grasp_position)
        self.attach.attach(flag_list)
    
    def place(self,flag_list=None):
        self.attach.detach(flag_list)

    def pre_place(self,flag_list=None):
        pass
    def pre_pick(self,flag_list=None):
        for i in range(100):
            self.world.step(render=True)
    
    def move_follow_gripper(self,next_position_mat):
        for i in range(self.robot_num):
            next_position=next_position_mat[i*3:i*3+3]
            if self.real_robot:
                gripper_position_next=self.trans.compute(next_position,i)
                #gripper_position_next=next_position
                self.robot.move_to_next_position(position=gripper_position_next,orientation=None,index=i)
                gripper_position,gripper_orientation=self.robot.get_current_position(index=i)
                a=self.Rotation(gripper_orientation,grasp_offset)
                print(gripper_position,gripper_orientation)
                print(a)
                block_position=gripper_position.cpu()+a
                current_position,_=self.move_block_controller_list[i].get_world_poses()
                current_position=current_position.cpu()
                target_vel=(block_position-current_position)/self.sim_dt
                orientation_ped=torch.zeros_like(target_vel)
                cmd=torch.cat([target_vel,orientation_ped],dim=-1)
                self.move_block_controller_list[i].set_velocities(cmd)
            else:
                current_position,_=self.move_block_controller_list[i].get_world_poses()
                current_position=current_position.cpu()
                target_vel=(next_position-current_position)/self.sim_dt
                orientation_ped=torch.zeros_like(target_vel)
                cmd=torch.cat([target_vel,orientation_ped],dim=-1)
                self.move_block_controller_list[i].set_velocities(cmd)
        self.world.step(render=True)
        
    def pause(self):
        self.log_final_scene()
        self.world.stop()
        self.reload_final_scene(phase1=True,phase2=False)
        print("######################################")
        print("###### align render and physics ######")
        print("######################################")
        

    def start(self):
        self.world.reset()
        self.robot.initialize()
        self.reload_final_scene(phase1=False,phase2=True)


    def end(self):
        for i in range(10000):
            self.world.step(render=True)


