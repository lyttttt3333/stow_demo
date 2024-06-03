import sys
import os
import numpy as np
import torch
from env.config.PATH import *
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World, SimulationContext, PhysicsContext
from omni.isaac.core.utils.types import ArticulationAction
from env.Robot.Robot import Robot
from env.config.config import *
from env.utils.isaac_utils import add_workspace
from env.mesh.garment.garment import Garment, Rigid, AttachmentBlock, WayPoint
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
from omni.isaac.franka import Franka
import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.physx.scripts import physicsUtils, deformableUtils, particleUtils
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat,quat_to_euler_angles
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid


class trajectory_transformer():
    def __init__(self,robot_config,unit) -> None:
        self.robot_num=robot_config.num
        self.position=robot_config.position
        self.orientation=robot_config.orientation
        self.unit=unit
    
    def compute(self,position,index):
        position=position-self.position[index]+self.unit*self.position[index]
        #position=self.apply_rotation(position,index)
        return position
    
    def apply_rotation(self,position,index):
        rotation=np.array([1.,0.,0.,0.]) if self.orientation[index] is None else self.orientation[index]
        q0=rotation[0]
        q1=rotation[1]
        q2=rotation[2]
        q3=rotation[3]
        R=torch.tensor(
            [
                [1-2*q2**2-2*q3**2,2*q1*q2-2*q0*q3,2*q1*q3+2*q0*q2],
                [2*q1*q2+2*q0*q3,1-2*q1**2-2*q3**2,2*q2*q3-2*q0*q1],
                [2*q1*q3-2*q0*q2,2*q2*q3+2*q0*q1,1-2*q1**2-2*q2**2],
            ]
        )
        R=R.float()
        position=torch.mm(position.unsqueeze(0),R.transpose(1,0))
        return position.squeeze(0)
    
def keep_action_consistency(input_seq,robot_num):
    if not isinstance(input_seq[0],list):
        raise ValueError("the first ep cannot be interval")
    ep_num=len(input_seq)
    for i in range(1,ep_num):
        ep=input_seq[i]
        if isinstance(ep,EpisodeConfig):
            assert ep.contain_task == False
            length=ep.length
            ep_list=[]
            last_ep=input_seq[i-1]
            for j in range(robot_num):
                block_position=last_ep[j].task_params[-1][0]
                ep=EpisodeConfig(contain_task=True,length=length)
                ep.add_task(
                    [
                        [block_position,None],
                        [block_position,length],
                    ]
                )
                ep_list.append(ep)
            input_seq[i]=ep_list
        elif isinstance(ep,list):
            last_ep=input_seq[i-1]
            current_ep=input_seq[i]
            for j in range(robot_num):
                last_block_position=last_ep[j].task_params[-1][0]
                current_block_position=current_ep[j].task_params[0][0]
                error=last_block_position-current_block_position
                if not (error == 0).all().item():
                    raise ValueError(f"for robot {j} in ep {i} and ep {i-1}, the end and start must be consistent")
        else:
            raise TypeError
    return input_seq


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


class DynamicsModule:
    def __init__(self, world, robot_initial_position,robot_num,load_waypoint=False,real_robot=False,block_visual=False,load_scene=False):
        self.unit=0.1
        self.sim_dt=1/60
        self.default_length=30
        self.world = world
        self.stage=self.world.stage
        self.robot_num=robot_num
        self.block_num=robot_num
        self.load_scene=load_scene
        self.config=Config()
        self.real_robot=real_robot
        self.robot=Robot(self.world,self.config.robot_config,robot_initial_position,robot_num)

        self.trans=trajectory_transformer(self.config.robot_config,self.unit)

        
        self.attach=AttachmentBlock(self.world, self.stage,"/World/attach",
                                    ["/World/Garment/garment_0/mesh","/World/Garment/garment_1/mesh","/World/Garment/garment_1/mesh"],
                                    robot_initial_position,self.robot_num)
        self.move_block_list=self.attach.create(block_visual)
        self.register_default_object()

        self.pusher = DynamicCuboid(prim_path="/World/pusher", color=np.array([1.0, 1.0, 1.0]),
                name="target_plane",
                position=np.array([-0.02294,0.60462,0.89842]),
                scale=np.array([0.12, 0.2, 0.2]),
                visible=False)
        self.pusher._rigid_prim_view.disable_gravities()
        self.world.scene.add(self.pusher)

    def register_default_object(self):
        self.objects=[]
        for i in range(self.robot_num):
            attachment={
                "controller":self.move_block_list[i],
                "params":None,
                "state":None,
                "type":"block",
            }
            robot={
                "controller":self.robot._robot[i]._robot,
                "params":None,
                "state":None,
                "type":"robot",
                "gripper":"open",
                "index":i
            }
            self.objects.append(attachment)
            self.objects.append(robot)

    def register_env_object(self,type_name,init_params=None,controller=None):
        if type_name == "garment":
            assert init_params is not None, "init_params must be given"
            assert controller is not None, "con must be given"
            garment={
                "controller":controller,
                "params":init_params,
                "state":None,
                "type":"garment",
            }
            self.objects.append(garment)
        elif type_name == "rigid":
            raise ValueError("No method about rigid currently")
        else:
            raise ValueError("Type must be in rigid or garment")

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
            collectionAPI_robot.CreateIncludesRel().AddTarget(f"/World/Target")
            collectionAPI_attach = Usd.CollectionAPI.Apply(filter_attach_list[i].GetPrim(), "colliders")
            collectionAPI_attach.CreateIncludesRel().AddTarget(f"/World/attach/attach_{i}")
        collectionAPI_garment = Usd.CollectionAPI.Apply(filter_garment.GetPrim(), "colliders")
        collectionAPI_garment.CreateIncludesRel().AddTarget("/World/Garment")
        collectionAPI_rigid = Usd.CollectionAPI.Apply(filter_rigid.GetPrim(), "colliders")
        collectionAPI_rigid.CreateIncludesRel().AddTarget("/World/Rigid_0")
        collectionAPI_rigid.CreateIncludesRel().AddTarget("/World/Rigid_1")
        if True:
            collectionAPI_rigid.CreateIncludesRel().AddTarget("/World/BOOK_0")
            collectionAPI_rigid.CreateIncludesRel().AddTarget("/World/BOOK_1")
            collectionAPI_rigid.CreateIncludesRel().AddTarget("/World/BOOK_2")
            collectionAPI_rigid.CreateIncludesRel().AddTarget("/World/BOOK_3")
            collectionAPI_rigid.CreateIncludesRel().AddTarget("/World/pusher")

    def warmup(self):
        self.world.reset()
        self.robot.initialize()
        self.pre_place([True]*self.robot_num)
        ep=EpisodeConfig(contain_task=False,length=10)
        self.sub_episode(ep)

    def episode(self,ep_seq):
        self.ep_sequence=ep_seq
        for index,ep in enumerate(self.ep_sequence):
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

    def Rotation(self,q,vector):
        q0=q[0].item()
        q1=q[1].item()
        q2=q[2].item()
        q3=q[3].item()
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
        self.trajectory=[]
        for i in range(0):
            self.world.step(render=True)

        self.pre_place([True])
        position=torch.tensor([[-0.45065,0.64568,0.79554]])
        ori=np.array([-0.50336,0.51135,-0.48358,0.50129])
        while self.check_error_gripper(position,[0]):
            gripper_pose=self.move_follow_gripper(position,
                                        cmd_list=["gripper_goto_block"],
                                        pair=[0],ori=ori)

        position=torch.tensor([[-0.35065,0.64568,0.79554]])
        ori=np.array([-0.50336,0.51135,-0.48358,0.50129])
        while self.check_error_gripper(position,[0]):
            gripper_pose=self.move_follow_gripper(position,
                                        cmd_list=["gripper_goto_block"],
                                        pair=[0],
                                        ori=ori)
            self.trajectory.append(gripper_pose.unsqueeze(0))

        self.pre_pick([True,True,True])
        position=torch.tensor([[-0.35065,0.64568,0.89554]])
        ori=np.array([-0.50336,0.51135,-0.48358,0.50129])
        while self.check_error_gripper(position,[0]):
            gripper_pose=self.move_follow_gripper(position,
                                        cmd_list=["gripper_goto_block"],
                                        pair=[0],
                                        ori=ori)
            self.trajectory.append(gripper_pose.unsqueeze(0))


        position=torch.tensor([[-0.06675,0.5,0.93929]])
        ori=np.array([0.707,0.0,0.707,0.0])
        while self.check_error_gripper(position,[0]):
            gripper_pose=self.move_follow_gripper(position,
                                        cmd_list=["gripper_goto_block"],
                                        pair=[0],
                                        ori=ori)
            self.trajectory.append(gripper_pose.unsqueeze(0))
            self.push_goto(gripper_pose)

        position=torch.tensor([[-0.06675,0.4,0.93929]])
        ori=np.array([0.707,0.0,0.707,0.0])
        while self.check_error_gripper(position,[0]):
            gripper_pose=self.move_follow_gripper(position,
                                        cmd_list=["gripper_goto_block"],
                                        pair=[0],
                                        ori=ori)
            self.trajectory.append(gripper_pose.unsqueeze(0))
            self.push_goto(gripper_pose)
            
        position=torch.tensor([[-0.06675,0.3,0.93929]])
        ori=np.array([0.707,0.0,0.707,0.0])
        while self.check_error_gripper(position,[0]):
            gripper_pose=self.move_follow_gripper(position,
                                        cmd_list=["gripper_goto_block"],
                                        pair=[0],
                                        ori=ori)
            self.trajectory.append(gripper_pose.unsqueeze(0))
            self.push_goto(gripper_pose)

        self.trajectory=torch.cat(self.trajectory,dim=0).cpu().numpy()
        np.savetxt("/home/sim/stow_diffusion/traj.txt",self.trajectory)

            
        for i in range(10000000):
            self.world.step(render=True)

    def push_goto(self,target_position):
        pusher_pose,_=self.pusher._rigid_prim_view.get_world_poses()
        pusher_pose=pusher_pose.cpu()
        target_position=target_position.cpu().unsqueeze(0)
        target_position[:,0]=0
        target_position[:,-1]=0
        pusher_pose[:,0]=0
        pusher_pose[:,-1]=0
        target_position[:,1]=target_position[:,1]+0.093
        target_vel=(target_position-pusher_pose)/self.sim_dt
        orientation_ped=torch.zeros_like(target_vel)
        cmd=torch.cat([target_vel,orientation_ped],dim=-1)
        self.pusher._rigid_prim_view.set_velocities(cmd)



    def move_follow_gripper(self,next_position_mat,cmd_list,pair=[0,1,2],rest=False,ori=None):
        for i in range(self.block_num):
            cmd = cmd_list[i]
            if cmd == "block_follow_gripper":
                robot_index=pair[i]
                next_position=next_position_mat[i]
                gripper_position_next=self.trans.compute(next_position,robot_index)
                self.robot.move_to_next_position(position=gripper_position_next,orientation=None,index=robot_index)
                gripper_position,gripper_orientation=self.robot.get_current_position(index=robot_index)
                a=self.Rotation(gripper_orientation,grasp_offset)
                block_position=gripper_position.cpu()+a
                current_position,_=self.move_block_list[i].get_world_poses()
                current_position=current_position.cpu()
                target_vel=(block_position-current_position)/self.sim_dt
                orientation_ped=torch.zeros_like(target_vel)
                cmd=torch.cat([target_vel,orientation_ped],dim=-1)
                self.move_block_list[i].set_velocities(cmd)
            elif cmd == "gripper_goto_block":
                gripper_position,gripper_orientation=self.robot.get_current_position(index=i)
                robot_index=pair[i]
                next_position=next_position_mat[i]
                gripper_position_next=self.trans.compute(next_position,robot_index)
                self.robot.move_to_next_position(position=gripper_position_next,orientation=ori,index=robot_index)
            elif cmd == "block_move_single":
                next_position=next_position_mat[i]
                current_position,_=self.move_block_list[i].get_world_poses()
                current_position=current_position.cpu()
                target_vel=(next_position-current_position)/self.sim_dt
                target_vel=target_vel/(np.linalg.norm(target_vel,ord=2)+0.05)
                orientation_ped=torch.zeros_like(target_vel)
                cmd=torch.cat([target_vel,orientation_ped],dim=-1)
                self.move_block_list[i].set_velocities(cmd)
            elif cmd == "stay":
                pass
            else:
                raise ValueError
        if rest:
            for i in range(self.robot_num):
                next_position=next_position_mat[i]
                current_position,_=self.move_block_list[i].get_world_poses()
                current_position=current_position.cpu()
                target_vel=(next_position-current_position)/self.sim_dt
                target_vel=target_vel/(np.linalg.norm(target_vel,ord=2)+0.05)
                orientation_ped=torch.zeros_like(target_vel)
                cmd=torch.cat([target_vel,orientation_ped],dim=-1)
                self.move_block_list[i].set_velocities(cmd*0)
        self.world.step(render=True)
        return gripper_position

    def check_error_gripper(self,target_position, moving_index):
        import torch.nn.functional as F
        error_list=[]
        for i in range(self.block_num):
            index=moving_index[i]
            if  index!=None:
                gripper_position,gripper_orientation=self.robot.get_current_position(index=index)
                current_position=gripper_position.cpu()
                error=F.mse_loss(current_position,target_position[i]).item()
                error_list.append(error)
            else:
                pass
        max_error=max(error_list)
        if max_error >=7e-4:
            return True
        else:
            return False
        
    def check_error_block(self,target_position, moving_list):
        import torch.nn.functional as F
        error_list=[]
        for i in range(self.robot_num):
            if i in moving_list:
                position,_=self.move_block_list[i].get_world_poses()
                current_position=position[0].cpu()
                error=F.mse_loss(current_position,target_position[i]).item()
                error_list.append(error)
            else:
                pass
        max_error=max(error_list)
        if max_error >=5e-4:
            return True
        else:
            return False

    def pre_place(self,flag_list=None):
        for i in range(100):
            self.robot.open(flag_list)
            self.world.step(render=True)

    def pre_pick(self,flag_list=None):
        for i in range(100):
            self.robot.close(flag_list)
            self.world.step(render=True)

    def start(self):
        self.robot.initialize()



