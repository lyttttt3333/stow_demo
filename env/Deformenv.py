import sys
import os
import numpy as np
import torch
from PATH import *
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+"/env")
from omni.isaac.kit import SimulationApp
simulation_app=SimulationApp({"headless":False})
from omni.isaac.core import World, SimulationContext, PhysicsContext
from omni.isaac.core.utils.types import ArticulationAction
from env.Robot.Robot import Robot
from env.config.config import *
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
        print(input_seq)
    return input_seq
    


class DynamicsModule:
    def __init__(self, world, robot_initial_position,robot_num,load_waypoint=False,real_robot=False,block_visual=True,load_scene=False):
        self.unit=0.1
        self.sim_dt=1/60
        self.default_length=30
        self.world = world
        self.stage=self.world.stage
        self.robot_num=robot_num
        self.load_scene=load_scene
        self.config=Config()
        self.real_robot=real_robot
        self.robot=Robot(self.world,self.config.robot_config,robot_initial_position,robot_num)

        self.trans=trajectory_transformer(self.config.robot_config)

        
        self.attach=AttachmentBlock(self.world, self.stage,"/World/attach","/World/Garment/garment/mesh",robot_initial_position_list,self.robot_num)
        self.move_block_list=self.attach.create(block_visual)
        self.register_default_object()

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
            object_type=object["type"]
            if object_type == "garment":
                object["state"]=object["controller"].get_world_positions()
            if object_type == "robot":     
                object["state"]=object["controller"].get_joint_positions()
            if object_type == "block":
                position,_=object["controller"].get_world_poses()
                object["state"]=position

    def reload_final_scene(self,phase1:bool,phase2:bool):
        for i in range(len(self.objects)):
            object=self.objects[i]
            object_type=object["type"]
            if object_type == "garment" and phase1:
                params=object["params"]
                q=params["q"]
                r=params["r"]
                scale=params["scale"]
                next_state=self.trans_points(q,r,scale,object["state"])
                object["controller"].set_world_positions(next_state.unsqueeze(0),force_set=True)
            if object_type == "robot" and phase2:
                next_state=object["state"]
                if next_state is not None:
                    object["controller"].set_joint_positions(next_state)
            if object_type == "block":
                next_state=object["state"]
                object["controller"].set_world_poses(next_state)

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
                self.robot.move_to_next_position(position=gripper_position_next,orientation=None,index=i)
                gripper_position,gripper_orientation=self.robot.get_current_position(index=i)
                a=self.Rotation(gripper_orientation,grasp_offset)
                print(gripper_position,gripper_orientation)
                print(a)
                block_position=gripper_position.cpu()+a
                current_position,_=self.move_block_list[i].get_world_poses()
                current_position=current_position.cpu()
                target_vel=(block_position-current_position)/self.sim_dt
                orientation_ped=torch.zeros_like(target_vel)
                cmd=torch.cat([target_vel,orientation_ped],dim=-1)
                self.move_block_list[i].set_velocities(cmd)
            else:
                current_position,_=self.move_block_list[i].get_world_poses()
                current_position=current_position.cpu()
                target_vel=(next_position-current_position)/self.sim_dt
                orientation_ped=torch.zeros_like(target_vel)
                cmd=torch.cat([target_vel,orientation_ped],dim=-1)
                self.move_block_list[i].set_velocities(cmd)
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


class BaseEnv:
    def __init__(self,robot_initial_position,robot_num,load_scene=False,load_waypoint=False,real_robot=False,block_visual=True,scene_path=LIVING_ROOM,rigid_path=BED):
        self.unit=0.1
        self.world = World(stage_units_in_meters=self.unit,backend="torch",device="cuda:0")
        self.stage=self.world.stage
        self.robot_num=robot_num
        self.config=Config()
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

        self.dynamcis=DynamicsModule(self.world,robot_initial_position=robot_initial_position,robot_num=robot_num,real_robot=real_robot,load_scene=load_scene)
        
        self.garment=Garment(self.world, self.stage,self.scene,GarmentConfig)
        init_params={
            "q":GarmentConfig["orientation"],
            "r":GarmentConfig["position"],
            "scale":1/GarmentConfig["scale"][0],
        }
        self.dynamcis.register_env_object("garment",init_params,self.garment.garment_mesh._cloth_prim_view)

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

    ep_sequence=keep_action_consistency(ep_sequence,robot_num)
    env=BaseEnv(robot_initial_position_list,robot_num,load_scene=True,load_waypoint=True,real_robot=True)
    env.test()
    env.warmup()
    env.episode(ep_seq=ep_sequence)


