from env.utils.transforms import quat_diff_rad
import numpy as np
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka
from omni.isaac.core.utils.prims import is_prim_path_valid, get_prim_at_path
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import torch
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
from omni.isaac.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem


class WrapFranka:
    def __init__(self,world:World,Position=torch.tensor,orientation=None,prim_path:str=None,robot_name:str=None,):
        self.world=world
        if prim_path is None:
            self._franka_prim_path = find_unique_string_name(
                initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
        else:
            self._franka_prim_path=prim_path

        if robot_name is None:
            self._franka_robot_name = find_unique_string_name(
                initial_name="my_franka", is_unique_fn=lambda x: not self.world.scene.object_exists(x)
            )
        else:
            self._franka_robot_name=robot_name

        self.init_position=Position
        self._robot=Franka(prim_path=self._franka_prim_path, name=self._franka_robot_name,position=Position,orientation=orientation)
        self.world.scene.add(self._robot)
        self._articulation_controller=self._robot.get_articulation_controller()
        self._controller=RMPFlowController(name="rmpflow_controller",robot_articulation=self._robot,physics_dt=1/240.0)
        self._kinematic_solover=KinematicsSolver(self._robot)
        self._pick_place_controller=PickPlaceController(name="pick_place_controller",robot_articulation=self._robot,gripper=self._robot.gripper)



    def initialize(self):
        self._controller.reset()
        #self._robot.initialize()


    def get_cur_ee_pos(self):
        ee_pos, R = self._controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
        return ee_pos, R

    def get_current_position(self):
        position,orientation=self._robot.gripper.get_world_pose()
        return position,orientation
    
    def move(self,position,orientation=None):
        #position,orientation=self.input_filter(position,orientation)
        position=position.cpu().numpy()/0.1
        orientation=np.array([0.,1.,0.,0.])
        #orientation=np.array([0.,0.70711,-0.70711,0.])
        actions = self._controller.forward(
            target_end_effector_position=position,
            #target_end_effector_orientation=orientation
            )
        #self._robot.set_joint_positions(joint)

        action_info=self._articulation_controller.apply_action(actions,True)
        #self._articulation_controller.apply_discrete_action(action_info)

    def input_filter(self,position,orientation):
        if orientation is None:
            pass
        if isinstance(position,torch.tensor):
            pass

        


    def change_rigid_state(self,enable:bool):
        path="/World/Franka/panda_rightfinger/geometry/panda_rightfinger"
        print(path)
        prim=get_prim_at_path(path)
        #UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(enable)
        if False:
            if enable:
                self._robot.end_effector.enable_rigid_body_physics()
            else:
                self._robot.end_effector.disable_rigid_body_physics()

    
    def open(self):
        self._robot.gripper.open()
    
    def close(self):
        self._robot.gripper.close()

    @staticmethod
    def interpolate(start_loc, end_loc, speed):
        start_loc = np.array(start_loc)
        end_loc = np.array(end_loc)
        dist = np.linalg.norm(end_loc - start_loc)
        chunks = dist // speed
        if chunks==0:
            chunks=1
        return start_loc + np.outer(np.arange(chunks+1,dtype=float), (end_loc - start_loc) / chunks)
    
    def position_reached(self, target,thres=0.01):
        if target is None:
            return True
        
        ee_pos, R = self._controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
        pos_diff = np.linalg.norm(ee_pos- target)
        if pos_diff < thres:
            return True
        else:
            return False 
    def rotation_reached(self, target):
        if target is None:
            return True
        
        ee_pos, R = self._controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
        angle_diff = quat_diff_rad(R, target)[0]
        # print(f'angle diff: {angle_diff}')
        if angle_diff < 0.1:
            return True
        
    def movep(self,end_loc,speed,max_step=200):
        self.world.step(render=True)
        start_loc=self.get_cur_ee_pos()[0]
        path=self.interpolate(start_loc,end_loc,speed)
        for i in range(len(path)):
            step_num=0
            while not self.position_reached(path[i]):
                step_num+=1
                if step_num>max_step:
                    break
                self.world.step(render=True)
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
                target_joint_positions = self._controller.forward(
                    target_end_effector_position=path[i], target_end_effector_orientation=end_effector_orientation
                )
                self._articulation_controller.apply_action(target_joint_positions)



