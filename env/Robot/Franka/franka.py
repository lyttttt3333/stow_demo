import numpy as np
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka import KinematicsSolver


class WrapFranka(Franka):
    def __init__(self,world:World,Position=np.ndarray,prim_path:str=None,robot_name:str=None,):
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
        self.world.scene.add(Franka(prim_path=self._franka_prim_path, name=self._franka_robot_name,position=Position))
        self.world.reset()
        self._robot:Franka=self.world.scene.get_object(self._franka_robot_name)
        self._articulation_controller=self._robot.get_articulation_controller()
        self._controller=RMPFlowController(name="rmpflow_controller",robot_articulation=self._robot)
        self._kinematic_solover=KinematicsSolver(self._robot)
        self._pick_place_controller=PickPlaceController(name="pick_place_controller",robot_articulation=self._robot,gripper=self._robot.gripper)
        self._controller.reset()
        self._pick_place_controller.reset()
    def get_cur_ee_pos(self):
        return self._kinematic_solover.compute_end_effector_pose()
    
    def movep(self, target_pos, target_ori, speed, max_limit=1000, eps=1e-4):
        for step_id in range(max_limit):
            cur_ee_pos=self.get_cur_ee_pos()
            delta=target_pos-cur_ee_pos
            dist=np.linalg.norm(delta)
            if dist<eps:
                break
            dirction=delta/dist
            if dist<speed:
                next_target=target_pos
            else:
                next_target=cur_ee_pos+dirction*speed
    
    def pick_and_place(self,pick,place):
        self._pick_place_controller.reset()
        while 1:
            self.world.step(render=True)
            actions=self._pick_place_controller.forward(
                picking_position=pick,
                placing_position=place,
                current_joint_positions=self._robot.get_joint_positions(),
                end_effector_offset=np.array([0,0.005,0]),
            )
            if self._pick_place_controller.is_done():
                break
            self._articulation_controller.apply_action(actions)
            


