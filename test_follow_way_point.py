# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka.tasks import FollowTarget
import numpy as np
from omni.isaac.franka import Franka

my_world = World(stage_units_in_meters=1.0,device="cuda:0",backend="torch")
my_task = FollowTarget(name="follow_target_task")
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("follow_target_task").get_params()
franka_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]
my_franka:Franka = my_world.scene.get_object(franka_name)
my_controller = RMPFlowController(name="target_follower_controller", robot_articulation=my_franka)
articulation_controller = my_franka.get_articulation_controller()
start_position=np.array([0.3, 0.1, 0.5])
end_position=np.array([0.7, 0.1, 0.1])
time_horizon=1000
control_seq=["open",[start_position,end_position],"close",[end_position,start_position],"open"]
gripper_time=100
run_time=1000
if True:
    my_world.reset()
    my_controller.reset()
    for step in control_seq:
        if isinstance(step,str):
            if step == "open":
                for i in range(gripper_time):
                    my_franka.gripper.open()
                    my_world.step(render=True)
            elif step == "close":
                for i in range(gripper_time):
                    my_franka.gripper.close()
                    my_world.step(render=True)
            else:
                raise ValueError("unknown command")
        elif isinstance(step, list):
            start_position=step[0]
            end_position=step[1]
            for i in range(run_time):
                next_step=start_position*(1-i/run_time)+end_position*(i/run_time)
                actions = my_controller.forward(
                    target_end_effector_position=next_step,
                    target_end_effector_orientation=np.array([0.,0.,1.,0.]))
                articulation_controller.apply_action(actions)
                my_world.step(render=True)
        else:
            raise ValueError("unknown type")

if False: 
    while simulation_app.is_running():
        for i in range(time_horizon):
            my_world.step(render=True)
            if my_world.current_time_step_index == 0:
                my_world.reset()
                my_controller.reset()
            observations = my_world.get_observations()
            next_step=start_position*(1-i/time_horizon)+end_position*(i/time_horizon)
            print(observations[target_name]["orientation"])
            actions = my_controller.forward(
                target_end_effector_position=next_step,
                target_end_effector_orientation=observations[target_name]["orientation"],
            )
            if i==10:
                my_franka.gripper.open()
            if i == 300:
                my_franka.gripper.close()
            articulation_controller.apply_action(actions)

simulation_app.close()



