import torch
import numpy as np
import math
from omni.isaac.core.utils.rotations import euler_angles_to_quat,quat_to_euler_angles

DEVICE="cuda:0"

INIT_POSITION=torch.tensor([[-0.2189,  0.0052,  0.9174]], device=DEVICE)



def push_command(start,end):
    PUSH_0_start=torch.tensor([start],device=DEVICE)
    PUSH_0_end=torch.tensor([end],device=DEVICE)
    PUSH_0_pre_start=torch.tensor([start],device=DEVICE) + torch.tensor([[0.,0.,0.2]],device=DEVICE)
    PUSH_0_pre_end=torch.tensor([end],device=DEVICE) + torch.tensor([[0.,0.,0.2]],device=DEVICE)
    move_vec=PUSH_0_end-PUSH_0_start
    move_x=move_vec[0,0].item()+0.001
    move_y=move_vec[0,1].item()
    theta=math.atan(move_y/move_x)
    PUSH_0_direction=euler_angles_to_quat(np.array([-180*np.pi/180,0,-theta]))
    PUSH_0_direction=torch.from_numpy(PUSH_0_direction).to(DEVICE)
    return (PUSH_0_start,PUSH_0_end,PUSH_0_pre_start,PUSH_0_pre_end,PUSH_0_direction)

PUSH0=push_command([-0.31806,0.19144,0.79409],[-0.31806,0.41424,0.79409])
PUSH1=push_command([-0.46495,0.54947,0.78409],[-0.35665,0.54947,0.78409])
PUSH2=push_command([-0.39665,0.54947,0.78409],[-0.30665,0.54947,0.78409])
