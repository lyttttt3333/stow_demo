import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name

def add_workspace(world:World):
    prim_path=find_unique_string_name(initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x))
    cube_name=find_unique_string_name(initial_name="workspace", is_unique_fn=lambda x: not world.scene.object_exists(x))
    world.scene.add(FixedCuboid(prim_path=prim_path, name=cube_name, position=np.array([1.1,0,0]),scale=np.array([2,2,0.1]),color=np.array([1,1,1])))
    return 
