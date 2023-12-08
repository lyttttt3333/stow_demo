
from omni.isaac.kit import SimulationApp

load = []
load.append("/home/luhr/isaacgarment/TCLC_002/TCLC_002_obj.usd")


simulation_app = SimulationApp({"headless": False})

import numpy as np
import torch
import asyncio
import random

from omni.isaac.core import SimulationContext, World
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import XFormPrim, ClothPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.core.utils.semantics import add_update_semantics, get_semantics
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.materials.particle_material import ParticleMaterial
from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.isaac.sensor import Camera
import omni.replicator.core as rep


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
    )


def get_indexes(arr, value):
    indexes = []
    for i in range(len(arr)):
        if arr[i] != value:
            indexes.append(i)
    return indexes


my_world = World(stage_units_in_meters=1.0, backend="torch", device="cuda:0")
my_world.scene.add_default_ground_plane()

particle_material = ParticleMaterial(
    prim_path="/World/particleMaterial",
    friction=0.2,
)

# radius = 0.5 * (0.6 / 5.0)
# restOffset = radius
# contactOffset = restOffset * 1.5
particle_system = ParticleSystem(
    prim_path="/World/particleSystem",
    simulation_owner=my_world.get_physics_context().prim_path,
)

for i in range(1):
    cloth_path = f"/World/cloth_{i}"
    add_reference_to_stage(usd_path=load[i], prim_path=cloth_path)

    p = random.uniform(0, 1)
    my_cloth = XFormPrim(
        prim_path=cloth_path,
        name="cloth",
        position=np.array([p, -p, p]),
        scale=np.array([0.005, 0.005, 0.005]),
    )

    my_cloth_mesh = ClothPrim(
        prim_path=cloth_path + "/mesh",
        particle_system=particle_system,
        particle_material=particle_material,
        stretch_stiffness=1e4,
        bend_stiffness=100.0,
        shear_stiffness=100.0,
        spring_damping=0.2,
    )

    add_update_semantics(prim=get_prim_at_path(prim_path=cloth_path), semantic_label=f"1{i}")
my_world.scene.add(my_cloth)

# ori = qua_mult([-0.5, 0.5, -0.5, -0.5], [0.20245, 0.11956, 0.5, 0.83692])

# camera = Camera(
#     prim_path="/World/camera",
#     position=torch.tensor([1.25922, 1.55992, 0.96736]),
#     frequency=20,
#     resolution=(256, 256),
#     orientation=ori,
# )

# assets_root_path = get_assets_root_path()
# asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
# add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka_1")
# articulated_system_1 = my_world.scene.add(Robot(prim_path="/World/Franka_1", name="my_franka_1"))

my_world.reset()

'''
camera.initialize()
camera.add_instance_segmentation_to_frame()
camera.add_pointcloud_to_frame(include_unlabelled=True)
'''

# file = "E:/ov/issaclearn/point_cloud_data.txt"

# RESOLUTION = (512, 512)
# # rep_camera = rep.create.camera(camera)
# render_product = rep.create.render_product(camera.prim_path, RESOLUTION)
# point_cloud = rep.AnnotatorRegistry.get_annotator("pointcloud", init_params={"includeUnlabelled": False})
# point_cloud.attach(render_product)
# rep.orchestrator.step()
# pc_data = point_cloud.get_data(device="cuda")
# np.savetxt(file, pc_data['data'])

'''
for i in range(110):
    simulation_app.update()
    print(i)
    if i == 100:
        print(list(camera.get_current_frame()['pointcloud']['info']))
        indexes = (get_indexes(camera.get_current_frame()['pointcloud']['info']['pointSemantic'], 1))
        np.savetxt(file, camera.get_current_frame()['pointcloud']['data'][indexes])
'''


# np.savetxt(file, camera.get_current_frame())
while simulation_app.is_running():
    my_world.step(render=True)

simulation_app.close()
