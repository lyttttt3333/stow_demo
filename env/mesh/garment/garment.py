import numpy as np
from omni.isaac.core import World
from omni.isaac.core.materials.particle_material import ParticleMaterial
from omni.isaac.core.prims.soft.cloth_prim import ClothPrim
from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView
from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import Gf, UsdGeom
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.core.utils.semantics import add_update_semantics, get_semantics
from omni.isaac.core.prims import XFormPrim, ClothPrim

class Garment:
    def __init__(self,world:World,usd_path:str):
        self.world=world
        self.usd_path=usd_path
        self.stage=world.stage
        self.garment_view=UsdGeom.Xform.Define(self.stage,"/World/Garment")
        self.garment_name=find_unique_string_name(initial_name="garment",is_unique_fn=lambda x: not world.scene.object_exists(x))
        self.garment_prim_path=find_unique_string_name("/World/Garment/garment",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.particle_system_path=find_unique_string_name("/World/Garment/particleSystem",is_unique_fn=lambda x: not is_prim_path_valid(x))
        self.particle_material_path=find_unique_string_name("/World/Garment/particleMaterial",is_unique_fn=lambda x: not is_prim_path_valid(x))

        self.particle_material=ParticleMaterial(prim_path=self.particle_material_path, drag=0.1, lift=0.3, friction=0.6)
        radius = 0.5 * (0.6 / 5.0)
        restOffset = radius
        contactOffset = restOffset * 1.5
        self.particle_system = ParticleSystem(
            prim_path=self.particle_system_path,
            simulation_owner=self.world.get_physics_context().prim_path,
            rest_offset=restOffset,
            contact_offset=contactOffset,
            solid_rest_offset=restOffset,
            fluid_rest_offset=restOffset,
            particle_contact_offset=contactOffset,
        )

        add_reference_to_stage(usd_path=self.usd_path,prim_path=self.garment_prim_path)
        self.garment_mesh_prim_path=self.garment_prim_path+"/mesh"

        self.garment=XFormPrim(
            prim_path=self.garment_prim_path,
            name=self.garment_name,
            position=np.array([1,1,1]),
            scale=np.array([0.005, 0.005, 0.005]))
        
        self.garment_mesh=ClothPrim(
            name=self.garment_name+"_mesh",
            prim_path=self.garment_mesh_prim_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
            stretch_stiffness=1e4,
            bend_stiffness=100.0,
            shear_stiffness=100.0,
            spring_damping=0.2,
        )
        self.world.scene.add(self.garment_mesh)
        
        
        

        
        
