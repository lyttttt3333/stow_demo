import numpy as np
import torch
from omni.isaac.core import World
from omni.isaac.core.materials.particle_material import ParticleMaterial
from omni.isaac.core.materials.physics_material import  PhysicsMaterial
from omni.isaac.core.prims.soft.cloth_prim import ClothPrim
from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView
from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.isaac.core.prims.soft.deformable_prim import DeformablePrim


from omni.isaac.core.utils.nucleus import get_assets_root_path
from pxr import Gf, UsdGeom,Sdf, UsdPhysics, PhysxSchema, UsdLux, UsdShade
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.core.utils.semantics import add_update_semantics, get_semantics
from omni.isaac.core.utils.rotations import euler_angles_to_quat,quat_to_euler_angles
from omni.isaac.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from env.utils.physics_utils import set_collision, set_mass
from omni.physx.scripts import physicsUtils, deformableUtils, particleUtils
from omni.isaac.core.objects import DynamicSphere, DynamicCuboid, VisualCuboid, VisualSphere, VisualCone, FixedCylinder, DynamicCone, FixedCuboid
import omni.kit.commands
import omni.physxdemos as demo
import math
import carb.settings
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim



# todo
# write randomlize function

class Garment():
    def __init__(self, world:World, stage, garment_params:dict, index):
        self.world=world
        self.stage=stage
        self.usd_path=garment_params["path"]
        self.garment_view=UsdGeom.Xform.Define(self.stage,f"/World/Garment")
        self.garment_name=f"garment_{index}"
        self.garment_prim_path=f"/World/Garment/garment_{index}"
        self.particle_system_path="/World/Garment/particleSystem"
        self.particle_material_path="/World/Garment/particleMaterial"

        self.particle_material=ParticleMaterial(
            prim_path=self.particle_material_path, 
            friction=0.5,
            drag=0.1,)
        
        self.particle_system = ParticleSystem(
            prim_path=self.particle_system_path,
            particle_contact_offset=0.005,
            enable_ccd=False,
            global_self_collision_enabled=True,
            non_particle_collision_enabled=True,
            solver_position_iteration_count=16,
            #wind=np.array([0.,0.2,0.]),
        )

        add_reference_to_stage(usd_path=self.usd_path,prim_path=self.garment_prim_path)
        
        self.garment_mesh_prim_path=self.garment_prim_path+"/mesh"
        self.garment=XFormPrim(
            prim_path=self.garment_prim_path,
            name=self.garment_name,
            orientation=garment_params["orientation"],
            position=garment_params["position"],
            scale=garment_params["scale"],
            )
        
        self.garment_mesh=ClothPrim(
            name=self.garment_name+"_mesh",
            prim_path=self.garment_mesh_prim_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
            stretch_stiffness=1e10,
            bend_stiffness=garment_params["stiffness"],
            shear_stiffness=100.0,
            spring_damping=0.5,
        )

        self.world.scene.add(self.garment_mesh)


class Deformable():
    def __init__(self,stage,config):
        self.usd_path=config["path"]
        self.stage=stage
        self.deformable_view=UsdGeom.Xform.Define(self.stage,"/World/Deformable")
        self.deformable_name="deformable"   
        self.deformable_prim_path=find_unique_string_name("/World/Deformable/deformable",is_unique_fn=lambda x: not is_prim_path_valid(x))
        
        self.deformable=XFormPrim(
            prim_path=self.deformable_prim_path,
            name=self.deformable_name,
            position=config["position"],
            orientation=config["orientation"],
            scale=config["scale"])
        
        self.deformable_material_path=find_unique_string_name("/World/Deformable/deformable_material",is_unique_fn=lambda x: not is_prim_path_valid(x))
        
        add_reference_to_stage(usd_path=self.usd_path,prim_path=self.deformable_prim_path)

        self.deformable_mesh_prim_path=self.deformable_prim_path+"/mesh"


        self.deformable_mesh=UsdGeom.Mesh.Get(self.stage, self.deformable_mesh_prim_path)
        self.deformable_points=self.deformable_mesh.GetPointsAttr().Get()
        self.deformable_indices=deformableUtils.triangulate_mesh(self.deformable_mesh)
        self.simulation_resolution=16
        self.mesh_scale=Gf.Vec3f(0.05,0.05,0.05)
        self.collision_points,self.collisions_indices=deformableUtils.compute_conforming_tetrahedral_mesh(self.deformable_points,self.deformable_indices)
        self.simulation_points,self.simulation_indices=deformableUtils.compute_voxel_tetrahedral_mesh(self.collision_points,self.collisions_indices,self.mesh_scale,self.simulation_resolution)

        self.deformable_mesh_prim=DeformablePrim(
                name=self.deformable_name+"_mesh",
                prim_path=self.deformable_mesh_prim_path,
                collision_indices=self.collisions_indices,
                collision_rest_points=self.collision_points,
                simulation_indices=self.simulation_indices,
                simulation_rest_points=self.simulation_points,
            )


class Human():
    def __init__(self,path):
        self.path=path
        self.prim_path="/World/Human"
        add_reference_to_stage(usd_path=path,prim_path=self.prim_path)

        self.rigid_form=XFormPrim(
            prim_path="/World/Human/male_adult_construction_03",
            name="human",
            position=np.array([0.0,0.0,0.0]),
            orientation=euler_angles_to_quat(np.array([0.,0.,np.pi/2]))
        )

        if True:
            self.geom_prim=GeometryPrim(
                prim_path="/World/Human/male_adult_construction_03/male_adult_construction_03/male_adult_construction_04/opaque__fabric__shirt",
                collision=True
            )
            self.geom_prim.set_collision_approximation("none")
            self.geom_prim=GeometryPrim(
                prim_path="/World/Human/male_adult_construction_03/male_adult_construction_03/male_adult_construction_04/trans__organic__whitemaleskin",
                collision=True
            )
            self.geom_prim.set_collision_approximation("none")
            self.geom_prim=GeometryPrim(
                prim_path="/World/Human/male_adult_construction_03/male_adult_construction_03/male_adult_construction_04/opaque__fabric__jeans",
                collision=True
            )
            self.geom_prim.set_collision_approximation("none")


class Rigid_old():
    def __init__(self, root_path, rigid_config):
        self._root = root_path
        self._render_material = False
        self.name="rigid_0"

        add_reference_to_stage(usd_path=rigid_config["path"],prim_path=self._root)

        # define path
        full_path = self._root
        mesh_path = self._root+"/World"
        matetial_path = self._root + "/material"

        self.rigid_form=XFormPrim(
            prim_path=full_path,
            name=self.name,
            position=rigid_config["position"],
            scale=rigid_config["scale"],
            orientation=rigid_config["orientation"],
        )
        
        self.geom_prim=GeometryPrim(
            prim_path=mesh_path,
            collision=True
        )
        self.geom_prim.set_collision_approximation("convexHull")

        self.physics_material=PhysicsMaterial(prim_path=matetial_path, dynamic_friction=0.99,static_friction=0.99)
        self.geom_prim.apply_physics_material(self.physics_material)

class Collider():
    def __init__(self, root_path, rigid_config, name):       
        self._root = root_path
        self._render_material = False
        self.name=name

        add_reference_to_stage(usd_path=rigid_config["path"],prim_path=self._root)

        # define path
        full_path = self._root
        mesh_path = self._root
        matetial_path = self._root + "/material"

        self.rigid_form=XFormPrim(
            prim_path=full_path,
            name=self.name,
            position=rigid_config["position"],
            scale=rigid_config["scale"],
            orientation=rigid_config["orientation"],
        )
        
        self.geom_prim=GeometryPrim(
            prim_path=mesh_path,
            collision=True
        )
        self.geom_prim.set_collision_approximation("convexDecomposition")



class Rigid():
    def __init__(self, root_path, rigid_config, name ):
        #rigid_config = self.add_noise(rigid_config)
            
        self._root = root_path
        self.name=name

        add_reference_to_stage(usd_path=rigid_config["path"],prim_path=self._root)

        # define path
        full_path = self._root
        mesh_path = self._root
        matetial_path = self._root + "/material"

        self.rigid_form=XFormPrim(
            prim_path=full_path,
            name=self.name,
            position=rigid_config["position"],
            scale=rigid_config["scale"],
            orientation=euler_angles_to_quat(rigid_config["orientation"]),
        )
        
        self.geom_prim=GeometryPrim(
            prim_path=mesh_path,
            collision=True
        )
        self.geom_prim.set_collision_approximation("convexHull")


        self.rigid_prim=RigidPrim(
            prim_path=mesh_path,
        )



class Human():
    def __init__(self,path):
        self.path=path
        self.prim_path="/World/Human"
        add_reference_to_stage(usd_path=path,prim_path=self.prim_path)

        self.rigid_form=XFormPrim(
            prim_path="/World/Human/male_adult_construction_03",
            name="human",
            position=np.array([-0.3,0.0,0.0]),
            orientation=euler_angles_to_quat(np.array([0.,0.,np.pi/2])),
            scale=np.array([0.8,0.8,0.8])
        )

        if True:
            self.geom_prim=GeometryPrim(
                prim_path="/World/Human/male_adult_construction_03/male_adult_construction_03/male_adult_construction_04/opaque__fabric__shirt",
                collision=True
            )
            self.geom_prim.set_collision_approximation("none")
            self.geom_prim=GeometryPrim(
                prim_path="/World/Human/male_adult_construction_03/male_adult_construction_03/male_adult_construction_04/trans__organic__whitemaleskin",
                collision=True
            )
            self.geom_prim.set_collision_approximation("none")
            self.geom_prim=GeometryPrim(
                prim_path="/World/Human/male_adult_construction_03/male_adult_construction_03/male_adult_construction_04/opaque__fabric__jeans",
                collision=True
            )
            self.geom_prim.set_collision_approximation("none")

class AttachmentBlock():
    def __init__(self,world, stage, root_path, mesh_0_path, init_place, block_num=1):
        self.world=world
        self.stage=stage
        self._root = root_path
        self._render_material = False
        self.name="attach"
        self.init_place=init_place
        self.block_num=block_num

        # define path
        self.matetial_path = self._root+"/PhysicsMaterial"
        self.mesh_0_path=mesh_0_path
        self.block_path_list=[]
        self.attachment_path_list=[]
        for i in range(block_num):
            self.block_path_list.append(self._root+f"/attach_{i}")
            self.attachment_path_list.append(mesh_0_path[i]+f"/attachment_{i}")

    def create(self,block_visual):
        self.block_prim_list=[]
        for i in range(self.block_num):
            prim = DynamicCuboid(prim_path=self.block_path_list[i], color=np.array([1.0, 0.0, 0.0]),
                        name=f"cube{i}",
                        position=self.init_place[i],
                        scale=np.array([0.015, 0.015, 0.015]),
                        mass=1e5,
                        visible=block_visual)
            self.block_prim_list.append(prim)
            self.world.scene.add(prim)
            #prim.set_mass(0.1)
        self.move_block_controller_list=[move_block._rigid_prim_view for move_block in self.block_prim_list]
        if True:
            for move_block in self.move_block_controller_list:
                move_block.disable_gravities()
        return self.move_block_controller_list


    def enable_gravity(self,flag):
        for index, move_block in enumerate(self.block_prim_list):
            if flag[index] != None:
                move_block.set_mass(0.1)
        for index, move_block in enumerate(self.move_block_controller_list):
            if flag[index] != None:
                move_block.enable_gravities()

    def set_position(self,grasp_point_list):

        for i in range(self.block_num):
            grasp_point=grasp_point_list[i]
            self.block_prim_list[i].set_world_pose(position=grasp_point)

    def get_position(self):
        pose_list=[]
        for i in range(self.block_num):
            pose,_=self.block_prim_list[i].get_world_pose()
            pose_list.append(pose.unsqueeze(0))
        pose_mat=torch.cat(pose_list,dim=0)
        return pose_mat
    
    def attach(self):

        #self.physics_material=PhysicsMaterial(prim_path=self.matetial_path, dynamic_friction=0.2,static_friction=0.99)
        #self.prim.apply_physics_material(self.physics_material)
        for i in range(self.block_num):
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, self.attachment_path_list[i])
            attachment.GetActor0Rel().SetTargets([self.mesh_0_path[i]])
            attachment.GetActor1Rel().SetTargets([self.block_path_list[i]])
            att=PhysxSchema.PhysxAutoAttachmentAPI(attachment.GetPrim())
            att.Apply(attachment.GetPrim())
            _=att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.0)
    
    def detach(self,flag_list):
        for i in range(self.block_num):
            flag=flag_list[i]
            if flag:
                delete_prim(self.attachment_path_list[i])


class WayPoint():
    def __init__(self, root_path, section_num):
        self._root = root_path
        self.whole_taget_list=[]

        if section_num>3:
            raise ValueError("too long")

        for i in range(section_num):
            # define path
            path_start = self._root + f"/start_{i}"
            path_end = self._root + f"/end_{i}"
            color = np.array([0.,0.,0.])
            color[i]=1.0

            # create block
            if i == 0:
                position=np.array([-0.62169,0,0])
            if i == 1:
                position=np.array([-0.35065,0.64568,0.77554])
            if i == 2:
                position=np.array([1.76565  , -0.38631, 1.40685 ])

            orientation=None if i !=0 else np.array([ 2.4701300e-04,  7.7067256e-01, -6.3723123e-01,  2.9873947e-04])
            
            prim_start = VisualCuboid(prim_path=path_start, color=color,
                                position=position, 
                                orientation=orientation,
                                scale=np.array([0.02, 0.02, 0.02]))
            self.whole_taget_list.append(prim_start)


class Camera():
    def create(self, stage, scene, root_path, mesh_0_path, attach_position):
        self._stage = stage
        self._scene = scene
        self._root = root_path
        self._render_material = False
        self.name="attach"

        # define path
        self.root_path = self._root+"/attach_0"
        self.mesh_0_path=mesh_0_path
        self.mesh_1_path=self.root_path
        self.attachment_path = mesh_0_path+"/attachment"

        # create block
        prim = DynamicCuboid(prim_path=self.root_path, color=np.array([1.0, 0.0, 0.0]), mass=1.0,
                            position=attach_position,
                            scale=np.array([3, 3, 3]))
        
        # create attachment
        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, self.attachment_path)
        attachment.GetActor0Rel().SetTargets([self.mesh_0_path])
        #attachment.GetActor1Rel().SetTargets([self.mesh_1_path])
        #attachment.GetActor1Rel().SetTargets(0)
        PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())