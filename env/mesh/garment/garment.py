import numpy as np
from omni.isaac.core import World
from omni.isaac.core.materials.particle_material import ParticleMaterial
from omni.isaac.core.materials.physics_material import  PhysicsMaterial
from omni.isaac.core.prims.soft.cloth_prim import ClothPrim
from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView
from omni.isaac.core.prims.soft.particle_system import ParticleSystem
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

class Garment:
    def __init__(self, world:World, stage, scene, garment_params:dict):
        self.world=world
        self.stage=stage
        self.usd_path=garment_params["path"]
        self.garment_view=UsdGeom.Xform.Define(self.stage,"/World/Garment")
        self.garment_name="garment"
        self.garment_prim_path="/World/Garment/garment"
        self.particle_system_path="/World/Garment/particleSystem"
        self.particle_material_path="/World/Garment/particleMaterial"

        self.particle_material=ParticleMaterial(
            prim_path=self.particle_material_path, 
            friction=0.5)
        
        self.particle_system = ParticleSystem(
            prim_path=self.particle_system_path,
            particle_contact_offset=0.005,
            enable_ccd=False,
            global_self_collision_enabled=True,
            non_particle_collision_enabled=True,
            solver_position_iteration_count=16
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
            bend_stiffness=100.0,
            shear_stiffness=100.0,
            spring_damping=0.2,
        )

        self.world.scene.add(self.garment_mesh)
        
class ParticleCloth():
    def __init__(self, stage, scene, root_path, usd_path):
        self._stage = stage
        self._scene = scene
        self._root = root_path
        self._render_material = False

        add_reference_to_stage(usd_path=usd_path,prim_path=self._root+"/plane0")

        # define path
        mesh_path = self._root+"/plane0"+"/mesh"

        particle_system_path = "/particleSystem"
        particle_system = PhysxSchema.PhysxParticleSystem.Define(stage, particle_system_path)
        particle_system.CreateSimulationOwnerRel().SetTargets([self._scene.GetPath()])
        particle_system.CreateParticleContactOffsetAttr().Set(0.005)

        # define configurations
        stretchStiffness = 10000.0
        bendStiffness = 200.0
        shearStiffness = 100.0
        damping = 0.2
        particle_mass = 0.0002
        particleUtils.add_physx_particle_cloth(
            stage=stage,
            path=mesh_path,
            dynamic_mesh_path=None,
            particle_system_path="/particleSystem",#system_path,
            spring_stretch_stiffness=stretchStiffness,
            spring_bend_stiffness=bendStiffness,
            spring_shear_stiffness=shearStiffness,
            spring_damping=damping,
            self_collision=True,
            self_collision_filter=True,
        )
        self.garment=XFormPrim(
            prim_path="/plane0",
            name="garment",
            position=np.array([0.50096,-0.83624,0.2877]),
            scale=np.array([0.01, 0.01, 0.01]),
            )
class ParticleDemoBase:
    # helpers:
    
    def setup_base_scene(self, stage):
        if False:
            #stage = omni.usd.get_context().get_stage()
            self._stage = stage
            UsdGeom.SetStageUpAxis(self._stage, UsdGeom.Tokens.y)
            UsdGeom.SetStageMetersPerUnit(self._stage, 0.01)
            

            # light
            lightPath="/World/SphereLight"
            self._sphereLight = UsdLux.SphereLight.Define(self._stage,lightPath)
            self._sphereLight.CreateRadiusAttr(150)
            self._sphereLight.CreateIntensityAttr(30000)
            self._sphereLight.AddTranslateOp().Set(Gf.Vec3f(200.0, 250, 800.0))

            # Physics scene
            scenePath = "/World/physicsScene"
            self._scene = UsdPhysics.Scene.Define(self._stage, scenePath)

            groundPath="/World/groundPlane"

            physicsUtils.add_ground_plane(
                self._stage,
                groundPath,
                UsdGeom.GetStageUpAxis(self._stage),
                1000.0,
                Gf.Vec3f(0.0),
                Gf.Vec3f(0.5),
            )

        # demo camera:
            self._cam = UsdGeom.Camera.Define(self._stage, self.demo_camera)
            self._cam.CreateFocalLengthAttr().Set(18.14756)
            location = Gf.Vec3f(0, 150, 750)
            physicsUtils.setup_transform_as_scale_orient_translate(self._cam)
            physicsUtils.set_or_add_translate_op(self._cam, translate=location)

    def create_pbd_material(self, root, mat_name: str, color_rgb: Gf.Vec3f = Gf.Vec3f(0.2, 0.2, 0.8)) -> Sdf.Path:
        # create material for extras
        create_list = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_created_list=create_list,
            bind_selected_prims=False,
        )
        target_path = root + "/Looks/" + mat_name
        if create_list[0] != target_path:
            omni.kit.commands.execute("MovePrims", paths_to_move={create_list[0]: target_path})
        shader = UsdShade.Shader.Get(self._stage, target_path + "/Shader")
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(color_rgb)
        return Sdf.Path(target_path)

    def create_particle_box_collider(
        self,
        path: Sdf.Path,
        side_length: float = 100.0,
        height: float = 50.0,
        translate: Gf.Vec3f = Gf.Vec3f(0, 0, 0),
        thickness: float = 10.0,
        add_cylinder_top=True,
    ):
        """
        Creates an invisible collider box to catch particles. Opening is in y-up

        Args:
            path:           box path (xform with cube collider children that make up box)
            side_length:    inner side length of box
            height:         height of box
            translate:      location of box, w.r.t it's bottom center
            thickness:      thickness of the box walls
        """
        xform = UsdGeom.Xform.Define(self._stage, path)
        xform.MakeInvisible()
        xform_path = xform.GetPath()
        physicsUtils.set_or_add_translate_op(xform, translate=translate)
        cube_width = side_length + 2.0 * thickness
        offset = side_length * 0.5 + thickness * 0.5
        # front and back (+/- x)
        cube = UsdGeom.Cube.Define(self._stage, xform_path.AppendChild("front"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(0, height * 0.5, offset))
        physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(cube_width, height, thickness))

        if add_cylinder_top:
            top_cylinder = UsdGeom.Cylinder.Define(self._stage, xform_path.AppendChild("front_top_cylinder"))
            top_cylinder.CreateRadiusAttr().Set(thickness * 0.5)
            top_cylinder.CreateHeightAttr().Set(cube_width)
            top_cylinder.CreateAxisAttr().Set("X")
            UsdPhysics.CollisionAPI.Apply(top_cylinder.GetPrim())
            physicsUtils.set_or_add_translate_op(top_cylinder, Gf.Vec3f(0, height, offset))

        cube = UsdGeom.Cube.Define(self._stage, xform_path.AppendChild("back"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(0, height * 0.5, -offset))
        physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(cube_width, height, thickness))

        if add_cylinder_top:
            top_cylinder = UsdGeom.Cylinder.Define(self._stage, xform_path.AppendChild("back_top_cylinder"))
            top_cylinder.CreateRadiusAttr().Set(thickness * 0.5)
            top_cylinder.CreateHeightAttr().Set(cube_width)
            top_cylinder.CreateAxisAttr().Set("X")
            UsdPhysics.CollisionAPI.Apply(top_cylinder.GetPrim())
            physicsUtils.set_or_add_translate_op(top_cylinder, Gf.Vec3f(0, height, -offset))

        # left and right:
        cube = UsdGeom.Cube.Define(self._stage, xform_path.AppendChild("left"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(-offset, height * 0.5, 0))
        physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(thickness, height, cube_width))

        if add_cylinder_top:
            top_cylinder = UsdGeom.Cylinder.Define(self._stage, xform_path.AppendChild("left_top_cylinder"))
            top_cylinder.CreateRadiusAttr().Set(thickness * 0.5)
            top_cylinder.CreateHeightAttr().Set(cube_width - thickness)
            top_cylinder.CreateAxisAttr().Set("Z")
            UsdPhysics.CollisionAPI.Apply(top_cylinder.GetPrim())
            physicsUtils.set_or_add_translate_op(top_cylinder, Gf.Vec3f(-offset, height, 0))

        cube = UsdGeom.Cube.Define(self._stage, xform_path.AppendChild("right"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(offset, height * 0.5, 0))
        physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(thickness, height, cube_width))

        if add_cylinder_top:
            top_cylinder = UsdGeom.Cylinder.Define(self._stage, xform_path.AppendChild("right_top_cylinder"))
            top_cylinder.CreateRadiusAttr().Set(thickness * 0.5)
            top_cylinder.CreateHeightAttr().Set(cube_width - thickness)
            top_cylinder.CreateAxisAttr().Set("Z")
            UsdPhysics.CollisionAPI.Apply(top_cylinder.GetPrim())
            physicsUtils.set_or_add_translate_op(top_cylinder, Gf.Vec3f(offset, height, 0))



class ParticleSamplerDemo(demo.Base, ParticleDemoBase):

    def create(self, stage, Sample_Volume, Particle_Contact_Offset,scene,root_path):
        self._scene = scene
        self._stage = stage
        self._root = root_path

        # configure and create particle system
        particle_system_path = "/World/fluid/particleSystem"
        particle_system = PhysxSchema.PhysxParticleSystem.Define(stage, particle_system_path)
        particle_system.CreateSimulationOwnerRel().SetTargets([self._scene.GetPath()])
        # The simulation determines the other offsets from the particle contact offset
        particle_system.CreateParticleContactOffsetAttr().Set(Particle_Contact_Offset)



        # configure and create particle system
        
        particle_material_path = "/World/fluid/particleMaterial"
        particleUtils.add_pbd_particle_material(stage, particle_material_path)
        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath(particle_system_path), particle_material_path
        )

        # create a cube mesh that shall be sampled:
        cube_mesh_path = "/World/fluid/Cube"
        cube_resolution = (5)
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",prim_path=cube_mesh_path, prim_type="Cube", u_patches=cube_resolution, v_patches=cube_resolution
        )
        cube_mesh = UsdGeom.Mesh.Get(stage, cube_mesh_path)
        physicsUtils.set_or_add_translate_op(cube_mesh, Gf.Vec3f(0.0, 0.0, 2.0))




        
        # configure target particle set:
        if True:
            particle_points_path = self._root+"/sampledParticles"
            points = UsdGeom.Points.Define(stage, particle_points_path)

            # add render material:
            material_path = self.create_pbd_material(self._root,"OmniPBR")
            omni.kit.commands.execute(
                "BindMaterialCommand", prim_path=particle_points_path, material_path=material_path, strength=None
            )
            particle_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(points.GetPrim())
            #particle_set_api.CreateParticleSystemRel().SetTargets([particle_system_path])
            PhysxSchema.PhysxParticleAPI(particle_set_api).CreateParticleSystemRel().SetTargets([particle_system_path])

            # compute particle sampler sampling distance
            # use particle fluid restoffset to determine sampler distance, using same formula as simulation, see
            # https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html#offset-autocomputation
            fluid_rest_offset = 0.99 * 0.6 * Particle_Contact_Offset
            particle_sampler_distance = 2.0 * fluid_rest_offset

            # reference the particle set in the sampling api
            sampling_api = PhysxSchema.PhysxParticleSamplingAPI.Apply(cube_mesh.GetPrim())
            sampling_api.CreateParticlesRel().AddTarget(particle_points_path)
            sampling_api.CreateSamplingDistanceAttr().Set(particle_sampler_distance)
            sampling_api.CreateMaxSamplesAttr().Set(5e4)
            sampling_api.CreateVolumeAttr().Set(Sample_Volume)

        # create catch box:
        if False:
            self.create_particle_box_collider(
                self._root+"/box",
                side_length=150.0,
                height=200.0,
                thickness=20.0,
                translate=Gf.Vec3f(0, -5, 0),
            )


class Rigid2():
    def __init__(self, root_path, usd_path):
        self._root = root_path
        self._render_material = False
        self.name="rigid_0"

        add_reference_to_stage(usd_path=usd_path,prim_path=self._root)

        # define path
        full_path = self._root
        mesh_path = self._root+"/World"
        matetial_path = self._root + "/material"

        self.rigid_form=XFormPrim(
            prim_path=full_path,
            name=self.name,
            position=np.array([0.74416, -0.25376, 0.57602]),
            scale=np.array([0.0001,0.008,0.001]),
            orientation=euler_angles_to_quat(np.array([0.,0.,np.pi/2]))
        )
        
        self.geom_prim=GeometryPrim(
            prim_path=mesh_path,
            collision=True
        )
        self.geom_prim.set_collision_approximation("convexHull")

        self.physics_material=PhysicsMaterial(prim_path=matetial_path, dynamic_friction=0.99,static_friction=0.99)
        self.geom_prim.apply_physics_material(self.physics_material)
        if False:

            self.geom_gripper_right=GeometryPrim(
                prim_path="/World/Franka/panda_rightfinger/geometry/panda_rightfinger",
                collision=True,
            )
            self.geom_gripper_lift=GeometryPrim(
                prim_path="/World/Franka/panda_leftfinger/geometry/panda_leftfinger",
                collision=True,
            )
            self.material4gripper=PhysicsMaterial(prim_path="/material", dynamic_friction=0.99,static_friction=0.99)
            self.geom_gripper_right.set_collision_approximation("sphereFill")
            #self.geom_gripper_right.apply_physics_material(self.material4gripper)
            self.geom_gripper_lift.set_collision_approximation("sphereFill")
            #self.geom_gripper_lift.apply_physics_material(self.material4gripper)


class Rigid():
    def __init__(self, root_path, usd_path):
        self._root = root_path
        self._render_material = False
        self.name="rigid_0"

        add_reference_to_stage(usd_path=usd_path,prim_path=self._root)

        # define path
        full_path = self._root
        mesh_path = self._root+"/World"
        matetial_path = self._root + "/material"

        self.rigid_form=XFormPrim(
            prim_path=full_path,
            name=self.name,
            position=np.array([0.15618, 0.06288, 0.23628]),
            scale=np.array([0.004,0.006,0.004]),
            orientation=euler_angles_to_quat(np.array([0.,0.,np.pi/2]))
        )
        
        self.geom_prim=GeometryPrim(
            prim_path=mesh_path,
            collision=True
        )
        self.geom_prim.set_collision_approximation("convexHull")

        self.physics_material=PhysicsMaterial(prim_path=matetial_path, dynamic_friction=0.99,static_friction=0.99)
        self.geom_prim.apply_physics_material(self.physics_material)


class Cloth(demo.Base, ParticleDemoBase):

    def create(self, stage, root_path, usd_path):
        self._stage = stage
        self._root = root_path
        self._render_material = False

        add_reference_to_stage(usd_path=usd_path,prim_path=self._root+"/plane0")

        # define path
        mesh_path = self._root+"/plane0"+"/mesh"
        system_path = self._root+"/plane0"+"/system"

        # define configurations
        stretchStiffness = 10000.0
        bendStiffness = 200.0
        shearStiffness = 100.0
        damping = 0.2
        particle_mass = 0.0002
        particleUtils.add_physx_particle_cloth(
            stage=stage,
            path=mesh_path,
            dynamic_mesh_path=None,
            particle_system_path="/World/particleSystem",
            spring_stretch_stiffness=stretchStiffness,
            spring_bend_stiffness=bendStiffness,
            spring_shear_stiffness=shearStiffness,
            spring_damping=damping,
            self_collision=True,
            self_collision_filter=True,
        )
        particleUtils.add_physx_particle_isosurface(
            stage=stage,
            path=mesh_path,
        )


        # configure mass:
        if False:
            plane_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            num_verts = len(plane_mesh.GetPointsAttr().Get())
            mass = particle_mass * num_verts
            massApi = UsdPhysics.MassAPI.Apply(plane_mesh.GetPrim())
            massApi.GetMassAttr().Set(mass)

        # add render material:
        if self._render_material:
            material_path = self.create_pbd_material(self._root, "OmniPBR")
            omni.kit.commands.execute(
                "BindMaterialCommand", prim_path=mesh_path, material_path=material_path, strength=None
            )

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
            self.attachment_path_list.append(mesh_0_path+f"/attachment_{i}")

    def create(self,block_visual):
        self.block_prim_list=[]
        for i in range(self.block_num):
            prim = DynamicCuboid(prim_path=self.block_path_list[i], color=np.array([1.0, 0.0, 0.0]),
                        name=f"cube{i}",
                        position=self.init_place[i],
                        scale=np.array([0.02, 0.02, 0.02]),
                        mass=1e5,
                        visible=block_visual)
            self.block_prim_list.append(prim)
            self.world.scene.add(prim)
        self.move_block_controller_list=[move_block._rigid_prim_view for move_block in self.block_prim_list]
        for move_block in self.move_block_controller_list:
            move_block.disable_gravities()
        return self.move_block_controller_list

    def set_position(self,grasp_point_list):
        assert grasp_point_list.shape[0]==self.block_num
        for i in range(self.block_num):
            grasp_point=grasp_point_list[i]
            self.block_prim_list[i].set_world_pose(position=grasp_point)

    def attach(self,flag_list):

        #self.physics_material=PhysicsMaterial(prim_path=self.matetial_path, dynamic_friction=0.2,static_friction=0.99)
        #self.prim.apply_physics_material(self.physics_material)
        for i in range(self.block_num):
            flag=flag_list[i]
            if flag:
                attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, self.attachment_path_list[i])
                attachment.GetActor0Rel().SetTargets([self.mesh_0_path])
                attachment.GetActor1Rel().SetTargets([self.block_path_list[i]])
                att=PhysxSchema.PhysxAutoAttachmentAPI(attachment.GetPrim())
                att.Apply(attachment.GetPrim())
                _=att.CreateDeformableVertexOverlapOffsetAttr(defaultValue=0.0)
            else:
                pass
    
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
                position=np.array([1.90277, -1.07215,0.50985])
            if i == 1:
                position=np.array([-0.54685  , 0.04618, 0.19871])
            if i == 2:
                position=np.array([1.76565  , -0.38631, 1.40685 ])

            orientation=None if i !=0 else np.array([ 2.4701300e-04,  7.7067256e-01, -6.3723123e-01,  2.9873947e-04])
            
            prim_start = VisualCone(prim_path=path_start, color=color,
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