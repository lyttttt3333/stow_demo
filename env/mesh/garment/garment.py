import numpy as np
from omni.isaac.core import World
from omni.isaac.core.materials.particle_material import ParticleMaterial
from omni.isaac.core.prims.soft.cloth_prim import ClothPrim
from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView
from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.isaac.core.utils.nucleus import get_assets_root_path
from pxr import Gf, UsdGeom,Sdf, UsdPhysics, PhysxSchema, UsdLux, UsdShade
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from omni.isaac.core.utils.semantics import add_update_semantics, get_semantics
from omni.isaac.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from env.utils.physics_utils import set_collision, set_mass
from omni.physx.scripts import physicsUtils, deformableUtils, particleUtils
from omni.isaac.core.objects import DynamicSphere, DynamicCuboid
import omni.kit.commands
import omni.physxdemos as demo
import math
import carb.settings


# todo
# write randomlize function

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

        self.particle_material=ParticleMaterial(prim_path=self.particle_material_path, friction=0.2)

        self.particle_system = ParticleSystem(
            prim_path=self.particle_system_path,
            simulation_owner=self.world.get_physics_context().prim_path,
            particle_contact_offset=0.02,
            contact_offset=0.02,
            rest_offset=0.012,
            enable_ccd=True,
            global_self_collision_enabled=True,
            non_particle_collision_enabled=True,
            solver_position_iteration_count=32
        )

        add_reference_to_stage(usd_path=self.usd_path,prim_path=self.garment_prim_path)
        
        self.garment_mesh_prim_path=self.garment_prim_path+"/mesh"
        self.garment=XFormPrim(
            prim_path=self.garment_prim_path,
            name=self.garment_name,
            position=np.array([0.5,-0.3,0.6]),
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

class Rigid:
    def __init__(self,world:World,usd_path:str):
        self.world=world
        self.usd_path=usd_path
        self.stage=world.stage
        self.name="rigid"

        add_reference_to_stage(usd_path=self.usd_path,prim_path="/World")
        
        
        self.rigid_form=XFormPrim(
            prim_path="/World/franka_gripper",
            name=self.name,
            position=np.array([0.5, 0.5, 0.2]),
        )
        
        self.rigid_prim=RigidPrim(
            name=self.name,
            prim_path="/World/franka_gripper",
        )
        self.rigid_prim.enable_rigid_body_physics()

        self.geom_prim=GeometryPrim(
            prim_path="/World/franka_gripper",
            collision=True
        )
        self.geom_prim.set_collision_approximation("convexHull")


        self.world.scene.add(self.rigid_prim)
        
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
    demo_camera = "/World/Camera"

    def create(self, stage, Sample_Volume, Particle_Contact_Offset,scene,root_path):
        self._scene = scene
        self._stage = stage
        self._root = root_path

        # configure and create particle system
        particle_system_path = "/World/particleSystem"

        # create particle material and assign it to the system:
        particle_material_path = self._root+"/particleMaterial"
        particleUtils.add_pbd_particle_material(stage, particle_material_path)
        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath(particle_system_path), particle_material_path
        )

        # create a cube mesh that shall be sampled:
        cube_mesh_path = Sdf.Path(omni.usd.get_stage_next_free_path(stage, "Cube", True))
        cube_mesh_path = self._root+"/Cube"
        cube_resolution = (
            2  # resolution can be low because we'll sample the surface / volume only irrespective of the vertex count
        )
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",prim_path=cube_mesh_path, prim_type="Cube", u_patches=cube_resolution, v_patches=cube_resolution
        )
        cube_mesh = UsdGeom.Mesh.Get(stage, cube_mesh_path)
        physicsUtils.set_or_add_translate_op(cube_mesh, Gf.Vec3f(0.0, 55.0, 0.0))

        # configure target particle set:
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
        sampling_api.CreateMaxSamplesAttr().Set(5e5)
        sampling_api.CreateVolumeAttr().Set(Sample_Volume)
        # reference the particle set in the sampling api
        sampling_api = PhysxSchema.PhysxParticleSamplingAPI.Apply(cube_mesh.GetPrim())
        sampling_api.CreateParticlesRel().AddTarget(particle_points_path)
        sampling_api.CreateSamplingDistanceAttr().Set(particle_sampler_distance)
        sampling_api.CreateMaxSamplesAttr().Set(5e5)
        sampling_api.CreateVolumeAttr().Set(Sample_Volume)

        # create catch box:
        if 0:
            self.create_particle_box_collider(
                self._root+"/box",
                side_length=150.0,
                height=200.0,
                thickness=20.0,
                translate=Gf.Vec3f(0, -5, 0),
            )


class BoxOnPlaneInstanced():
    def create(self, stage, scene, root_path, usd_path):
        self._stage = stage
        self._scene = scene
        self._root = root_path
        self._render_material = False
        self.name="rigid_0"

        add_reference_to_stage(usd_path=usd_path,prim_path=self._root+"/rigid_0")

        # define path
        full_path = self._root+"/rigid_0"+"/franka_gripper"
        mesh_path = self._root+"/rigid_0"+"/franka_gripper"+"/mesh"

        self.rigid_form=XFormPrim(
            prim_path=full_path,
            name=self.name,
            position=np.array([0.5, 0.5, 0.2]),
        )
        
        self.rigid_prim=RigidPrim(
            name=self.name,
            prim_path=full_path
        )
        self.rigid_prim.enable_rigid_body_physics()

        self.geom_prim=GeometryPrim(
            prim_path=mesh_path,
            collision=True
        )
        self.geom_prim.set_collision_approximation("convexHull")
    
class BoxOnPlaneInstancedOld():
    def create(self, stage, defaultPrimPath):

        geomPointInstancerPath = defaultPrimPath + "/pointinstancer"

        # Box instanced
        boxActorPath = geomPointInstancerPath + "/boxActor"
        size = 25.0
        color = Gf.Vec3f(71.0 / 255.0, 165.0 / 255.0, 1.0)

        cubeGeom = UsdGeom.Cube.Define(stage, boxActorPath)
        cubePrim = stage.GetPrimAtPath(boxActorPath)
        cubeGeom.CreateSizeAttr(size)
        cubeGeom.AddScaleOp().Set(Gf.Vec3f(1.0))
        cubeGeom.CreateDisplayColorAttr().Set([color])

        UsdPhysics.CollisionAPI.Apply(cubePrim)
        physicsAPI = UsdPhysics.RigidBodyAPI.Apply(cubePrim)        
        UsdPhysics.MassAPI.Apply(cubePrim)

        # indices
        meshIndices = [0, 0]
        positions = [Gf.Vec3f(-125.0, 0.0, 500.0), Gf.Vec3f(125.0, 0.0, 500.0)]
        orientations = [Gf.Quath(1.0, 0.0, 0.0, 0.0), Gf.Quath(0.8660254, 0.0, 0.5, 0.0)]
        linearVelocities = [Gf.Vec3f(0.0), Gf.Vec3f(0.0, 0.0, 0.0)]
        angularVelocities = [Gf.Vec3f(0.0, 10.0, 0.0), Gf.Vec3f(0.0)]

        # Create point instancer
        shapeList = UsdGeom.PointInstancer.Define(stage, Sdf.Path(geomPointInstancerPath))
        meshList = shapeList.GetPrototypesRel()
        # add mesh reference to point instancer
        meshList.AddTarget(Sdf.Path(boxActorPath))

        shapeList.GetProtoIndicesAttr().Set(meshIndices)
        shapeList.GetPositionsAttr().Set(positions)
        shapeList.GetOrientationsAttr().Set(orientations)
        shapeList.GetVelocitiesAttr().Set(linearVelocities)
        shapeList.GetAngularVelocitiesAttr().Set(angularVelocities)


        self.autofocus = True # autofocus on the scene at first update
        self.autofocus_zoom = 0.28 # Get a bit closer

class ParticleCloth(demo.Base, ParticleDemoBase):

    def create(self, stage, scene, root_path, usd_path):
        self._stage = stage
        self._scene = scene
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
            particle_system_path="/World/particleSystem",#system_path,
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
        attachment.GetActor1Rel().SetTargets([self.mesh_1_path])
        PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())

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
        attachment.GetActor1Rel().SetTargets([self.mesh_1_path])
        PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())