import sys
import os
import numpy as np
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+"/env")
from omni.isaac.kit import SimulationApp
simulation_app=SimulationApp({"headless":False})
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from env.Robot.Robot import Robot
from env.config.config import Config
from env.utils.isaac_utils import add_workspace
from env.mesh.garment.garment import Garment, Rigid, ParticleSamplerDemo,BoxOnPlaneInstanced, ParticleCloth, AttachmentBlock
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.physx.scripts import physicsUtils, deformableUtils, particleUtils

class DeformEnvOld:
    def __init__(self,config:Config):

        self.world = World(backend="torch",device="cuda:0")
        self.world.scene.add_default_ground_plane()
        self.config=config
        self.stage=self.world.stage
        # self.robot=Robot(self.world,self.config.robot_config)
        # add_workspace(self.world)
        self.garment=Garment(self.world,"/home/sim/isaacgarment/ClothesNetData/ClothesNetMUSD/Tops/Collar_Lsleeve_FrontClose/TCLC_074.usd")
        self.world.reset()
        for i in range(500):
            self.world.step(render=True)

    
    def pick_and_place_test(self):
        # self.robot.pick_and_place([np.array([0.5,0.4,0.1])],[np.array([0.5,-0.3,0.15])])
        # self.robot.pick_and_place([np.array([0.3,0.3,0.1])],[np.array([0.5,-0.3,0.1])])
        self.robot.open()
        self.robot.movep([np.array([0.3,0.3,0.11])])
        print("out")
        for i in range(100):
            self.world.step(render=True)

class DeformEnv:
    def __init__(self,config:Config):

        self.config=config
        self.defaultPrimPath = "/World"
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)

        # light
        sphereLight = UsdLux.SphereLight.Define(stage, self.defaultPrimPath + "/SphereLight")
        sphereLight.CreateRadiusAttr(150)
        sphereLight.CreateIntensityAttr(30000)
        sphereLight.AddTranslateOp().Set(Gf.Vec3f(650.0, 0.0, 1150.0))

        # Physics scene
        scene = UsdPhysics.Scene.Define(stage, self.defaultPrimPath + "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(981.0)

        # groundplane
        physicsUtils.add_ground_plane(
            stage,
            "/World/groundPlane",
            UsdGeom.GetStageUpAxis(stage),
            1000.0,
            Gf.Vec3f(0.0),
            Gf.Vec3f(0.5),
            )
        
        # particle system
        Particle_Contact_Offset =1
        particle_system_path = "/World/particleSystem"
        particle_system = PhysxSchema.PhysxParticleSystem.Define(stage, particle_system_path)
        particle_system.CreateSimulationOwnerRel().SetTargets([scene.GetPath()])
        # The simulation determines the other offsets from the particle contact offset
        particle_system.CreateParticleContactOffsetAttr().Set(Particle_Contact_Offset)


        # import rigid
        usd_path = "omniverse://localhost/Users/sim/frank_gripper_rigid.usd"#usd_path = "omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Furniture/Desks/Desk_01.usd"
        rigid=BoxOnPlaneInstanced()
        rigid.create(stage,scene,self.defaultPrimPath+"/rigid",usd_path)

        # import fluid
        fluid=ParticleSamplerDemo()
        fluid.create(stage,True,Particle_Contact_Offset, scene,self.defaultPrimPath + "/fluid")

        # import cloth
        usd_path = "/home/sim/isaacgarment/ClothesNetData/ClothesNetM_usd/Dress/Long_Gallus/DLG_Dress032_1/DLG_Dress032_1_obj.usd"
        cloth=ParticleCloth()
        cloth.create(stage,scene,self.defaultPrimPath+"/cloth",usd_path=usd_path)

        # import attachment
        attach = AttachmentBlock()
        attach_position=np.array([8.78402, 72.81305, 13.25772])
        attach.create(stage,scene,self.defaultPrimPath+"/attachment","/World/cloth/plane0/mesh",attach_position)

        while simulation_app.is_running():
            simulation_app.update()

    def create_basic_scene(self):
        pass


    
    def pick_and_place_test(self):
        # self.robot.pick_and_place([np.array([0.5,0.4,0.1])],[np.array([0.5,-0.3,0.15])])
        # self.robot.pick_and_place([np.array([0.3,0.3,0.1])],[np.array([0.5,-0.3,0.1])])
        self.robot.open()
        self.robot.movep([np.array([0.3,0.3,0.11])])
        print("out")
        for i in range(100):
            self.world.step(render=True)
        
if __name__=="__main__":
    config=Config()
    env=DeformEnv(config)

    while simulation_app.is_running():
        simulation_app.update()