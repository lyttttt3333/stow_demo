from pxr import Usd, UsdPhysics, UsdShade, UsdGeom
from omni.physx.scripts import physicsUtils
from omni.physx.scripts.utils import setRigidBody


def set_collision(stage, prim : Usd.Prim, approximationShape:str):
    collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath())
    if not collision_api:
        collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    
    collision_api.CreateApproximationAttr().Set(approximationShape)


def set_mass(stage, prim: Usd.Prim, mass:float):
    mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
    if not mass_api:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(mass)
    else:
        mass_api.GetMassAttr().Set(mass)



