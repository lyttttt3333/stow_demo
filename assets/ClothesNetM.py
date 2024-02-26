import argparse
import asyncio
import omni
import os
from omni.isaac.kit import SimulationApp


async def convert(in_file, out_file, load_materials=False):
    # This import causes conflicts when global
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # setup converter and flags
    converter_context.ignore_materials = False
    # converter_context.ignore_animation = False
    # converter_context.ignore_cameras = True
    # converter_context.single_mesh = True
    # converter_context.smooth_normals = True
    # converter_context.preview_surface = False
    # converter_context.support_point_instancer = False
    # converter_context.embed_mdl_in_usd = False
    # converter_context.use_meter_as_world_unit = True
    # converter_context.create_world_as_default_root_prim = False
    converter_context.merge_all_meshes=True
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, progress_callback, converter_context)
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success


def asset_convert(shape_path,shape_dest_path):
        # for shape in os.listdir(type_path):
        #     shape_path=os.path.join(type_path,shape)
        #     shape_dest_path=os.path.join(type_dest_path,shape)
        #     if not os.path.exists(shape_dest_path):
        #         os.makedirs(shape_dest_path)
        #     print(f"\nConverting shape {shape}...")

            for model in os.listdir(shape_path):
                model_path=os.path.join(shape_path,model)
                model_obj_path=os.path.join(model_path,model+".obj")
                model_usd_path=os.path.join(shape_dest_path,model+".usd")
                status=asyncio.get_event_loop().run_until_complete(convert(model_obj_path,model_usd_path,True))
                if not status:
                    print(f"ERROR Status is {status}")
                print(f"---Added {model_usd_path}")



if __name__ == "__main__":
    kit = SimulationApp()

    from omni.isaac.core.utils.extensions import enable_extension

    enable_extension("omni.kit.asset_converter")

    parser = argparse.ArgumentParser("Convert OBJ/STL assets to USD")
    parser.add_argument("--folder", type=str, default="/home/isaac/garmentIsaac/ClothesNetData/ClothesNetM/Mask")
    parser.add_argument("--dest_folder", type=str, default="/home/isaac/garmentIsaac/ClothesNetData/ClothesNetMUSD/Mask")
    
    args, unknown_args = parser.parse_known_args()
    folder=args.folder
    dest_folder=args.dest_folder

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    asset_convert(folder,dest_folder)
   
    # cleanup
    kit.close()
