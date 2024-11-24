import torch
import pytorch3d
from pytorch3d.io import load_objs_as_meshes
import os

class View:
    def __init__(self, obj_file, is_vertex_color, device, normalize_vertexes=True):
        mesh = load_objs_as_meshes([obj_file], device=device, load_textures= True,
                                    create_texture_atlas = True,
                                    texture_atlas_size = 100,
                                    texture_wrap = "repeat",)
        self.mesh = mesh
        if normalize_vertexes:
            vertexes = self.mesh.verts_packed()
            center = vertexes.mean(0); scale = max((vertexes - center).abs().max(0)[0])  
            self.mesh.offset_verts_(-center); self.mesh.scale_verts_((1.0 / float(scale)))    #recover: vertexes_normalized * scale + center
        self.lights = pytorch3d.renderer.AmbientLights(device=device)  #pytorch3d.renderer.DirectionalLights(direction=[[0.0,+3.0,0.0]], device=device)  #pytorch3d.renderer.PointLights(location=[[0.0,0.0,-3.0]], device=device)

    def look(self, distance, elevation, azimuth_all, device, image_size):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=distance, elev=elevation, azim=azimuth_all)  #distance:距离(远)  elevation:海拔(高)  azimuth:方位(平)-180,180
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)  #FoVPerspectiveCameras  FoVOrthographicCameras
        mesh_renderer = pytorch3d.renderer.MeshRenderer(rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=cameras, raster_settings=pytorch3d.renderer.RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)), shader=pytorch3d.renderer.SoftPhongShader(lights=self.lights, cameras=cameras, device=device))  #SoftPhongShader  HardPhongShader
        rendered_rgba = mesh_renderer(meshes_world=self.mesh.extend(azimuth_all.shape[0]), lights=self.lights, cameras=cameras)  #materials=materials
        #print('rendered_rgba', rendered_rgba.shape)
        #alpha_mask = (rendered_rgb[..., 3] > 0)
        #rendered_rgba = torch.cat((rendered_rgb, alpha_mask.unsqueeze(-1)), dim=-1)
        return rendered_rgba

def mesh(obj_file, is_vertex_color, image_path, elevation_number=24, azimuth_number=12, distance=2.0, image_size=512, step=4, device=['cpu','cuda'][torch.cuda.is_available()]):
    def save(rendered_rgba_all, elevation, azimuth_all, image_path):
        for rendered_rgba,azimuth in zip(rendered_rgba_all, azimuth_all):
            save_file=image_path+'/image__distance_%s__elevation_%03d__azimuth_%03d.png'%(str(distance).replace('.','_'),elevation,azimuth)
            import os; os.makedirs(os.path.dirname(save_file), exist_ok=True)
            from PIL import Image
            rendered_rgba[...,3] = (rendered_rgba[...,3]>=0.5)  #灰白/半透明->全透明
            rgba = (rendered_rgba[...,:].cpu().clamp(0.0,1.0)*255.0).numpy().astype('uint8')
            Image.frombuffer("RGBA", rgba.shape[0:2], rgba, "raw", "RGBA", 0, 1).save(save_file)
    import os
    if not os.path.exists(image_path):  
        print('render', 'todo', obj_file)
        elevation_all = torch.linspace(0, 360-(360//elevation_number), elevation_number)
        azimuth_all = torch.linspace(0, 360-(360//azimuth_number), azimuth_number)   
        view = View(obj_file=obj_file, is_vertex_color=is_vertex_color, device=device)
        for elevation in elevation_all:
            for i in range(0, len(azimuth_all), step):
                print('render','','elevation:', elevation.item(), '', 'azimuth:',azimuth_all[i:i+step])
                rendered_rgba_all = view.look(distance=distance, elevation=elevation, azimuth_all=azimuth_all[i:i+step], device=device, image_size=image_size)
                save(rendered_rgba_all=rendered_rgba_all, elevation=elevation, azimuth_all=azimuth_all[i:i+step], image_path=image_path)
    else:
        print('render', 'skip', obj_file)

import os
from glob import glob
def main():
    obj_dir = './data/mesh/resin/'  
    image_output_dir = './data/image/resin/' 
    obj_files = sorted(glob(os.path.join(obj_dir, '**', '*.obj'), recursive=True))
    print(f"Found {len(obj_files)} .obj files.")
    batch_size = 10
    for i in range(0, len(obj_files), batch_size):
        batch_files = obj_files[i:i + batch_size] 
        print(f"Processing batch {i // batch_size + 1}/{(len(obj_files) - 1) // batch_size + 1}")
        for obj_file in batch_files:
            model_name = os.path.basename(os.path.dirname(obj_file))
            image_path = os.path.join(image_output_dir, model_name, 'images_mesh/')
            mesh(
                obj_file=obj_file,
                is_vertex_color=0,
                image_path=image_path
            )
if __name__ == '__main__':
    main()
