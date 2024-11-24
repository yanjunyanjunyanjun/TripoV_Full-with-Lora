import torch
import os
class SuperFocus:
    def __init__(self, device):
        from carvekit.api.high import HiInterface  #pip install carvekit --extra-index-url https://download.pytorch.org/whl/cpu  #cu117
        self.hiInterface = HiInterface(object_type=['object','hairs-like'][0],  batch_size_seg=1, batch_size_matting=1, seg_mask_size=640, matting_mask_size=2048, trimap_prob_threshold=231, trimap_dilation=30, trimap_erosion_iters=5, fp16=False, device=device)
    @torch.no_grad()
    def __call__(self, image):
        image = self.hiInterface([image])[0]
        return image
def focus(image_source_path, image_target_path):
    import os
    #if os.path.exists(image_target_path): return False
    super_focus = SuperFocus(device=['cpu','cuda'][torch.cuda.is_available()])
    image_todo = sorted(os.listdir(image_source_path))
    for index, image_file in enumerate(image_todo):
        image_save = os.path.join(image_target_path, image_file[0:-len(image_file.split('.')[-1])-1]+'.png')
        if not os.path.exists(image_save):
            image_full = os.path.join(image_source_path, image_file)
            image_rgba = super_focus(image_full)  # [H, W, 4]
            os.makedirs(os.path.dirname(image_save), exist_ok=True)
            image_rgba.save(image_save)
        print('focus:', '%08d/%08d'%(index, len(image_todo)), image_save)
    return True

def get_subfolders(path):
    try:
        subfolders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        print("Subfolders:", subfolders)
        return subfolders
    except FileNotFoundError:
        print(f"Path not found: {path}")
        return []
def main(folders:list):
    for obj in folders:
        focus(image_source_path='./data/image/resin/'+obj+'/images_mesh/', image_target_path='./data/image/resin/'+obj+'/images_focus/')

if __name__ == '__main__':
    base_path = './data/image/resin/'
    subfolders = get_subfolders(base_path)
    main(subfolders)

