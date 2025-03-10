def split(image_path_source, image_path_target, train_percent):
    import os
    import random
    import shutil

    print('split ...')
    sub_sets = ['train','valid']
    for sub_set in sub_sets:
        sub_path = os.path.join(image_path_target, sub_set)
        shutil.rmtree(sub_path, ignore_errors=True)
        os.makedirs(sub_path, exist_ok=True)
    random.seed(122333)
    unit = 10000
    for one_file in os.listdir(image_path_source):
        if os.path.isfile(os.path.join(image_path_source, one_file)): 
            sub_path = sub_sets[0] if random.randrange(0,unit)< train_percent*unit else sub_sets[1]      
            shutil.copy(os.path.join(image_path_source, one_file), os.path.join(image_path_target, sub_path, one_file))

import os
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
        split(image_path_source='./data/image/resin/'+obj+'/images_focus/', image_path_target='./data/image/resin/'+obj+'/images_split/', train_percent=1)
    print('splitting is done')

if __name__ == '__main__':    #python -Bu superv_01_data_03_split.py
    base_path = './data/image/resin/'
    subfolders = get_subfolders(base_path)
    main(subfolders)

