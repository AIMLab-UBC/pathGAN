import h5py
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm


def read_data_ids(data_id_path):
    with open(data_id_path) as f:
        data_ids = f.readlines()
        data_ids = [x.strip() for x in data_ids]
    return data_ids


def unify_ids(patch_ids):
    unified_ids = {}
    for patch_id in patch_ids:
        if 'TCGA_Fake_1024_Downsampled' in patch_id:
            unified_ids[patch_id] = unify_fake_id(patch_id)
        elif 'TCGA_Training_1024_Downsampled' in patch_id:
            unified_ids[patch_id] = unify_train_id(patch_id)
        elif 'TCGA_Val_Test_1024_Downampled' in patch_id:
            unified_ids[patch_id] = unify_val_test_id(patch_id)
        else:
            raise NotImplementedError
    return unified_ids


def unify_train_id(train_id):
    train_id_no_ext = train_id[:-4]
    info = train_id_no_ext.split('/')[-2:]
    patch_id_info = info[-1].split('-')
    patch_idx = patch_id_info[-1]
    patch_id_info = '-'.join(patch_id_info[:-1])
    unified_id = '/'.join([info[0], patch_id_info, patch_idx])
    return unified_id


def unify_val_test_id(val_id):
    val_id_no_ext = val_id[:-4]
    info = val_id_no_ext.split('/')[-4:]
    info = [info[0], info[1], info[3]]
    unified_id = '/'.join(info)
    return unified_id


def unify_fake_id(fake_id):
    fake_id_no_ext = fake_id[:-4]
    info = fake_id_no_ext.split('/')[-3:]
    info[-2] = 'fake_wsi'
    unified_id = '/'.join(info)
    return unified_id


# fake_images = glob.glob('/projects/ovcare/classification/TCGA_Fake_1024_Downsampled/**/imgs/**')
# train_images = glob.glob('/projects/ovcare/classification/TCGA_Training_1024_Downsampled/**/*.png')
# val_test_images = glob.glob('/projects/ovcare/classification/TCGA_Val_Test_1024_Downampled/**/**/Tumor/*.png')
id_files = glob.glob(
    '/projects/ovcare/classification/ywang/gan_tcga_dataset/patch_ids/*.txt')
patch_ids = []

for id_file in id_files:
    patch_ids += read_data_ids(id_file)

patch_ids = set(patch_ids)
patch_ids = list(patch_ids)

unified_ids = unify_ids(patch_ids)

with h5py.File('/projects/ovcare/classification/ywang/gan_tcga_dataset/images.h5', 'w') as f:
    prefix = 'Storing Images: '
    for patch_id in tqdm(patch_ids, desc=prefix, dynamic_ncols=True):
        cur_image = Image.open(patch_id).convert('RGB')
        image_grp = f.require_group(unified_ids[patch_id])
        image_grp.create_dataset(
            'image_data', data=np.asarray(cur_image))
