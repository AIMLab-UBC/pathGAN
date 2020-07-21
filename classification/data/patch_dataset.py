from data.base_dataset import BaseDataset
from torchvision import transforms
from PIL import Image
import utils.utils as utils
import data.preprocess as preprocess
import models.models as models
import numpy as np
import os
import torch
import random
import h5py


class PatchDataset(BaseDataset):
    def __init__(self, config, apply_color_jitter=True):
        super().__init__(config)
        self.apply_color_jitter = apply_color_jitter

    def __getitem__(self, index):
        if self.use_equalized_batch and not self.is_eval:
            sample_label = index % self.n_subtypes
            label_idx = self.label_list == sample_label
            patch_id = np.random.choice(self.cur_data_ids[label_idx])
        else:
            patch_id = self.cur_data_ids[index]

        cur_label = utils.get_label_by_patch_id(
            patch_id, is_tcga=self.is_tcga)

        if self.test_augmentation:
            if patch_id in self.aug_target:
                cur_image = self.aug_target[patch_id]['image_data'][()]
            elif patch_id in self.aug_real:
                cur_image = self.aug_real[patch_id]['image_data'][()]
            elif patch_id in self.aug_fake:
                cur_image = self.aug_fake[patch_id]['image_data'][()]
        else:
            cur_image = self.preload_images[patch_id]['image_data'][()]
            cur_image = Image.fromarray(cur_image)

        cur_tensor = preprocess.raw(
            cur_image, apply_color_jitter=self.apply_color_jitter)

        cur_label = torch.tensor(cur_label).type(torch.LongTensor).cuda()

        return cur_tensor, cur_label, patch_id
