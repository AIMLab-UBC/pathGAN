from PIL import Image
import utils.utils as utils
import numpy as np
import torch
import os
import h5py


class BaseModel():
    def name(self):
        n = [self.model_name_prefix, self.deep_classifier, self.deep_model, self.optim, 'lr' +
             str(self.lr), 'bs' + str(self.batch_size), 'e' + str(self.epoch)]
        if self.n_eval_samples != 100:
            n += ['neval'+str(self.n_eval_samples)]
        if self.use_pretrained:
            n += ['pw']
        if self.use_equalized_batch:
            n += ['eb']
        if self.l2_decay != 0:
            n += ['l2' + str(self.l2_decay)]
        return '_'.join(n).lower()

    def __init__(self, config):
        self.config = config
        self.deep_classifier = config.deep_classifier
        self.deep_model = config.deep_model
        self.model_name_prefix = config.model_name_prefix
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.epoch = config.epoch
        self.patch_size = config.patch_size
        self.n_subtypes = config.n_subtypes
        self.save_dir = config.save_dir
        self.n_eval_samples = config.n_eval_samples
        self.use_pretrained = config.use_pretrained
        self.l2_decay = config.l2_decay
        self.continue_train = config.continue_train
        self.log_patches = config.log_patches
        self.optim = config.optim
        self.load_pretrained = config.load_pretrained
        self.use_equalized_batch = config.use_equalized_batch
        self.test_augmentation = config.test_augmentation
        self.load_model_id = config.load_model_id
        self.is_tcga = config.is_tcga

        # store evaluation data labels
        self.eval_data_labels = []
        # load GAN augmentation dataset
        if self.test_augmentation:
            self.aug_target = h5py.File(os.path.join(
                config.dataset_dir, config.aug_target_file_name), 'r')
            self.aug_real = h5py.File(os.path.join(
                config.dataset_dir, config.aug_real_file_name), 'r')
            self.aug_fake = h5py.File(os.path.join(
                config.dataset_dir, config.aug_fake_file_name), 'r')
        else:
            self.eval_images = h5py.File(os.path.join(
                config.dataset_dir, config.preload_image_file_name), 'r')

    def eval(self, eval_data_ids):
        pass

    def load_state(self, model_id, load_pretrained=False):
        pass

    def save_state(self, model_id):
        pass

    def get_current_errors(self):
        return self.loss.item()

    def optimize_parameters(self, logits, labels):
        pass

    def forward(self):
        pass
