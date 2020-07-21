from torch.utils.data import Dataset
import numpy as np
import utils.utils as utils
import h5py
import torch
import torchvision
import os


class BaseDataset(Dataset):
    def name(self):
        return 'base_dataset'

    def __init__(self, config):
        self.config = config
        self.mode = config.mode
        self.preload_image_file_name = config.preload_image_file_name
        self.is_eval = config.mode != 'Training'
        self.deep_classifier = config.deep_classifier
        self.n_eval_samples = config.n_eval_samples
        self.log_patches = config.log_patches
        self.n_subtypes = config.n_subtypes
        self.use_equalized_batch = config.use_equalized_batch
        self.test_augmentation = config.test_augmentation
        self.is_tcga = config.is_tcga
        self.preload_images = h5py.File(os.path.join(
            config.dataset_dir, config.preload_image_file_name), 'r')

        if config.mode == 'Testing':
            # Testing mode is used for evaluating model in the test dataset
            self.cur_data_ids = utils.read_data_ids(
                os.path.join(config.dataset_dir, config.test_ids_file_name))
        elif config.mode == 'Validation':
            # Validation mode is used for evaluating model in the validation dataset
            self.cur_data_ids = utils.read_data_ids(
                os.path.join(config.dataset_dir, config.val_ids_file_name))
        else:
            # Training mode is used for training model in the training dataset
            self.cur_data_ids = utils.read_data_ids(
                os.path.join(config.dataset_dir, config.train_ids_file_name))
            # also pass the validation set to evaluate model during training
            self.eval_data_ids = utils.read_data_ids(
                os.path.join(config.dataset_dir, config.val_ids_file_name))

        if self.test_augmentation:
            # evaluate GAN augmentation
            self.aug_target = h5py.File(os.path.join(
                config.dataset_dir, config.aug_target_file_name), 'r')
            self.aug_real = h5py.File(os.path.join(
                config.dataset_dir, config.aug_real_file_name), 'r')
            self.aug_fake = h5py.File(os.path.join(
                config.dataset_dir, config.aug_fake_file_name), 'r')

        if self.use_equalized_batch:
            # extract patch label beforehand
            self.label_list = []
            for cur_data_id in self.cur_data_ids:
                cur_label = utils.get_label_by_patch_id(
                    cur_data_id, is_tcga=self.is_tcga)
                self.label_list += [cur_label]
            # for fast numpy indexing
            self.label_list = np.asarray(self.label_list)
            self.cur_data_ids = np.asarray(self.cur_data_ids)

    def __len__(self):
        return len(self.cur_data_ids)

    def __getitem__(self, index):
        pass
