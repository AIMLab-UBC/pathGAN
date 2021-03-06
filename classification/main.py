#!/usr/bin/env python3

from config import get_config, print_usage
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import models.models as models
import numpy as np
import utils.utils as utils
import data.patch_dataset as patch_dataset
import data.postprocess as postprocess
import torch
import os


def evaluate(config):
    patches = patch_dataset.PatchDataset(config, apply_color_jitter=False)

    print('{} mode and there are {} patches...'.format(
        config.mode, str(len(patches))))

    data_loader = torch.utils.data.DataLoader(
        patches,
        batch_size=config.batch_size)

    model = models.DeepModel(config, is_eval=True)
    prefix = config.mode + ' Epoch: '

    pred_labels = []
    gt_labels = []
    pred_probs = np.array([]).reshape(0, config.n_subtypes)
    for data in tqdm(data_loader, desc=prefix, dynamic_ncols=True):
        cur_data, cur_label, patch_infos = data
        with torch.no_grad():
            _, pred_prob = model.forward(cur_data)
            pred_labels += torch.argmax(pred_prob,
                                        dim=1).cpu().numpy().tolist()
            gt_labels += cur_label.cpu().numpy().tolist()
            pred_prob = pred_prob.cpu().numpy()
            pred_probs = np.vstack((pred_probs, pred_prob))

    utils.compute_metric(gt_labels, pred_labels, pred_probs, verbose=True)


def train(config):
    patches = patch_dataset.PatchDataset(config, apply_color_jitter=True)

    print('{} mode and there are {} patches...'.format(
        config.mode, str(len(patches))))

    data_loader = torch.utils.data.DataLoader(
        patches,
        batch_size=config.batch_size)

    model = models.DeepModel(config, is_eval=False)

    writer = SummaryWriter(log_dir=os.path.join(config.log_dir, model.name()))
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    iter_idx = -1
    max_val_acc = float('-inf')
    intv_loss = 0
    for epoch in range(config.epoch):
        prefix = 'Training Epoch {:3d}: '.format(epoch)
        for data in tqdm(data_loader, desc=prefix):
            iter_idx += 1

            train_data, train_labels, orig_patch = data

            pred_labels_logits, pred_labels_probs = model.forward(
                train_data)
            model.optimize_parameters(pred_labels_logits, train_labels)

            intv_loss = intv_loss + model.get_current_errors()

            if iter_idx % config.rep_intv == 0:
                writer.add_scalar('Training CrossEntropyLoss', intv_loss / config.rep_intv,
                                  global_step=iter_idx)
                intv_loss = 0

                val_acc = model.eval(eval_data_ids=patches.eval_data_ids)

                if max_val_acc < val_acc:
                    max_val_acc = val_acc
                    model.save_state(model_id='max_val_acc')

                writer.add_scalar('Validation Accuracy',
                                  val_acc, global_step=iter_idx)

                if config.log_patches:
                    concat_patches = postprocess.hori_concat_img(
                        orig_patch.numpy().transpose(0, 3, 1, 2))
                    writer.add_images('Patches', concat_patches,
                                      global_step=iter_idx, dataformats='HWC')


def main(config):
    utils.set_gpus(n_gpus=1)
    if config.mode == 'Training':
        train(config)
    elif config.mode == 'Validation':
        evaluate(config)
    elif config.mode == 'Testing':
        evaluate(config)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    config, unparsed = get_config()

    if len(unparsed) > 0:
        print(unparsed)
        print_usage()
        exit(1)

    main(config)
