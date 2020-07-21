from models.base_model import BaseModel
from PIL import Image
import utils.utils as utils
import data.preprocess as preprocess
import numpy as np
import torch
import torchvision
import os


class DeepModel(BaseModel):
    def __init__(self, config, is_eval=False):
        super().__init__(config)
        self.is_eval = is_eval

        if 'vgg' in self.deep_classifier:
            model = getattr(torchvision.models, self.deep_classifier)
            model = model(pretrained=self.use_pretrained)
            model.classifier._modules['6'] = torch.nn.Linear(
                4096, self.n_subtypes)
        else:
            raise NotImplementedError

        self.model = model.cuda()
        if not self.is_eval:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.l2_decay)
        else:
            raise NotImplementedError

        if self.continue_train:
            self.load_state(config.load_model_id,
                            load_pretrained=self.load_pretrained)

        if self.is_eval:
            self.load_state(config.load_model_id)
            self.model.eval()

    def forward(self, x):
        logits = self.model.forward(x)
        probs = torch.softmax(logits, dim=1)
        return logits, probs

    def optimize_parameters(self, logits, labels):
        self.loss = self.criterion(logits.type(
            torch.float), labels.type(torch.long))
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save_state(self, model_id):
        filename = '{}_{}.pth'.format(self.name(), str(model_id))
        save_path = os.path.join(self.save_dir, filename)
        state = {
            'iter_idx': model_id,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, save_path)

    def load_state(self, model_id, load_pretrained=False):
        filename = '{}_{}.pth'.format(self.name(), str(model_id))
        save_path = os.path.join(self.save_dir, filename)

        if torch.cuda.is_available():
            state = torch.load(save_path)
        else:
            state = torch.load(save_path, map_location='cpu')

        try:
            self.model.load_state_dict(state['state_dict'])
        except RuntimeError:
            pretrained_dict = state['state_dict']
            model_dict = self.model.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.model.load_state_dict(pretrained_dict)

        self.optimizer.load_state_dict(state['optimizer'])

        model_id = state['iter_idx']
        return model_id

    def eval(self, eval_data_ids):
        self.model.eval()
        # set the number of evaluation samples
        if self.n_eval_samples == -1:
            # -1 means eval model on all validation set
            cur_eval_ids = eval_data_ids
        else:
            # convert to numpy array for fast indexing
            eval_data_ids = np.asarray(eval_data_ids)
            # start to extract patch label beforehand
            if len(self.eval_data_labels) == 0:
                for eval_data_id in eval_data_ids:
                    self.eval_data_labels += [
                        utils.get_label_by_patch_id(eval_data_id, is_tcga=self.is_tcga)]
                # convert to numpy array for fast indexing
                self.eval_data_labels = np.asarray(self.eval_data_labels)
            # store the evaluation patch ids
            cur_eval_ids = []
            # compute the number of patches per subtype for evaluation
            per_subtype_samples = self.n_eval_samples // self.n_subtypes
            # select patch ids by subtype
            for subtype in range(self.n_subtypes):
                # numpy advanced indexing
                cur_subtype_idx = self.eval_data_labels == subtype
                # randomly pick patch ids
                try:
                    # pick without replacement
                    cur_eval_ids += np.random.choice(
                        eval_data_ids[cur_subtype_idx], per_subtype_samples, replace=False).tolist()
                except ValueError:
                    # if has less samples in a subtype, pick with replacement
                    cur_eval_ids += np.random.choice(
                        eval_data_ids[cur_subtype_idx], per_subtype_samples).tolist()
        # evaluation during training
        n_correct = 0

        for cur_eval_id in cur_eval_ids:
            gt_label = utils.get_label_by_patch_id(
                cur_eval_id, is_tcga=self.is_tcga)

            if self.test_augmentation:
                cur_image = self.aug_target[cur_eval_id]['image_data'][()]
            else:
                cur_image = self.eval_images[cur_eval_id]['image_data'][()]

            cur_image = Image.fromarray(cur_image)
            cur_tensor = preprocess.raw(
                cur_image, is_eval=True, apply_color_jitter=False)

            with torch.no_grad():
                _, prob = self.forward(cur_tensor)
                pred_label = torch.argmax(prob).item()
            if pred_label == gt_label:
                n_correct = n_correct + 1

        self.model.train()
        return n_correct / len(cur_eval_ids)
