import os
import torch

from utils import util
from datasets.base_dataset import BaseDataset


class AudioExpressionDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.data_dir = opt.data_dir
        self.Nw = opt.Nw

        self.feature_list = util.load_coef(os.path.join(self.data_dir, 'feature'))
        self.filenames = util.get_file_list(os.path.join(self.data_dir, 'feature'))

        if opt.isTrain:
            self.alpha_list = util.load_coef(os.path.join(self.data_dir, 'alpha'))
            self.beta_list = util.load_coef(os.path.join(self.data_dir, 'beta'))
            self.delta_list = util.load_coef(os.path.join(self.data_dir, 'delta'))
            self.gamma_list = util.load_coef(os.path.join(self.data_dir, 'gamma'))
            self.rotation_list = util.load_coef(os.path.join(self.data_dir, 'rotation'))
            self.translation_list = util.load_coef(os.path.join(self.data_dir, 'translation'))

    def __len__(self):
        return len(self.feature_list)

    def __getitem__(self, index):
        feature_list = []

        if index < self.Nw // 2:
            for i in range(self.Nw // 2 - index):
                feature_list.append(torch.zeros(256, dtype=torch.float32))

            for i in range(index + self.Nw // 2 + 1):
                feature_list.append(self.feature_list[i])
        elif index > len(self) - self.Nw // 2 - 1:
            for i in range(index - self.Nw // 2, len(self)):
                feature_list.append(self.feature_list[i])

            for i in range(index + self.Nw // 2 - len(self) + 1):
                feature_list.append(torch.zeros(256, dtype=torch.float32))
        else:
            for i in range(index - self.Nw // 2, index + self.Nw // 2 + 1):
                feature_list.append(self.feature_list[i])

        feature = torch.stack(feature_list, dim=0)

        filename = os.path.basename(self.filenames[index])

        if not self.opt.isTrain:
            return {'feature': feature, 'filename': filename}
        else:
            alpha = self.alpha_list[index]
            beta = self.beta_list[index]
            delta = self.delta_list[index]
            gamma = self.gamma_list[index]
            rotation = self.rotation_list[index]
            translation = self.translation_list[index]

            return {
                'feature': feature, 'filename': filename,
                'alpha': alpha, 'beta': beta, 'delta_gt': delta,
                'gamma': gamma, 'translation': translation, 'rotation': rotation
            }
