import os
from collections import OrderedDict
import torch
from torchvision import utils

from . import networks
from utils.util import create_dir


class ResnetModel:
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.isTrain = opt.isTrain

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['Photometric', 'Landmark', 'Alpha', 'Beta', 'Delta']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['gt', 'render', 'masked_gt', 'overlay']

        # define networks (both generator and discriminator)
        self.net = networks.ResnetFaceModelOptimizer(opt)

        if self.isTrain:
            # define loss functions
            self.criterionPhotometric = networks.PhotometricLoss() if self.opt.lambda_photo > 0.0 else None
            self.criterionLandmark = networks.LandmarkLoss(opt) if self.opt.lambda_land > 0.0 else None
            self.regularizationAlpha = networks.CoefficientRegularization()
            self.regularizationBeta = networks.CoefficientRegularization()
            self.regularizationDelta = networks.CoefficientRegularization()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam([{'params': self.net.fc.parameters()}, {'params': self.net.pretrained_model.parameters()},
                                               {'params': self.net.alpha, 'lr': 1e-3}, {'params': self.net.beta, 'lr': 1e-3}],
                                               lr=opt.lr, betas=(0.5, 0.999)
            )

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.input = input['input'].to(self.device)
        self.gt = input['gt'].to(self.device)
        self.landmark_gt = input['landmark_gt'].to(self.device)
        self.image_name = input['image_name']

    def forward(self):
        self.alpha, self.delta, self.beta, self.gamma, self.rotation, self.translation, self.render, self.mask, self.landmark = self.net(self.input)
        self.masked_gt = self.gt * self.mask
        self.overlay = self.render + (1 - self.mask) * self.gt

    def backward(self):
        self.loss_Photometric = self.criterionPhotometric(self.render, self.masked_gt) if self.opt.lambda_photo > 0.0 else 0.0
        self.loss_Landmark = self.criterionLandmark(self.landmark, self.landmark_gt) if self.opt.lambda_land > 0.0 else 0.0

        self.loss_Alpha = self.regularizationAlpha(self.alpha)
        self.loss_Beta = self.regularizationBeta(self.beta)
        self.loss_Delta = self.regularizationDelta(self.delta)

        # combine loss and calculate gradients
        self.loss = self.opt.lambda_photo * self.loss_Photometric \
            + self.opt.lambda_reg * \
            (self.opt.lambda_alpha * self.loss_Alpha + self.opt.lambda_beta * self.loss_Beta + self.opt.lambda_delta * self.loss_Delta) \
            + self.opt.lambda_land * self.loss_Landmark

        self.loss.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_result(self):
        """Save 3DMM coef and image"""
        create_dir(os.path.join(self.opt.data_dir, 'render'))
        create_dir(os.path.join(self.opt.data_dir, 'overlay'))
        create_dir(os.path.join(self.opt.data_dir, 'alpha'))
        create_dir(os.path.join(self.opt.data_dir, 'beta'))
        create_dir(os.path.join(self.opt.data_dir, 'delta'))
        create_dir(os.path.join(self.opt.data_dir, 'gamma'))
        create_dir(os.path.join(self.opt.data_dir, 'rotation'))
        create_dir(os.path.join(self.opt.data_dir, 'translation'))

        for i in range(self.opt.batch_size):
            utils.save_image(self.render[i], os.path.join(self.opt.data_dir, 'render', self.image_name[i]))
            utils.save_image(self.overlay[i], os.path.join(self.opt.data_dir, 'overlay', self.image_name[i]))

            torch.save(self.alpha[0].detach().cpu(), os.path.join(self.opt.data_dir, 'alpha', self.image_name[i][:-4]+'.pt'))
            torch.save(self.beta[0].detach().cpu(), os.path.join(self.opt.data_dir, 'beta', self.image_name[i][:-4]+'.pt'))
            torch.save(self.delta[i].detach().cpu(), os.path.join(self.opt.data_dir, 'delta', self.image_name[i][:-4]+'.pt'))
            torch.save(self.gamma[i].detach().cpu(), os.path.join(self.opt.data_dir, 'gamma', self.image_name[i][:-4]+'.pt'))
            torch.save(self.rotation[i].detach().cpu(), os.path.join(self.opt.data_dir, 'rotation', self.image_name[i][:-4]+'.pt'))
            torch.save(self.translation[i].detach().cpu(), os.path.join(self.opt.data_dir, 'translation', self.image_name[i][:-4]+'.pt'))
