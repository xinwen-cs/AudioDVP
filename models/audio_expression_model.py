import os
from collections import OrderedDict

import torch
import torch.nn as nn
from . import networks


class AudioExpressionModel:
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.isTrain = opt.isTrain

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['Delta']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = []

        # define networks (both generator and discriminator)
        self.net = networks.AudioExpressionModule(opt).to(self.device)

        if self.isTrain:
            # define loss functions
            self.criterionDelta = nn.MSELoss() if self.opt.lambda_delta > 0.0 else None

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        if self.opt.isTrain:
            self.alpha = input['alpha'].to(self.device)
            self.beta = input['beta'].to(self.device)
            self.gamma = input['gamma'].to(self.device)
            self.rotation = input['rotation'].to(self.device)
            self.translation = input['translation'].to(self.device)
            self.delta_gt = input['delta_gt'].to(self.device)

        self.feature = input['feature'].to(self.device)
        self.filename = input['filename']

    def forward(self):
        self.delta = self.net(self.feature)

    def backward(self):
        self.loss_Delta = self.criterionDelta(self.delta, self.delta_gt) if self.opt.lambda_delta > 0.0 else 0.0

        # combine loss and calculate gradients
        self.loss = self.opt.lambda_delta * self.loss_Delta

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

    def eval(self):
        """Make models eval mode during test time"""
        self.net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_delta(self):
        torch.save(self.delta[0], os.path.join(self.opt.data_dir, 'reenact_delta', self.filename[0]))

    def update_learning_rate(self):
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_network(self):
        save_path = os.path.join(self.opt.net_dir, 'delta_net.pth')
        torch.save(self.net.cpu().state_dict(), save_path)

    def load_network(self):
        load_path = os.path.join(self.opt.net_dir, 'delta_net.pth')
        state_dict = torch.load(load_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
