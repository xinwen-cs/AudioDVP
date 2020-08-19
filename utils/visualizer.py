import torchvision
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    def __init__(self, opt):

        self.opt = opt  # cache the option
        self.port = opt.display_port
        self.writer = SummaryWriter()

    def display_current_results(self, visuals, steps):
        for label, image in visuals.items():
            self.writer.add_image(label, torchvision.utils.make_grid(image), steps)

    def plot_current_losses(self, total_iters, losses):
        """display the current losses on tensorboard display: dictionary of error labels and values
        Parameters:
            total_iters(int) -- total_iters
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        for label, loss in losses.items():
            self.writer.add_scalar(label, loss, total_iters)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (not normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, data: %.3f, comp: %.3f) ' % (epoch, iters, t_data, t_comp)
        for k, v in losses.items():
            message += '%s: %.5f ' % (k, v)

        print(message)  # print the message
