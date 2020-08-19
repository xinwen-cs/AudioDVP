import time

from options.options import Options
from models import audio_expression_model
from datasets import create_dataset
from utils.visualizer import Visualizer


if __name__ == '__main__':
    opt = Options().parse_args()   # get training options

    dataset = create_dataset(opt)

    model = audio_expression_model.AudioExpressionModel(opt)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    total_iters = 0

    for epoch in range(opt.num_epoch):

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                visualizer.display_current_results(model.get_current_visuals(), total_iters)

            if total_iters % opt.print_freq == 0:    # print training losses
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.num_epoch, time.time() - epoch_start_time))

    model.save_network()
