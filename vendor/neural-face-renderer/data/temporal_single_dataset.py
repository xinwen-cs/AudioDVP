import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch

class TemporalSingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating Reenact results only for one side with the model option '-model temporal_test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.Nw = opt.Nw

        self.A_Images = []

        for i in range(len(self.A_paths)):
            self.A_Images.append(Image.open(self.A_paths[i]).convert('RGB'))
            transform_params = get_params(self.opt, self.A_Images[i].size)
            self.A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
            self.A_Images[i] = self.A_transform(self.A_Images[i])

        self.image_shape = self.A_Images[0].shape

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, A_paths
            A (tensor) - - an image in the input domain (which consists of Nw temporal pics eg.[1-11])
            A_paths (str) - - image paths
        """
        list_A = []

        if index < self.Nw // 2:
            for i in range(self.Nw // 2 - index):
                list_A.append(torch.zeros(*self.image_shape, dtype=torch.float32))

            for i in range(index + self.Nw // 2 + 1):
                list_A.append(self.A_Images[i])
        elif index > len(self) - self.Nw // 2 - 1:
            for i in range(index - self.Nw // 2, len(self)):
                list_A.append(self.A_Images[i])

            for i in range(index + self.Nw // 2 - len(self) + 1):
                list_A.append(torch.zeros(*self.image_shape, dtype=torch.float32))
        else:
            for i in range(index - self.Nw // 2, index + self.Nw // 2 + 1):
                list_A.append(self.A_Images[i])

        A = torch.cat(list_A, dim=0)

        A_path = self.A_paths[index]

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
