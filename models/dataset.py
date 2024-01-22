import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import Resize as VisionResize
import torch
from torch import Tensor
import torch.nn.functional 
from torch.utils.data import random_split, TensorDataset, DataLoader, Dataset
import h5py


class SupResDataset(Dataset):
    """
    Make the pairs of LR, HR img pairs
    """
    def __init__(self, imgs, lr_size, hr_size, transform=None):
        """
        imgs: original img in shape of (N,C,H,W)
        lr_size: size of low resolution
        hr_size: size of high resolution
        transform: operations of data augmentation, before resizing to HR and LR
        """
        self.imgs = imgs
        self.transform = transform
        self.lr_resize = VisionResize(lr_size, antialias=False)
        self.hr_resize = VisionResize(hr_size, antialias=False)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        LR_img = self.lr_resize(img)
        HR_img = self.hr_resize(img)
            
        return LR_img, HR_img


class Translate(object):
    """
    Apply translation to the input image
    """
    def __init__(self, ndim):
        self.ndim = ndim
        
    def __call__(self, sample):
        in_img = sample 
        
        shift_dims = tuple(np.arange(self.ndim)-self.ndim)
        shift_pixels = tuple([torch.randint(in_img.shape[d], (1,)).item() for d in shift_dims])

        in_img = torch.roll(in_img, shift_pixels, dims=shift_dims)
        
        return in_img
    

class Flip(object):
    """
    Flip the input images 
    """
    def __init__(self, ndim):
        self.axes = None
        self.ndim = ndim

    def __call__(self, sample):
        assert self.ndim > 1, "flipping is ambiguous for 1D scalars/vectors"

        self.axes = torch.randint(2, (self.ndim,), dtype=torch.bool)
        self.axes = torch.arange(self.ndim)[self.axes]

        in_img = sample

        if in_img.shape[0] == self.ndim:  # flip vector components
            in_img[self.axes] = -in_img[self.axes]

        shifted_axes = (1 + self.axes).tolist()
        in_img = torch.flip(in_img, shifted_axes)

        return in_img
    

class Permutate(object):
    """
    Permutate the input images 
    """
    def __init__(self, ndim):
        self.axes = None
        self.ndim = ndim

    def __call__(self, sample):
        assert self.ndim > 1, "permutation is not necessary for 1D fields"

        self.axes = torch.randperm(self.ndim)
        
        in_img = sample

        if in_img.shape[0] == self.ndim:  # permutate vector components
            in_img = in_img[self.axes]

        shifted_axes = [0] + (1 + self.axes).tolist()
        in_img = in_img.permute(shifted_axes)

        return in_img