"""Utility Code."""

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

normalize = T.Normalize(mean=MEAN, std=STD)
denormalize = T.Normalize(mean=[-m/s for m, s in zip(MEAN, STD)],
                          std=[1/std for std in STD])


def get_transforms(imsize=None, cropsize=None, cencrop=False):
    """Get the transforms."""
    transformer = []
    if imsize:
        transformer.append(T.Resize(imsize))
    if cropsize:
        if cencrop:
            transformer.append(T.CenterCrop(cropsize))
        else:
            transformer.append(T.RandomCrop(cropsize))

    transformer.append(T.ToTensor())
    transformer.append(normalize)
    return T.Compose(transformer)


def imload(path, imsize=None, cropsize=None, cencrop=False):
    """Load a image."""
    transformer = get_transforms(imsize=imsize,
                                 cropsize=cropsize,
                                 cencrop=cencrop)
    image = Image.open(path).convert("RGB")
    return transformer(image).unsqueeze(0)


def imsave(image, save_path):
    """Save a image."""
    image = denormalize(torchvision.utils.make_grid(image)).clamp_(0.0, 1.0)
    torchvision.utils.save_image(image, save_path)
    return None


class ImageDataset:
    """Image Dataset."""

    def __init__(self, dir_path):
        """Init."""
        self.images = sorted(list(dir_path.glob('*.jpg')))

    def __len__(self):
        """Return the Number of data sampels."""
        return len(self.images)

    def __getitem__(self, index):
        """Get Image and Index."""
        img = Image.open(self.images[index]).convert('RGB')
        return img, index


class DataProcessor:
    """Data Processor."""

    def __init__(self, imsize=256, cropsize=240, cencrop=False):
        """Init."""
        self.transforms = get_transforms(imsize=imsize,
                                         cropsize=cropsize,
                                         cencrop=cencrop)

    def __call__(self, batch):
        """Process the batch."""
        images, indices = list(zip(*batch))

        inputs = torch.stack([self.transforms(image) for image in images])
        return inputs, indices
