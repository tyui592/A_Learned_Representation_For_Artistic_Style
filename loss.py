"""Loss Function Code."""

import torch
from torch.nn.functional import mse_loss


def calc_content_loss(features, targets, nodes):
    """Calculate Content Loss."""
    content_loss = 0
    for node in nodes:
        content_loss += mse_loss(features[node], targets[node])
    return content_loss


def gram(x):
    """Transfer a feature to gram matrix."""
    b, c, h, w = x.size()
    f = x.flatten(2)
    g = torch.bmm(f, f.transpose(1, 2))
    return g.div(h*w)


def calc_style_loss(features, targets, nodes):
    """Calcuate Gram Loss."""
    gram_loss = 0
    for node in nodes:
        gram_loss += mse_loss(gram(features[node]), gram(targets[node]))
    return gram_loss


def calc_tv_loss(x):
    """Calc Total Variation Loss."""
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss
