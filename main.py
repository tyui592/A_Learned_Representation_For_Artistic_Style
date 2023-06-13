"""Pytorch Implementation Code.

Reference: 'A Learned Representation for Artistic Style'
"""

import torch
import argparse
from pathlib import Path
from torch.optim import Adam
from network import StyleTransferNetwork
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from utils import ImageDataset, DataProcessor, imsave, imload
from loss import calc_content_loss, calc_style_loss, calc_tv_loss

NUM_STYLE = 16


def train(style_path, content_path,
          style_weight=5.0, tv_weight=1e-5,
          lr=1e-4, batch_size=8, iterations=40_000):
    """Train Network."""
    content_nodes = ['relu_3_3']
    style_nodes = ['relu_1_2', 'relu_2_2', 'relu_3_3', 'relu_4_2']
    return_nodes = {3: 'relu_1_2',
                    8: 'relu_2_2',
                    15: 'relu_3_3',
                    22: 'relu_4_2'}
    device = torch.device('cuda')

    # data
    content_dataset = ImageDataset(dir_path=Path(content_path))
    style_dataset = ImageDataset(dir_path=Path(style_path))

    data_processor = DataProcessor(imsize=256,
                                   cropsize=240,
                                   cencrop=False)
    content_dataloader = DataLoader(dataset=content_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    collate_fn=data_processor)
    style_dataloader = DataLoader(dataset=style_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=data_processor)

    # loss network
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    for param in vgg.parameters():
        param.requires_grad = False
    loss_network = create_feature_extractor(vgg, return_nodes).to(device)

    # network
    model = StyleTransferNetwork()
    model.train()
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    losses = {'content': [], 'style': [], 'tv': [], 'total': []}
    print("Start training...")
    for i in range(1, 1+iterations):
        content_images, _ = next(iter(content_dataloader))
        style_images, style_indices = next(iter(style_dataloader))

        style_codes = torch.zeros(batch_size, NUM_STYLE, 1)
        for b, s in enumerate(style_indices):
            style_codes[b, s] = 1

        content_images = content_images.to(device)
        style_images = style_images.to(device)
        style_codes = style_codes.to(device)

        output_images = model(content_images, style_codes)

        content_features = loss_network(content_images)
        style_features = loss_network(style_images)
        output_features = loss_network(output_images)

        style_loss = calc_style_loss(output_features,
                                     style_features,
                                     style_nodes)
        content_loss = calc_content_loss(output_features,
                                         content_features,
                                         content_nodes)
        tv_loss = calc_tv_loss(output_images)

        total_loss = content_loss \
            + style_loss * style_weight \
            + tv_loss * tv_weight

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses['content'].append(content_loss.item())
        losses['style'].append(style_loss.item())
        losses['tv'].append(tv_loss.item())
        losses['total'].append(total_loss.item())

        if i % 100 == 0:
            log = f"iter.: {i}"
            for k, v in losses.items():
                # calcuate a recent average value
                avg = sum(v[-50:]) / 50
                log += f", {k}: {avg:1.4f}"
            print(log)

    torch.save({"state_dict": model.state_dict()}, "model.ckpt")


def evaluate(content_path, style_index):
    """Evaluate the network."""
    device = torch.device('cpu')
    ckpt = torch.load('model.ckpt', map_location=device)

    model = StyleTransferNetwork()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    content_image = imload(args.content_path, imsize=256)
    # for all styles
    if style_index == -1:
        style_code = torch.eye(NUM_STYLE).unsqueeze(-1)
        content_image = content_image.repeat(NUM_STYLE, 1, 1, 1)

    # for specific style
    elif style_index in range(NUM_STYLE):
        style_code = torch.zeros(1, NUM_STYLE, 1)
        style_code[:, style_index, :] = 1

    else:
        raise RuntimeError("Not expected style index")

    stylized_image = model(content_image, style_code)
    imsave(stylized_image, 'stylized_images.jpg')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train',
                        help="'train' | 'eval'")
    parser.add_argument('--style_path', type=str, default=None,
                        help="Path of style image.")
    parser.add_argument('--content_path', type=str, default=None,
                        help="Path of content image.")
    parser.add_argument('--style_index', type=int, default=0,
                        help="Index for stylization, -1: all styles.")

    args = parser.parse_args()

    if args.mode == 'train':
        train(args.style_path, args.content_path)

    elif args.mode == 'eval':
        evaluate(args.content_path, args.style_index)

    else:
        raise RuntimeError("Not exepcted mode")
