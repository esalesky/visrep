""" Train a word embedding model."""
import os
import argparse
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from torchvision.utils import save_image
from torchsummary import summary
from augment import ImageAug
from dataset import ImageDataset
from models import VisualNet, Softmax, VisualTrainer
import torch.nn.functional as F
import logging

LOG = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input', type=str, help='Input seed text',
        default='')
    parser.add_argument(
        '--font', type=str, help='Input train font file',
        default='')
    parser.add_argument(
        '--valid_font', type=str, help='Input validation font file',
        default='')
    parser.add_argument(
        '--output', type=str, help='Output directory',
        default='')
    parser.add_argument(
        '--layer', type=str, help='ResNet layer [avgpool, layer4, fc]',
        default='avgpool')

    parser.add_argument(
        '--augment', action='store_true',
        help='train with augmentation')
    parser.add_argument(
        "--image-height", type=int, help="Image height",
        default=32)
    parser.add_argument(
        "--image-width", type=int, help="Image width",
        default=128)
    parser.add_argument(
        "--batch-size", type=int, help="Mini-batch size",
        default=128)
    parser.add_argument(
        "--epochs", type=int, help="Nbr epochs",
        default=150)
    parser.add_argument(
        "--num_workers", type=int, help="Nbr dataset workers",
        default=8)
    parser.add_argument(
        "--lr", type=float, help="learning rate",
        default=1e-3)
    parser.add_argument(
        "--min-lr", type=float, default=1e-7,
        help="Minimum learning rate for ReduceLROnPlateau")

    args = parser.parse_args(argv)
    for arg in vars(args):
        LOG.info('%s %s', arg, getattr(args, arg))

    return args


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main(args):
    start_time = time.clock()

    LOG.info('__Python VERSION: %s', sys.version)
    LOG.info('__PyTorch VERSION: %s', torch.__version__)
    LOG.info('__CUDNN VERSION: %s', torch.backends.cudnn.version())
    LOG.info('__Number CUDA Devices: %s', torch.cuda.device_count())
    LOG.info('__Active CUDA Device: GPU %s', torch.cuda.current_device())
    LOG.info('__CUDA_VISIBLE_DEVICES %s ',
             str(os.environ["CUDA_VISIBLE_DEVICES"]))

    ckpts_output = args.output + '/checkpoints'
    if not os.path.exists(ckpts_output):
        os.makedirs(ckpts_output)

    samples_train_output = args.output + '/samples_train'
    if not os.path.exists(samples_train_output):
        os.makedirs(samples_train_output)

    samples_valid_output = args.output + '/samples_valid'
    if not os.path.exists(samples_valid_output):
        os.makedirs(samples_valid_output)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.augment:
        train_transform = transforms.Compose([
            ImageAug(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    train_dataset = ImageDataset(text_file_path=args.input, font_file_path=args.font,
                                 image_height=args.image_height, image_width=args.image_width,
                                 transform=train_transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers,
                                              pin_memory=True)

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    valid_dataset = ImageDataset(text_file_path=args.input, font_file_path=args.valid_font,
                                 image_height=args.image_height, image_width=args.image_width,
                                 default_image=True, transform=valid_transform,
                                 label_dict=train_dataset.label_dict,
                                 rev_label_dict=train_dataset.rev_label_dict)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0,
                                              pin_memory=True)

    backbone = VisualNet(dim=512, input_shape=(args.image_height, args.image_width), model_name='resnet18',
                         extract=args.layer, normalize=False)
    head = Softmax(dim=512, dim_out=len(
        train_dataset.label_dict), log_softmax=False)
    model = VisualTrainer(backbone, head)
    model.to(device)

    last_checkpoint = os.path.join(args.output, 'checkpoints/model.pth')
    LOG.info('...searching for %s', last_checkpoint)
    if os.path.isfile(last_checkpoint):
        checkpoint = torch.load(last_checkpoint)
        LOG.info('Loading checkpoint...')
        LOG.info(' epoch %d', checkpoint['epoch'])
        LOG.info(' loss %f', checkpoint['loss'])
        LOG.info(' len vocab %s', len(checkpoint['vocab']))
        LOG.info(' len rev_vocab %s', len(checkpoint['rev_vocab']))
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    summary(backbone, input_size=(3, args.image_height, args.image_width))
    criterion = nn.NLLLoss()  # nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, min_lr=args.min_lr, verbose=True)

    model.train()
    valid_cnt = 0
    for epoch in range(args.epochs):
        for i, (inputs, labels, text) in enumerate(trainloader):

            iteration_start = time.time()

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            embed, prelogits = model(inputs)
            logits = F.log_softmax(prelogits, dim=-1)

            # LOG.info('Embed %s, logits %s, and labels %s',
            #         embed.shape, logits.shape, labels.shape)

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            duration = time.time() - iteration_start
            examples_per_sec = args.batch_size / duration
            sec_per_batch = float(duration)

            if i % 10 == 0 and i > 0:
                LOG.info("Epoch: %d (%d/%d), Batch Size: %d, Loss: %.4f, LR: %.8f, ex/sec: %.1f, sec/batch: %.2f",
                         epoch, i + 1 % len(trainloader),
                         len(trainloader), len(inputs), loss.item(),
                         get_lr(optimizer), examples_per_sec, sec_per_batch)

            if epoch == 0 and i < 10:
                image_list = inputs.cpu().numpy()
                label_list = labels.cpu().numpy()
                for img_idx, img in enumerate(inputs):
                    label_name = str(
                        train_dataset.rev_label_dict[label_list[img_idx].squeeze()])

                    image = image_list[img_idx, :].squeeze()
                    image = np.uint8(image.transpose((1, 2, 0)) * 255)
                    image = image[:, :, ::-1].copy()  # rgb to bgr

                    outpath = samples_train_output + '/' + label_name + \
                        '_' + str(epoch) + '_' + str(i) + \
                        '_' + str(img_idx) + '.png'
                    cv2.imwrite(outpath, image)

        if epoch % 10 == 0:
            valid_cnt += 1
            torch.save({'epoch': epoch,
                        'loss': loss.item(),
                        'vocab': train_dataset.label_dict,
                        'rev_vocab': train_dataset.rev_label_dict,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        },
                       os.path.join(ckpts_output, 'model.pth'))

            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for i, (inputs, labels, text) in enumerate(validloader):

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # outputs = model(inputs)
                    #logits, embed = model(inputs)
                    embed, prelogits = model(inputs)
                    logits = F.log_softmax(prelogits, dim=-1)

                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    if valid_cnt == 1 and i < 10:
                        image_list = inputs.cpu().numpy()
                        label_list = labels.cpu().numpy()
                        for img_idx, img in enumerate(inputs):
                            label_name = str(
                                train_dataset.rev_label_dict[label_list[img_idx].squeeze()])

                            image = image_list[img_idx, :].squeeze()
                            image = np.uint8(image.transpose((1, 2, 0)) * 255)
                            image = image[:, :, ::-1].copy()  # rgb to bgr

                            outpath = samples_valid_output + '/' + label_name + \
                                '_' + str(epoch) + '_' + str(i) + \
                                '_' + str(img_idx) + '.png'
                            cv2.imwrite(outpath, image)

            accuracy = 100 * correct / (total * 1.0)
            LOG.info('Epoch %d, Count %d, LRate %.8f, Accuracy %.2f',
                     epoch, total, get_lr(optimizer), accuracy)

            model.train()

            scheduler.step(accuracy)

    LOG.info('...complete, time %s', time.clock() - start_time)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
