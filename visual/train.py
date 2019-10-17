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
from models import ResNet


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
        default=1000)
    parser.add_argument(
        "--num_workers", type=int, help="Nbr dataset workers",
        default=16)
    parser.add_argument(
        "--lr", type=float, help="learning rate",
        default=1e-3)
    parser.add_argument(
        "--min-lr", type=float, default=1e-7,
        help="Minimum learning rate for ReduceLROnPlateau")

    args = parser.parse_args(argv)
    for arg in vars(args):
        print('{} {}'.format(arg, getattr(args, arg)))

    return args


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main(args):
    start_time = time.clock()

    print('__Python VERSION:', sys.version)
    print('__PyTorch VERSION:', torch.__version__)
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Active CUDA Device: GPU', torch.cuda.current_device())
    print('__CUDA_VISIBLE_DEVICES %s \n' %
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

    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        ImageAug(),
        transforms.ToTensor(),
    ])
    train_dataset = ImageDataset(text_file_path=args.input, font_file_path=args.font,
                                 image_height=args.image_height, image_width=args.image_width,
                                 transform=train_transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)

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
                                              shuffle=False, num_workers=0)

    model = ResNet(dim=512, nbr_classes=len(
        train_dataset.label_dict), extract='avgpool').to(device)
    summary(model, input_size=(3, args.image_height, args.image_width))
    criterion = nn.NLLLoss()  # nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, min_lr=args.min_lr)

    model.train()
    valid_cnt = 0
    for epoch in range(args.epochs):
        for i, (inputs, labels) in enumerate(trainloader):

            iteration_start = time.time()

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            duration = time.time() - iteration_start
            examples_per_sec = args.batch_size / duration
            sec_per_batch = float(duration)

            if i % 10 == 0 and i > 0:
                print("Epoch: %d (%d/%d), Batch Size: %d, Loss: %f, LR: %f, ex/sec: %.1f, sec/batch: %.2f" % (
                    epoch, i + 1 % len(trainloader),
                    len(trainloader), len(inputs), loss.item(),
                    get_lr(optimizer), examples_per_sec, sec_per_batch))

            if epoch == 0 and i < 10:
                image_list = inputs.cpu().numpy()
                label_list = labels.cpu().numpy()
                for img_idx, img in enumerate(inputs):
                    image = np.uint8(
                        image_list[img_idx, :].squeeze().transpose((1, 2, 0)) * 255)
                    label_name = str(
                        train_dataset.rev_label_dict[label_list[img_idx].squeeze()])

                    #outpath = samples_train_output + '/' + label_name
                    # if not os.path.exists(outpath):
                    #    os.makedirs(outpath)
                    # outpath = samples_train_output + '/' + label_name + \
                    #    '_' + str(i) + '_' + str(img_idx) + '.png'
                    cv2.imwrite(samples_train_output + '/' + label_name +
                                '_' + str(epoch) + '_' + str(i) + '_' + str(img_idx) + '.png', image)

        if epoch % 10 == 0 and epoch > 0:
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
                for i, (inputs, labels) in enumerate(validloader):

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    if valid_cnt == 1 and i < 10:
                        image_list = inputs.cpu().numpy()
                        label_list = labels.cpu().numpy()
                        for img_idx, img in enumerate(inputs):
                            image = np.uint8(
                                image_list[img_idx, :].squeeze().transpose((1, 2, 0)) * 255)
                            label_name = str(
                                train_dataset.rev_label_dict[label_list[img_idx].squeeze()])
                            outpath = samples_valid_output + '/' + label_name + \
                                '_' + str(i) + '_' + str(img_idx) + '.png'
                            cv2.imwrite(outpath, image)

            accuracy = 100 * correct / total
            print('Epoch %d, Count %d, LRate %.4f, Accuracy %d %%' %
                  (epoch, total, get_lr(optimizer), accuracy))
            scheduler.step(accuracy)

            model.train()

    print('...complete, time {}'.format((time.clock() - start_time)))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
