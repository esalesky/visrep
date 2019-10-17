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
        '--model-path', type=str, help='Input model path',
        default='')
    parser.add_argument(
        '--input', type=str, help='Input seed text',
        default='')
    parser.add_argument(
        '--font', type=str, help='Input train font file',
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
        "--num_workers", type=int, help="Nbr dataset workers",
        default=16)

    args = parser.parse_args(argv)
    for arg in vars(args):
        print('{} {}'.format(arg, getattr(args, arg)))

    return args


def main(args):
    start_time = time.clock()

    print('__Python VERSION:', sys.version)
    print('__PyTorch VERSION:', torch.__version__)
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Active CUDA Device: GPU', torch.cuda.current_device())
    print('__CUDA_VISIBLE_DEVICES %s \n' %
          str(os.environ["CUDA_VISIBLE_DEVICES"]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path)
    print('Loading checkpoint...')
    print(' epoch %d' % checkpoint['epoch'])
    print(' loss %f' % checkpoint['loss'])
    print(' len vocab %s' % len(checkpoint['vocab']))
    print(' len rev_vocab %s' % len(checkpoint['rev_vocab']))

    samples_output = args.output + '/wrong'
    if not os.path.exists(samples_output):
        os.makedirs(samples_output)

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    test_dataset = ImageDataset(text_file_path=args.input, font_file_path=args.font,
                                image_height=args.image_height, image_width=args.image_width,
                                default_image=True, transform=test_transform,
                                label_dict=checkpoint['vocab'],
                                rev_label_dict=checkpoint['rev_vocab'])
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)

    model = ResNet(dim=512, nbr_classes=len(
        test_dataset.label_dict), extract='avgpool', log_softmax=False).to(device)
    # test_dataset.label_dict), extract='avgpool', log_softmax=True).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    #correct = 0
    #total = 0

    text_file = open(args.output + "/word_embeddings.txt", "w")
    print(str(len(checkpoint['vocab'])) + ' ' + '512',
          file=text_file)  # vocab size and dimension

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = outputs.cpu().numpy()

            np.savetxt(text_file, outputs, delimiter=" ")

            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

            # image_list = inputs.cpu().numpy()
            # label_list = labels.cpu().numpy()
            # predicted_list = predicted.cpu().numpy()
            # for img_idx, img in enumerate(inputs):
            #     if predicted[img_idx] != labels[img_idx]:
            #         image = np.uint8(
            #             image_list[img_idx, :].squeeze().transpose((1, 2, 0)) * 255)
            #         label_name = str(
            #             test_dataset.rev_label_dict[predicted_list[img_idx].squeeze()])
            #         outpath = samples_output + '/' + label_name + \
            #             '_' + str(i) + '_' + str(img_idx) + '.png'
            #         cv2.imwrite(outpath, image)

    text_file.close()

    #accuracy = 100 * correct / total
    #print('Accuracy %d %%' % (accuracy))

    print('...complete, time {}'.format((time.clock() - start_time)))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
