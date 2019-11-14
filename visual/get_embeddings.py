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
from models import VisualNet, Softmax, Trainer
import csv


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
        '--layer', type=str, help='ResNet layer [avgpool, layer4, fc]',
        default='avgpool')

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


def np_norm(np_embed_norm):
    # print('np norm')
    # print(np_embed_norm[0][0:10])
    np_embed_norm_copy = np.copy(np_embed_norm)
    np_norm_val = np.linalg.norm(np_embed_norm_copy,
                                 axis=1, keepdims=True)
    # print(np_norm_val[0])
    # print('')
    np_embed_norm_copy /= np_norm_val

    # print(np_embed_norm[0][0:10])

    # confirm unit norm
    # sq_np_embed_norm = np_embed_norm ** 2
    # print(np.sum(sq_np_embed_norm, axis=1))

    return np_embed_norm_copy


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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)

    backbone = VisualNet(dim=512, input_shape=(args.image_height, args.image_width), model_name='resnet18',
                         extract=args.layer)
    head = Softmax(dim=512, dim_out=len(
        test_dataset.label_dict), log_softmax=True)
    model = Trainer(backbone, head)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    text_file = open(args.output + "/word_embeddings.txt", "w")
    norm_file = open(args.output + "/norm_word_embeddings.txt", "w")
    print(str(len(checkpoint['vocab'])) + ' ' + '512',
          file=text_file)  # vocab size and dimension
    print(str(len(checkpoint['vocab'])) + ' ' + '512',
          file=norm_file)  # vocab size and dimension

    #print_format = ['%.18e'] * 513
    #print_format[0] = '%s'
    #text_file = open(text_file, "w", newline="")
    text_file_writer = csv.writer(text_file, delimiter=' ')
    norm_file_writer = csv.writer(norm_file, delimiter=' ')

    all_features = []
    all_labels = []

    with torch.no_grad():
        for i, (inputs, labels, text) in enumerate(test_loader):
            if i % 10 == 0:
                print('Batch %d of %d' %
                      (i + 1 % len(test_loader), len(test_loader)))

            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, embed = model(inputs, labels)

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            image_list = inputs.cpu().numpy()
            label_list = labels.cpu().numpy()
            predicted_list = predicted.cpu().numpy()
            # print(text)
            text_list = np.asarray(text)

            # for img_idx, img in enumerate(inputs):
            #     if predicted[img_idx] != labels[img_idx]:
            #         image = np.uint8(
            #             image_list[img_idx, :].squeeze().transpose((1, 2, 0)) * 255)
            #         label_name = str(
            #             test_dataset.rev_label_dict[predicted_list[img_idx].squeeze()])
            #         outpath = samples_output + '/' + label_name + \
            #             '_' + str(i) + '_' + str(img_idx) + '.png'
            #         cv2.imwrite(outpath, image)

            embed = embed.cpu().numpy()
            np_embed_norm = np_norm(embed)

            if i == 0:
                np.set_printoptions(precision=6, linewidth=120)
                for img_idx, img in enumerate(inputs):
                    id_idx = predicted_list[img_idx].squeeze()
                    val_idx = embed[img_idx][0:5]
                    norm_idx = np_embed_norm[img_idx][0:5]
                    label_idx = test_dataset.rev_label_dict[id_idx]
                    print('%s, %s - %s, %s, %s, %s' %
                          (label_list[img_idx], text[img_idx], id_idx, label_idx, val_idx, norm_idx))

                    if img_idx > 25:
                        break

            for img_idx, img in enumerate(inputs):
                text_label = text[img_idx].strip()
                text_row = [text_label]
                text_row = text_row + embed[img_idx].tolist()
                text_file_writer.writerow(text_row)

            for img_idx, img in enumerate(inputs):
                text_label = text[img_idx].strip()
                text_row = [text_label]
                text_row = text_row + np_embed_norm[img_idx].tolist()
                norm_file_writer.writerow(text_row)

            all_features.append(embed)
            all_labels.append(text_list)

            # savetxt does not work with different types (label, feature vector)
            # np.savetxt(text_file, embed_label,
            #           delimiter=" ", fmt=print_format)

    text_file.close()
    norm_file.close()

    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)
    print('feature shape {}, labels shape {}'.format(
        all_features.shape, all_labels.shape))
    np.savez_compressed(args.output + "/word_embeddings.npz",
                        features=all_features, labels=all_labels)

    accuracy = 100 * correct / (total * 1.0)
    print('Accuracy %.2f %%' % (accuracy))

    print('...complete, time {}'.format((time.clock() - start_time)))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
