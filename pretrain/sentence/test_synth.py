""" Train a sentence embedding model."""
import os
import argparse
import sys
import time
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from ocr_dataset import LineSynthDataset, ImageGroupSampler, image_collater
import tqdm
from imgaug import augmenters as iaa


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--test', type=str,
                        default='', help='Input test seed text')

    parser.add_argument('--test-font', type=str,
                        default='', help='Input test font file')

    parser.add_argument('--test-background', type=str,
                        default='', help='Input test background file')

    parser.add_argument('--output', type=str,
                        default='', help='Output directory')

    parser.add_argument('--augment', action='store_true',
                        help='train with augmentation')

    parser.add_argument('--write-image', action='store_true',
                        help='write images')

    parser.add_argument("--image-height", type=int,
                        default=30, help="Image height")

    parser.add_argument('--pad-image', action='store_true',
                        help='Pad images instead of resize')

    parser.add_argument('--use-image-cache', action='store_true',
                        help='Cache images')
    parser.add_argument("--max-image-cache", type=int,
                        default=250000, help="max image cache")
    parser.add_argument("--max-cache-write", type=int,
                        default=5000, help="max cache write")

    parser.add_argument("--max-seed", type=int,
                        default=500000, help="max seed")
    parser.add_argument("--num-workers", type=int,
                        default=8, help="Nbr dataset workers")

    parser.add_argument('--image-verbose', action='store_true',
                        help='Debug info')

    parser.add_argument('--use-default-image', action='store_true',
                        help='Use default image ')

    parser.add_argument("--batch-size", type=int,
                        default=8, help="Mini-batch size")
    parser.add_argument("--epochs", type=int,
                        default=150, help="Nbr epochs")

    args = parser.parse_args(argv)
    for arg in vars(args):
        print('%s %s' % (arg, getattr(args, arg)))

    return args


def image_augment(line_image, line_height, test_image_output, seed_text):
    height, width, channels = line_image.shape

    resize_line_image = cv2.resize(line_image,
                                   (int(width * 2.0), line_height), interpolation=cv2.INTER_AREA)
    outpath = test_image_output + '/' + seed_text + '_resize' + '.png'
    cv2.imwrite(outpath, resize_line_image)

    squeeze_line_image = cv2.resize(line_image,
                                    (int(width * 0.5), line_height), interpolation=cv2.INTER_AREA)
    outpath = test_image_output + '/' + seed_text + '_squeeze' + '.png'
    cv2.imwrite(outpath, squeeze_line_image)

    gaussian_line_image = iaa.imgcorruptlike.GaussianBlur(
        severity=1)(image=line_image)
    outpath = test_image_output + '/' + seed_text + '_gaussian' + '.png'
    cv2.imwrite(outpath, gaussian_line_image)

    elastic_line_image = iaa.imgcorruptlike.ElasticTransform(
        severity=3)(image=line_image)
    outpath = test_image_output + '/' + seed_text + '_elastic' + '.png'
    cv2.imwrite(outpath, elastic_line_image)

    salt_line_image = iaa.SaltAndPepper(.10)(image=line_image)
    outpath = test_image_output + '/' + seed_text + '_salt' + '.png'
    cv2.imwrite(outpath, salt_line_image)

    compress_line_image = iaa.imgcorruptlike.JpegCompression(
        severity=3)(image=line_image)
    outpath = test_image_output + '/' + seed_text + '_compress' + '.png'
    cv2.imwrite(outpath, compress_line_image)

    pixelate_line_image = iaa.imgcorruptlike.Pixelate(
        severity=3)(image=line_image)
    outpath = test_image_output + '/' + seed_text + '_pixelate' + '.png'
    cv2.imwrite(outpath, pixelate_line_image)

    pad_line_image = iaa.Pad(
        percent=(.05), sample_independently=True, pad_mode='constant', pad_cval=255)(image=line_image)
    outpath = test_image_output + '/' + seed_text + '_pad' + '.png'
    cv2.imwrite(outpath, pad_line_image)

    crop_line_image = iaa.Crop(
        percent=(.01), sample_independently=True)(image=line_image)
    outpath = test_image_output + '/' + seed_text + '_crop' + '.png'
    cv2.imwrite(outpath, crop_line_image)


def main(args):
    start_time = time.clock()

    print('__Python VERSION: %s' % sys.version)
    print('__PyTorch VERSION: %s' % torch.__version__)
    print('__CUDNN VERSION: %s' % torch.backends.cudnn.version())
    print('__Number CUDA Devices: %s' % torch.cuda.device_count())
    print('__Active CUDA Device: GPU %s' % torch.cuda.current_device())
    print('__CUDA_VISIBLE_DEVICES %s ' %
          str(os.environ["CUDA_VISIBLE_DEVICES"]))

    test_image_output = args.output + '/images'
    if not os.path.exists(test_image_output):
        os.makedirs(test_image_output)

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    test_dataset = LineSynthDataset(text_file_path=args.test,
                                    font_file_path=args.test_font,
                                    bkg_file_path=args.test_background,
                                    image_height=args.image_height,
                                    enable_image_pad=args.pad_image,
                                    transform=test_transform,
                                    use_default_image=args.use_default_image,
                                    max_seed=args.max_seed,
                                    image_verbose=args.image_verbose)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              sampler=ImageGroupSampler(
                                                  test_dataset, rand=False),
                                              collate_fn=lambda b: image_collater(
                                                  b, args.image_verbose),
                                              num_workers=args.num_workers, pin_memory=True)

    for batch, sample in tqdm.tqdm(
            enumerate(test_loader), total=len(test_loader), leave=False):

        image_list = sample['net_input']['src_tokens'].cpu().numpy()
        label_list = sample['seed_text']

        if args.write_image:
            for img_idx, img in enumerate(image_list):

                seed_text = label_list[img_idx]

                image = np.uint8(img.transpose((1, 2, 0)) * 255)
                line_image = image[:, :, ::-1].copy()  # rgb to bgr

                if args.augment:
                    image_augment(line_image, args.image_height,
                                  test_image_output, seed_text)

                outpath = test_image_output + '/' + seed_text + '.png'
                cv2.imwrite(outpath, image)

    print('...complete, time %s' % (time.clock() - start_time))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
