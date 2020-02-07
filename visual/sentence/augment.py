from imgaug import augmenters as iaa


class ImageAug(object):

    def __init__(self):
        def sometimes(aug): return iaa.Sometimes(.50, aug)
        seq = iaa.Sequential(
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0.25, 1.0)),  # blur images with a sigma
                    # randomly remove up to n% of the pixels
                    iaa.Dropout((0.01, 0.05), per_channel=0.5),
                    iaa.CropAndPad(
                        percent=(-0.05, 0.05),
                        pad_mode=["constant"],
                        pad_cval=255
                    ),
                    iaa.Affine(
                        shear=(-2, 2),
                    ),
                    iaa.Affine(
                        rotate=(-2, 2),
                    )
                ]),

            ],
            random_order=True
        )
        self.seq = seq

    def __call__(self, img):
        images_aug = self.seq.augment_image(img)
        return images_aug
