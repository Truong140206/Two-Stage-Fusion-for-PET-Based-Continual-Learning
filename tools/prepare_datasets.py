#!/usr/bin/env python3
"""Prepare ImageNet-A and 5-Datasets for the reported experiments.

Why a tool rather than a few lines inside the training command. Both have a
first-use side effect that is destructive and easy to get wrong, and both are
worth failing fast on before any GPU time is spent.

The ImageNet-A split. Imagenet_A.__init__ splits 80/20 with
torch.utils.data.random_split over the flat list of all 7,500 images, then
moves the files into imagenet-a/train and imagenet-a/test. That draw is not
stratified by class, and ImageNet-A has classes with only a handful of images,
so classes land in train with nothing at all in test -- eight of them on the
first attempt here. Modern torchvision's ImageFolder refuses a class directory
with no files, so the dataset then cannot be constructed at all. ImageNet-R
runs the identical code without trouble only because it has 30,000 images over
the same 200 classes.

So the split is done here instead, per class and with a fixed seed, and the
directories are left in place; Imagenet_A.__init__ checks for train/ and test/
and skips its own split when they exist, so the dataset class is untouched.
Two departures from the released behaviour, both deliberate:

  * Stratified. Every class is represented in both halves, so every class is
    actually evaluated. Under the released draw the eight empty classes would
    have contributed to training and then never been tested.
  * Seeded. The released split is unseeded, so it cannot be reproduced even by
    its own authors. Seeding it means a later rebuild of this directory gives
    the same split, and any number measured against it stays meaningful.

Usage:
    python tools/prepare_datasets.py --which all
    python tools/prepare_datasets.py --which imagenet-a --root /path/to/datasets
"""
import argparse
import os
import random
import shutil
import sys
import tarfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

IMAGENET_A_TAR = 'imagenet-a.tar'
IMAGENET_A_URL = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar'
IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff',
             '.webp')
TRAIN_FRACTION = 0.8
SPLIT_SEED = 42


def default_root():
    """Same convention as training_scripts/train_any_4090.sh."""
    return os.path.join(os.path.dirname(REPO_ROOT), 'datasets')


def stratified_split(fpath, seed=SPLIT_SEED):
    """Move each class's images into train/ and test/, 80/20 within the class.

    Returns (n_train, n_test, smallest_class) or None if the layout is wrong.
    """
    classes = sorted(d for d in os.listdir(fpath)
                     if os.path.isdir(os.path.join(fpath, d))
                     and d not in ('train', 'test'))
    if len(classes) < 2:
        print('  KHONG DUNG: mong doi hang tram thu muc lop trong %s' % fpath)
        return None

    train_root = os.path.join(fpath, 'train')
    test_root = os.path.join(fpath, 'test')
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    rng = random.Random(seed)
    n_train = n_test = 0
    smallest = None
    singletons = []

    for name in classes:
        class_dir = os.path.join(fpath, name)
        files = sorted(f for f in os.listdir(class_dir)
                       if f.lower().endswith(IMAGE_EXT))
        if smallest is None or len(files) < smallest[1]:
            smallest = (name, len(files))
        if len(files) < 2:
            singletons.append((name, len(files)))

        rng.shuffle(files)
        n_keep = int(round(TRAIN_FRACTION * len(files)))
        # Both halves must be non-empty or ImageFolder rejects the class.
        n_keep = max(1, min(n_keep, len(files) - 1)) if len(files) >= 2 else len(files)

        for target_root, chosen in ((train_root, files[:n_keep]),
                                    (test_root, files[n_keep:])):
            target_dir = os.path.join(target_root, name)
            os.makedirs(target_dir, exist_ok=True)
            for filename in chosen:
                shutil.move(os.path.join(class_dir, filename),
                            os.path.join(target_dir, filename))

        n_train += n_keep
        n_test += len(files) - n_keep
        os.rmdir(class_dir)

    if singletons:
        print('  CANH BAO: %d lop chi co duoi 2 anh, khong the co mat o ca hai '
              'nua: %s' % (len(singletons), singletons[:5]))
    print('  lop nho nhat: %s co %d anh' % smallest)
    return n_train, n_test, smallest


def prepare_imagenet_a(root):
    from continual_datasets.continual_datasets import Imagenet_A

    fpath = os.path.join(root, 'imagenet-a')
    train_dir = os.path.join(fpath, 'train')
    test_dir = os.path.join(fpath, 'test')

    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        empty = [d for d in sorted(os.listdir(test_dir))
                 if not os.listdir(os.path.join(test_dir, d))]
        if empty:
            print('  DA CHIA NHUNG HONG: %d lop rong trong test/ (%s ...)'
                  % (len(empty), ', '.join(empty[:4])))
            print('  Xoa %s rooi dung lai tu parquet, roi chay lai lenh nay.'
                  % fpath)
            return False
        print('  da chia tu truoc, khong dung vao')
    else:
        if not os.path.isdir(fpath):
            tar_path = os.path.join(root, IMAGENET_A_TAR)
            if not os.path.isfile(tar_path):
                print('  THIEU %s' % tar_path)
                print('  Tai bang:  wget -c -P %s %s' % (root, IMAGENET_A_URL))
                print('  Hoac dung tools/imagenet_a_from_parquet.py neu may '
                      'khong toi duoc may chu do.')
                return False
            print('  giai nen %s ...' % tar_path)
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(root)

        n_img = sum(len(f) for _, _, f in os.walk(fpath))
        print('  truoc khi chia: %d tep' % n_img)
        print('  chia 80/20 theo tung lop, seed %d (di chuyen tep) ...'
              % SPLIT_SEED)
        result = stratified_split(fpath)
        if result is None:
            return False
        print('  da chia: %d train, %d test' % result[:2])

    train_set = Imagenet_A(root, train=True, download=True).data
    val_set = Imagenet_A(root, train=False, download=True).data
    print('  ImageNet-A: %d lop, %d anh train, %d anh test'
          % (len(val_set.classes), len(train_set), len(val_set)))
    return True


def prepare_five(root):
    from torchvision import datasets as tvd
    from continual_datasets.continual_datasets import (
        MNIST_RGB, FashionMNIST, NotMNIST, SVHN)

    # Exactly the five names datasets.py uses for '5-datasets', and exactly the
    # constructor calls get_dataset makes for each.
    jobs = [
        ('SVHN', lambda tr: SVHN(root, split='train' if tr else 'test',
                                 download=True)),
        ('MNIST', lambda tr: MNIST_RGB(root, train=tr, download=True)),
        ('CIFAR10', lambda tr: tvd.CIFAR10(root, train=tr, download=True)),
        ('NotMNIST', lambda tr: NotMNIST(root, train=tr, download=True)),
        ('FashionMNIST', lambda tr: FashionMNIST(root, train=tr, download=True)),
    ]

    ok = True
    for name, build in jobs:
        try:
            train_set, val_set = build(True), build(False)
            n_tr = len(getattr(train_set, 'data', train_set))
            n_te = len(getattr(val_set, 'data', val_set))
            print('  %-13s %6d train, %6d test' % (name, n_tr, n_te))
        except Exception as exc:                      # noqa: BLE001
            print('  %-13s HONG: %s: %s' % (name, type(exc).__name__, exc))
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None,
                        help='thu muc du lieu (mac dinh: canh repo)')
    parser.add_argument('--which', default='all',
                        choices=['all', 'imagenet-a', 'five'])
    args = parser.parse_args()

    root = args.root or default_root()
    os.makedirs(root, exist_ok=True)
    print('thu muc du lieu: %s\n' % root)

    ok = True
    if args.which in ('all', 'imagenet-a'):
        print('ImageNet-A')
        ok = prepare_imagenet_a(root) and ok
        print()
    if args.which in ('all', 'five'):
        print('5-Datasets')
        ok = prepare_five(root) and ok
        print()

    print('SAN SANG' if ok else 'CHUA XONG -- doc thong bao ben tren')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
