import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler
from .data_zoo import RSNAData, RSNA_MEAN, RSNA_STD

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    # dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(key='train', config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset ({len(dataset_train)} images)")
    dataset_val, _ = build_dataset(key='val', config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset ({len(dataset_val)} images)")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        collate_fn=med_collate_fn if config.MI.MULTI_CROP else None,
    )

    if config.DATA.DATASET in ['rsna']:
        dataset_test, _ = build_dataset(key='test', config=config)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test) if config.TEST.SEQUENTIAL else torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=config.TEST.SHUFFLE
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False,
            collate_fn=med_collate_fn if config.MI.MULTI_CROP else None,
        )
        print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build test dataset ({len(dataset_test)} images)")
    else:
        dataset_test = None
        data_loader_test = None
        print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} test dataset not found!!!")

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn


def med_collate_fn(batch_list):
    assert type(batch_list) == list, f"Error"
    batch_size = len(batch_list)
    #data = torch.cat([item[0] for item in batch_list]).reshape(batch_size, -1)
    data = [item[0] for item in batch_list]
    labels = torch.from_numpy(np.concatenate([item[1] for item in batch_list])).reshape(batch_size, -1)
    return data, labels

def med_multi_crop_transform(img_size, interpolation, mean, std, is_no_crop=False):
    t = []
    size = int(img_size) if is_no_crop else int(1.3 * img_size)
    #size = int(1.3 * img_size)
    t.append(transforms.Resize(size, interpolation=_pil_interp(interpolation)),
                # to maintain same ratio w.r.t. 224 images
            )
    if not is_no_crop:
        t.append(transforms.RandomCrop(img_size, padding=4))
    #.append(transforms.RandomHorizontalFlip(p=0.5))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_dataset(key, config):
    mean = RSNA_MEAN if config.DATA.DATASET == 'rsna' else IMAGENET_DEFAULT_MEAN
    std = RSNA_STD if config.DATA.DATASET == 'rsna' else IMAGENET_DEFAULT_STD
    mean = config.MI.IMAGE_MEAN if config.MI.IMAGE_MEAN is not None else mean
    std = config.MI.IMAGE_STD if config.MI.IMAGE_STD is not None else std
    transform = build_transform(key=='train', config, mean, std)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if key=='train' else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if key=='train' else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'rsna':
        dataset = RSNAData(config.DATA.DATA_PATH, transform=transform, key=key, seperator=config.MI.LABEL_SEPERATOR, gender_filter=config.MI.GENDER_FILTER)
        nb_classes = -1
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
            mean=mean,
            std=std,
        )
        if config.MI.PAD_TO_SIZE:
            resize = [transforms.Resize(config.DATA.IMG_SIZE-1, interpolation=_pil_interp(config.DATA.INTERPOLATION), max_size=config.DATA.IMG_SIZE)]
            transform.transforms[0] = transforms.CenterCrop(config.DATA.IMG_SIZE)
            transform.transforms = resize + transform.transforms
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.MI.PAD_TO_SIZE:
            t.append(transforms.Resize(config.DATA.IMG_SIZE-1, interpolation=_pil_interp(config.DATA.INTERPOLATION), max_size=config.DATA.IMG_SIZE))
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        elif config.MI.MULTI_CROP:
            return None
        elif config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
