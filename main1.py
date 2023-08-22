import torch
from spatial_transforms import (Compose, Scale, Normalize, Resize, CenterCrop, ToTensor)

from configs.parser import parse_args, load_config
from dataset import Video
from model import generate_model_PTV
from train import train, val, test
from lr_schedulers import make_scheduler

from torch.utils.data.sampler import WeightedRandomSampler


def main():
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    cfg = load_config(args, args.cfg_files)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_dict = {'dive': 0, 'walk': 1, 'observe':2, 'work': 3, 'ascend': 4, 'off': 5, 'other': 6}

    spatial_transform = Compose([
        Scale(cfg.DATA.TRAIN_CROP_SIZE),
        CenterCrop(cfg.DATA.TRAIN_CROP_SIZE),
        ToTensor(),
        Normalize(cfg.DATA.MEAN, cfg.DATA.STD)
    ])

    train_dataset = Video(
        root_path=cfg.DATA.DATA_PATH,
        label_dict=label_dict,
        flag='train',
        spatial_transform=spatial_transform, 
        downsample_rate=cfg.DATA.SAMPLING_RATE,
        sample_duration=cfg.DATA.NUM_FRAMES
        )
    test_dataset = Video(
        root_path=cfg.DATA.DATA_PATH,
        label_dict=label_dict,
        flag='test',
        spatial_transform=spatial_transform,
        downsample_rate=cfg.DATA.SAMPLING_RATE,
        sample_duration=cfg.DATA.NUM_FRAMES
        )
    
    # sample_class = []
    # num_classes = [0]*len(label_dict)
    # for _, each_target in train_dataset:
    #     num_classes[int(each_target)] += 1
    #     sample_class.append(int(each_target))
    # weight = 1. / num_classes.float()
    # print("class weight:", weight)
    # samples_weight = torch.tensor([weight[t] for t in sample_class])
    # print("sample length:", len(samples_weight))
    # sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    print("train dataset length:", len(train_dataset))
    print("test dataset length:", len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=True, 
        # sampler=sampler,
        num_workers=cfg.TRAIN.NUM_WORKERS, 
        pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS, 
        pin_memory=False
    )

    model = generate_model_PTV(cfg)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=None)  # 默认用全部的GPU

    if cfg.TRAIN.ENABLE:
        train_epoch = cfg.SOLVER.EPOCHS + cfg.SOLVER.WARMUP_EPOCHS if cfg.SOLVER.WARMUP else cfg.SOLVER.EPOCHS
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.SOLVER.BASE_LR, 
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
            )
        num_iters_per_epoch = len(train_loader)
        scheduler = make_scheduler(optimizer, cfg, num_iters_per_epoch, last_epoch=-1)
        train(
            model,
            criterion,
            optimizer,
            scheduler,
            train_loader,
            test_loader,
            train_epoch, 
            device, 
            cfg,
            label_dict
            )


if __name__ == "__main__":
    main()
