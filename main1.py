import torch
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop, ToTensor)

from configs.parser import parse_args, load_config
from dataset import Video
from model import generate_model_PTV
from train import train, val, test

def main():
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    cfg = load_config(args, args.cfg_files)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spatial_transform = Compose([
        Resize(cfg.DATA.TRAIN_CROP_SIZE),
        CenterCrop(cfg.DATA.TRAIN_CROP_SIZE),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = Video(
        root_path=cfg.DATA.DATA_PATH, 
        flag='train',
        spatial_transform=spatial_transform, 
        downsample_rate=cfg.DATA.SAMPLING_RATE,
        sample_duration=cfg.DATA.NUM_FRAMES
        )
    test_dataset = Video(
        root_path=cfg.DATA.DATA_PATH, 
        flag='test', 
        spatial_transform=spatial_transform,
        downsample_rate=cfg.DATA.SAMPLING_RATE,
        sample_duration=cfg.DATA.NUM_FRAMES
        )

    print("train dataset length:", len(train_dataset))
    print("test dataset length:", len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=True, 
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
        train_epoch = cfg.SOLVER.MAX_EPOCH
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.SOLVER.BASE_LR, 
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
            )
        train(
            model,
            criterion,
            optimizer,
            train_loader,
            test_loader,
            train_epoch, 
            device, 
            cfg
            )

if __name__ == "__main__":
    main()