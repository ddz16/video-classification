import torch
from spatial_transforms import (Compose, Scale, Normalize, Resize, CenterCrop, ToTensor)

from configs.parser import parse_args, load_config
from dataset import ExtractFeatureVideo
from model import generate_model_PTV
import numpy as np


def main():
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    cfg = load_config(args, args.cfg_files)

    cfg.FEATURES.EXTRACT = True

    spatial_transform = Compose([
        Resize((cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE)),
        # CenterCrop(cfg.DATA.TRAIN_CROP_SIZE),
        ToTensor(),
        Normalize(cfg.DATA.MEAN, cfg.DATA.STD)
    ])

    my_dataset = ExtractFeatureVideo(
        cfg.FEATURES.FRAMES_PATH,
        spatial_transform,
        sample_duration=cfg.DATA.NUM_FRAMES,
        stride=cfg.FEATURES.STRIDE
        )
    print("feature length:", len(my_dataset))  # L
    my_loader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=False
    )

    model = generate_model_PTV(cfg)
    model = model.cuda()
    model.eval()

    all_features_list = []

    for i, clips in enumerate(my_loader):
        clips = clips.cuda()
        features = model(clips)  # B*D
        all_features_list.append(features)

    all_features = torch.cat(all_features_list, dim=0)  # L*D
    assert all_features.shape[0] == len(my_dataset)
    assert all_features.shape[1] == 2048

    all_features_npy = all_features.cpu().numpy()
    save_name = cfg.FEATURES.FRAMES_PATH.split('/')[1] + '.npy'
    np.save(save_name, all_features_npy)
