import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import copy
import json
from torchvision import get_image_backend
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)



def read_json(path):
    # 读取JSON文件内容
    with open(path, 'r') as file:
        data = json.load(file)

    # 解析JSON数据
    file_name = list(data.keys())[0]
    fps = data[file_name]['fps']
    duration = data[file_name]['duration']
    annotations = data[file_name]['annotation']
    
    labels = []
    segments = []

    # 输出解析结果
    print("-" * 60)
    print("文件名:", file_name)
    print("帧率:", fps)
    print("持续时间:", duration)

    print("注释信息:")
    for annotation in annotations:
        label = annotation['label']
        segment = annotation['segment']
        labels.append(label)
        segments.append(segment)
        print("标签:", label)
        print("区间:", segment)
        print("-" * 20)
    print("-" * 60)
    
    return fps, labels, segments
        
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_default_image_loader():
    return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video  # 是一个list，每个元素是每帧图像被image_loader读取后的结果


def get_default_video_loader():
    return functools.partial(video_loader, image_loader=get_default_image_loader())


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, label_dict, downsample_rate, sample_duration, flag="train"):
    dataset = []  # 每个元素是一个clip，它由一个字典表示，该字典有三个键，分别是video_path，16个帧的索引frame_indices，类别label

    file = open(os.path.join(root_path, flag+".txt"), "r")
    video_folders = file.readlines()
    file.close()
    
    for video_folder in video_folders:
        video_folder = video_folder.strip()
        _, labels, segments = read_json(os.path.join(root_path, video_folder, "coarse_segment.json"))
        for i in range(len(labels)):
            label_id = label_dict[labels[i]]
            start_index = segments[i][0] * downsample_rate + 1
            end_index = segments[i][1] * downsample_rate + 1
            for index in range(start_index, (end_index - sample_duration + 1), sample_duration):
                sample_i = {}
                sample_i['video_path'] = os.path.join(root_path, video_folder, "frames")
                sample_i['frame_indices'] = list(range(index, index + sample_duration))
                sample_i['label'] = label_id
                dataset.append(sample_i)
                
    return dataset


class Video(data.Dataset):
    def __init__(
        self,
        root_path,
        label_dict,
        flag='train', 
        spatial_transform=None, 
        temporal_transform=None, 
        get_loader=get_default_video_loader,
        downsample_rate=2,
        sample_duration=16,
        ):

        self.data = make_dataset(root_path, label_dict=label_dict, downsample_rate=downsample_rate, sample_duration=sample_duration, flag=flag)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        video_path = self.data[index]['video_path']
        frame_indices = self.data[index]['frame_indices']
        label = self.data[index]['label']

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(video_path, frame_indices)
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)  # (C,T,H,W)
        # print(clip.shape)

        return clip, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    spatial_transform = []

   
    spatial_transform.append(Resize(224))
    spatial_transform.append(CenterCrop(224))
    # normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
    #                                  opt.no_std_norm)

    # spatial_transform.append(RandomHorizontalFlip())
    # spatial_transform.append(ColorJitter())
    spatial_transform.append(ToTensor())

    # spatial_transform.append(ScaleValue(opt.value_scale))
    # spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    mydataset = Video(root_path='data/', spatial_transform=spatial_transform)
    print(len(mydataset))
    print(mydataset[10][0].shape)
    # train_loader = torch_geometric.loader.DataLoader(
    #         mydataset,
    #         batch_size=32,
    #         shuffle=True,
    #         num_workers=0,
    #         pin_memory=False,
    #         drop_last=True,
    #         # persistent_workers = True
    #     )

    # for i, (aug1, aug2, text1, mask1, text2, mask2) in enumerate(train_loader):
    #     print(aug1)
        # print(aug1.x.shape)
        # print(aug2)
        # print(aug2.x.dtype)
        # print(text1.shape)
        # print(mask1.shape)
        # print(text2.shape)
        # print(mask2.shape)
