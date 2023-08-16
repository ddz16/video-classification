import cv2
import os
import glob
import argparse
import functools
from PIL import Image
import numpy as np
import json
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian

from model import generate_model
# from train import train, val, test
# from dataset import Video
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop, ToTensor)

import torch

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def video_loader(video_dir_path, frame_indices):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(pil_loader(image_path))
        else:
            return video

    return video  # 是一个list，每个元素是每帧图像


def load_video(video_path):
    input_path_extension = video_path.split('.')[-1]
    if input_path_extension in ['mp4']:
        return [video_path]
    elif input_path_extension == "txt":
        with open(video_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(os.path.join(video_path, "*.mp4"))


def save_img(input_video_path, output_path, timeF=5):
    """
    为视频抽帧, 读取视频并将视频转换为图片
    input_video_path: | a folder contains several videos (.mp4) |
                      | a video's path (.mp4) | 
                      | a path of a text (.txt) containing all the paths of videos |
    output_path: 抽帧的保存路径
    timeF: 抽帧频率, 多少帧抽一次
    """
    # input_video_path = r'E:/VideoDir/'
    videos = load_video(input_video_path)
    for video_name in videos:
        vc = cv2.VideoCapture(video_name)  # 读入视频文件

        out_dir = os.path.join(output_path, os.path.basename(video_name)[:-4])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        c = 1
        if vc.isOpened():  # 判断是否正常打开
            rval, frame = vc.read()
        else:
            rval = False
        while rval:  # 循环读取视频帧
            rval, frame = vc.read()
            if (c % timeF == 0):  # 每隔timeF帧进行存储操作
                if frame is not None:
                    cv2.imwrite(os.path.join(out_dir,'{:05d}.jpg'.format(c//timeF)), frame)
            c = c + 1
            cv2.waitKey(1)
        vc.release()
    
    return out_dir


def crf_process(probs, num_classes):

    d = dcrf.DenseCRF(probs.shape[1], num_classes)
    u = unary_from_softmax(probs, scale=None, clip=1e-7)  # sm: (C, L), 
    d.setUnaryEnergy(u)

    feats = create_pairwise_gaussian(sdims=(1.0,), shape=probs.shape[1:])
    d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.FULL_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # 进行10次推理
    Q = d.inference(10)

    # 找出每个帧最可能的类
    MAP = np.argmax(Q, axis=0)

    # print(probs)
    # print(np.argmax(probs, axis=0))
    # print(MAP)
    return MAP




def get_video_info(video_path):
    # 遍历指定路径下的所有文件
    for file in os.listdir(video_path):
        # 判断文件是否为视频文件
        if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # 获取视频文件的完整路径
            video_file = os.path.join(video_path, file)
            
            # 使用OpenCV读取视频文件
            cap = cv2.VideoCapture(video_file)
            
            # 获取视频的帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 获取视频的总帧数
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            # 计算视频的总秒数
            duration = total_frames / fps
            
            break
    
    return file, fps, duration


def find_change_indices(result_list):
    # 找到一个list中变化时刻的索引，比如['a','a','b','b','a','a']中，变化索引为1和3
    change_indices = []
    prev_value = None
    for i, value in enumerate(result_list):
        if value != prev_value:
            change_indices.append(i)
        prev_value = value
    return change_indices


def list2annotation(result_list):
    change_indices = find_change_indices(result_list)
    
    annotation = []
    cur_time = 0
    for change_index in change_indices:
        cur_dic = {"label": result_list[change_index], "segment": [cur_time, change_index//2]}
        annotation.append(cur_dic)
        cur_time = change_index//2
    
    last_dic = {"label": result_list[-1], "segment": [cur_time, len(result_list)//2]}
    annotation.append(last_dic)
    
    return annotation


def results2json(video_name, fps, duration, result_list, path):
    annotation = list2annotation(result_list)
    data = {video_name : {"fps": fps, "duration": duration, "annotation": annotation}}
    
    with open(os.path.join(path, "coarse_segment.json"), "w") as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/', type=str, help='dataset file path')
    parser.add_argument('--sample_size', default=256, type=int, help='the size of input frame')
    parser.add_argument('--sample_duration', default=16, type=int, help='the length of clip')
    parser.add_argument('--n_classes', default=7, type=int, help='the number of classes')
    parser.add_argument('--model', default='/home/detection/ddz/video-classification/checkpoints/save_100.pth', type=str, help='checkpoint model file')
    parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--model_name', default='resnet', type=str, help='Currently only support resnet')
    parser.add_argument('--model_depth', default=34, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')

    opt = parser.parse_args()

    num_classes = 7
    label_dic = {0: 'dive', 1: 'walk', 2: 'observe', 3: 'work', 4: 'ascend', 5: 'off', 6: 'other'}
    
    video_path = './data/xxxx/'
    video_name, fps, duration = get_video_info(video_path)
    print('视频名称:', video_name)
    print('帧率:', fps)
    print('总秒数:', duration)

    # 抽帧后的帧存放路径
    frames_path = video_path + "frames/"  
    all_frames = os.listdir(frames_path)
    print('total frame num: {}'.format(len(all_frames)))
    
    # 数据增强
    spatial_transform = Compose([
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    temporal_transform = None

    # 模型
    device = torch.device("cuda:0")
    model = generate_model(opt).to(device)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    model.load_state_dict(model_data['state_dict'])
    model.eval()

    probs_list = []
    pre_label_list = []
    post_label_list = []
    with torch.no_grad():
        # 从头到尾，每16个帧为一个clip，遍历进行分类
        clip_index = 1
        for begin_index in range(1, len(all_frames)-opt.sample_duration+2, opt.sample_duration):  # 每个clip的开始帧的索引     
            frame_indices = list(range(begin_index, begin_index+opt.sample_duration))
            if temporal_transform is not None:
                frame_indices = temporal_transform(frame_indices)
            clip = video_loader(frames_path, frame_indices)
            if spatial_transform is not None:
                clip = [spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)  # (C,T,H,W)
            clip = clip.to(device)
            output = model(clip.unsqueeze(0)).squeeze(0)
            score = torch.softmax(output, dim=0)
            conf, cls = torch.max(score, dim=0)
            print("clip: {}, cls: {}, action: {}, conf: {:.2f}".format(clip_index, cls.item(), label_dic[cls.item()], conf.item()))
            prob = score.cpu().numpy()
            probs_list.append(prob)
            clip_index += 1
            # if clip_index > 10:
            #     break

    probs_all = np.stack(probs_list, axis=1)
    pre_result = np.argmax(probs_all, axis=0)
    
    for each in pre_result:
        pre_label_list.extend([label_dic[each] for _ in range(opt.sample_duration)])

    post_result = crf_process(probs_all, num_classes)
    for each in post_result:
        post_label_list.extend([label_dic[each] for _ in range(opt.sample_duration)])
        
    results2json(video_name, fps, duration, post_label_list, video_path)
