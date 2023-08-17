from functools import partial
import torch
import torch.nn as nn


from pytorchvideo.models.resnet import create_bottleneck_block, create_resnet
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.x3d import (
    Swish,
    create_x3d,
    create_x3d_bottleneck_block,
)
from pytorchvideo.models.r2plus1d import (
    create_2plus1d_bottleneck_block,
    create_r2plus1d,
)
from pytorchvideo.models.vision_transformers import (
    create_multiscale_vision_transformers,
)
from pytorchvideo.models.slowfast import create_slowfast

from fvcore.common.config import CfgNode


from models.batchnorm_helper import get_norm



# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "slow_c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow_i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "slow_c2d": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "slow_i3d": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}


def get_head_act(act_func):
    """
    Return the actual head activation function given the activation fucntion name.

    Args:
        act_func (string): activation function to use. 'softmax': applies
        softmax on the output. 'sigmoid': applies sigmoid on the output.
    Returns:
        nn.Module: the activation layer.
    """
    if act_func == "softmax":
        return nn.Softmax(dim=1)
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError(
            "{} is not supported as a head activation "
            "function.".format(act_func)
        )
    

class PTVResNet(nn.Module):
    """
    ResNet models using PyTorchVideo model builder.
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(PTVResNet, self).__init__()

        assert cfg.MODEL.ARCH in [
            "c2d",
            "slow",
            "i3d",
        ], f"Unsupported MODEL.ARCH type {cfg.MODEL.ARCH} for PTVResNet"

        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # Params from configs.
        norm_module = get_norm(cfg)
        head_act = get_head_act(cfg.MODEL.HEAD_ACT)
        pool_size = _POOL1[cfg.MODEL.ARCH]
        num_groups = cfg.RESNET.NUM_GROUPS
        spatial_dilations = cfg.RESNET.SPATIAL_DILATIONS
        spatial_strides = cfg.RESNET.SPATIAL_STRIDES
        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        stage1_pool = pool_size[0][0] != 1 or len(set(pool_size[0])) > 1
        stage_spatial_stride = (
            spatial_strides[0][0],
            spatial_strides[1][0],
            spatial_strides[2][0],
            spatial_strides[3][0],
        )
        if cfg.MODEL.ARCH == "i3d":
            stage_conv_a_kernel_size = (
                (3, 1, 1),
                [(3, 1, 1), (1, 1, 1)],
                [(3, 1, 1), (1, 1, 1)],
                [(1, 1, 1), (3, 1, 1)],
            )
        else:
            stage_conv_a_kernel_size = (
                (temp_kernel[1][0][0], 1, 1),
                (temp_kernel[2][0][0], 1, 1),
                (temp_kernel[3][0][0], 1, 1),
                (temp_kernel[4][0][0], 1, 1),
            )

        self.model = create_resnet(
            # Input clip configs.
            input_channel=cfg.DATA.INPUT_CHANNEL_NUM[0],
            # Model configs.
            model_depth=cfg.RESNET.DEPTH,
            model_num_class=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            # Normalization configs.
            norm=norm_module,
            # Activation configs.
            activation=partial(nn.ReLU, inplace=cfg.RESNET.INPLACE_RELU),
            # Stem configs.
            stem_dim_out=cfg.RESNET.WIDTH_PER_GROUP,
            stem_conv_kernel_size=(temp_kernel[0][0][0], 7, 7),
            stem_conv_stride=(1, 2, 2),
            stem_pool=nn.MaxPool3d,
            stem_pool_kernel_size=(1, 3, 3),
            stem_pool_stride=(1, 2, 2),
            # Stage configs.
            stage1_pool=nn.MaxPool3d if stage1_pool else None,
            stage1_pool_kernel_size=pool_size[0],
            stage_conv_a_kernel_size=stage_conv_a_kernel_size,
            stage_conv_b_kernel_size=(
                (1, 3, 3),
                (1, 3, 3),
                (1, 3, 3),
                (1, 3, 3),
            ),
            stage_conv_b_num_groups=(
                num_groups,
                num_groups,
                num_groups,
                num_groups,
            ),
            stage_conv_b_dilation=(
                (1, spatial_dilations[0][0], spatial_dilations[0][0]),
                (1, spatial_dilations[1][0], spatial_dilations[1][0]),
                (1, spatial_dilations[2][0], spatial_dilations[2][0]),
                (1, spatial_dilations[3][0], spatial_dilations[3][0]),
            ),
            stage_spatial_h_stride=stage_spatial_stride,
            stage_spatial_w_stride=stage_spatial_stride,
            stage_temporal_stride=(1, 1, 1, 1),
            bottleneck=create_bottleneck_block,
            # Head configs.
            head=create_res_basic_head,
            head_pool=nn.AvgPool3d,
            head_pool_kernel_size=(
                cfg.DATA.NUM_FRAMES // pool_size[0][0],
                cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
            ),
            head_activation=None,
            head_output_with_global_average=False,
        )

        self.post_act = head_act

    def forward(self, x):
        x = self.model(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.post_act(x)
            x = x.mean([2, 3, 4])
            
        x = x.view(x.shape[0], -1)
        return x


class PTVX3D(nn.Module):
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(PTVX3D, self).__init__()

        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # Params from configs.
        norm_module = get_norm(cfg)
        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.model = create_x3d(
            # Input clip configs.
            input_channel=cfg.DATA.INPUT_CHANNEL_NUM[0],
            input_clip_length=cfg.DATA.NUM_FRAMES,
            input_crop_size=cfg.DATA.TRAIN_CROP_SIZE,
            # Model configs.
            model_num_class=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            width_factor=cfg.X3D.WIDTH_FACTOR,
            depth_factor=cfg.X3D.DEPTH_FACTOR,
            # Normalization configs.
            norm=norm_module,
            norm_eps=1e-5,
            norm_momentum=0.1,
            # Activation configs.
            activation=partial(nn.ReLU, inplace=cfg.RESNET.INPLACE_RELU),
            # Stem configs.
            stem_dim_in=cfg.X3D.DIM_C1,
            stem_conv_kernel_size=(temp_kernel[0][0][0], 3, 3),
            stem_conv_stride=(1, 2, 2),
            # Stage configs.
            stage_conv_kernel_size=(
                (temp_kernel[1][0][0], 3, 3),
                (temp_kernel[2][0][0], 3, 3),
                (temp_kernel[3][0][0], 3, 3),
                (temp_kernel[4][0][0], 3, 3),
            ),
            stage_spatial_stride=(2, 2, 2, 2),
            stage_temporal_stride=(1, 1, 1, 1),
            bottleneck=create_x3d_bottleneck_block,
            bottleneck_factor=cfg.X3D.BOTTLENECK_FACTOR,
            se_ratio=0.0625,
            inner_act=Swish,
            # Head configs.
            head_dim_out=cfg.X3D.DIM_C5,
            head_pool_act=partial(nn.ReLU, inplace=cfg.RESNET.INPLACE_RELU),
            head_bn_lin5_on=cfg.X3D.BN_LIN5,
            head_activation=None,
            head_output_with_global_average=False,
        )

        self.post_act = get_head_act(cfg.MODEL.HEAD_ACT)

    def forward(self, x):
        x = self.model(x)
        # Performs fully convlutional inference.
        if not self.training:
            x = self.post_act(x)
            x = x.mean([2, 3, 4])

        x = x.reshape(x.shape[0], -1)
        return x


class PTVR2plus1D(nn.Module):
    """
    R(2+1)D models using PyTorchVideo model builder.
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(PTVR2plus1D, self).__init__()

        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a single pathway R(2+1)D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        self.model = create_r2plus1d(
            # Input clip configs.
            input_channel=cfg.DATA.INPUT_CHANNEL_NUM[0],
            # Model configs.
            model_depth=cfg.RESNET.DEPTH,
            model_num_class=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            # Normalization configs.
            norm=get_norm(cfg),
            norm_eps=1e-5,
            norm_momentum=0.1,
            # Activation configs.
            activation=partial(nn.ReLU, inplace=cfg.RESNET.INPLACE_RELU),
            # Stem configs.
            stem_dim_out=cfg.RESNET.WIDTH_PER_GROUP,
            stem_conv_kernel_size=(1, 7, 7),
            stem_conv_stride=(1, 2, 2),
            # Stage configs.
            stage_conv_a_kernel_size=(
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
            ),
            stage_conv_b_kernel_size=(
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
            ),
            stage_conv_b_num_groups=(1, 1, 1, 1),
            stage_conv_b_dilation=(
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
            ),
            stage_spatial_stride=(2, 2, 2, 2),
            stage_temporal_stride=(1, 1, 2, 2),
            stage_bottleneck=(
                create_2plus1d_bottleneck_block,
                create_2plus1d_bottleneck_block,
                create_2plus1d_bottleneck_block,
                create_2plus1d_bottleneck_block,
            ),
            # Head configs.
            head_pool=nn.AvgPool3d,
            head_pool_kernel_size=(
                cfg.DATA.NUM_FRAMES // 4,
                cfg.DATA.TRAIN_CROP_SIZE // 32,
                cfg.DATA.TRAIN_CROP_SIZE // 32,
            ),
            head_activation=None,
            head_output_with_global_average=False,
        )

        self.post_act = get_head_act(cfg.MODEL.HEAD_ACT)

    def forward(self, x):
        x = self.model(x)
        # Performs fully convlutional inference.
        if not self.training:
            x = self.post_act(x)
            x = x.mean([2, 3, 4])

        x = x.view(x.shape[0], -1)
        return x


class PTVSlowFast(nn.Module):
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(PTVSlowFast, self).__init__()

        self.ALPHA = cfg.SLOWFAST.ALPHA
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        _MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

        # Params from configs.
        norm_module = get_norm(cfg)
        pool_size = _POOL1[cfg.MODEL.ARCH]
        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        spatial_dilations = cfg.RESNET.SPATIAL_DILATIONS
        spatial_strides = cfg.RESNET.SPATIAL_STRIDES
        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        num_block_temp_kernel = cfg.RESNET.NUM_BLOCK_TEMP_KERNEL
        stage_depth = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        stage_conv_a_kernel_sizes = [[], []]
        for pathway in range(2):
            for stage in range(4):
                stage_conv_a_kernel_sizes[pathway].append(
                    ((temp_kernel[stage + 1][pathway][0], 1, 1),)
                    * num_block_temp_kernel[stage][pathway]
                    + ((1, 1, 1),)
                    * (
                        stage_depth[stage]
                        - num_block_temp_kernel[stage][pathway]
                    )
                )

        # Head from config
        # Number of stages = 4
        stage_dim_in = cfg.RESNET.WIDTH_PER_GROUP * 2 ** (4 + 1)
        head_in_features = stage_dim_in + stage_dim_in // cfg.SLOWFAST.BETA_INV


        head_pool_kernel_sizes = (
            (
                cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA // pool_size[0][0],
                cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
            ),
            (
                cfg.DATA.NUM_FRAMES // pool_size[1][0],
                cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][1],
                cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][2],
            ),
        )

        self.model = create_slowfast(
            # SlowFast configs.
            slowfast_channel_reduction_ratio=cfg.SLOWFAST.BETA_INV,
            slowfast_conv_channel_fusion_ratio=cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            slowfast_fusion_conv_kernel_size=(
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                1,
                1,
            ),
            slowfast_fusion_conv_stride=(cfg.SLOWFAST.ALPHA, 1, 1),
            # Input clip configs.
            input_channels=cfg.DATA.INPUT_CHANNEL_NUM,
            # Model configs.
            model_depth=cfg.RESNET.DEPTH,
            model_num_class=cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            # Normalization configs.
            norm=norm_module,
            # Activation configs.
            activation=partial(nn.ReLU, inplace=cfg.RESNET.INPLACE_RELU),
            # Stem configs.
            stem_dim_outs=(
                width_per_group,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ),
            stem_conv_kernel_sizes=(
                (temp_kernel[0][0][0], 7, 7),
                (temp_kernel[0][1][0], 7, 7),
            ),
            stem_conv_strides=((1, 2, 2), (1, 2, 2)),
            stem_pool=nn.MaxPool3d,
            stem_pool_kernel_sizes=((1, 3, 3), (1, 3, 3)),
            stem_pool_strides=((1, 2, 2), (1, 2, 2)),
            # Stage configs.
            stage_conv_a_kernel_sizes=stage_conv_a_kernel_sizes,
            stage_conv_b_kernel_sizes=(
                ((1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)),
                ((1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)),
            ),
            stage_conv_b_num_groups=(
                (num_groups, num_groups, num_groups, num_groups),
                (num_groups, num_groups, num_groups, num_groups),
            ),
            stage_conv_b_dilations=(
                (
                    (1, spatial_dilations[0][0], spatial_dilations[0][0]),
                    (1, spatial_dilations[1][0], spatial_dilations[1][0]),
                    (1, spatial_dilations[2][0], spatial_dilations[2][0]),
                    (1, spatial_dilations[3][0], spatial_dilations[3][0]),
                ),
                (
                    (1, spatial_dilations[0][1], spatial_dilations[0][1]),
                    (1, spatial_dilations[1][1], spatial_dilations[1][1]),
                    (1, spatial_dilations[1][1], spatial_dilations[1][1]),
                    (1, spatial_dilations[1][1], spatial_dilations[1][1]),
                ),
            ),
            stage_spatial_strides=(
                (
                    spatial_strides[0][0],
                    spatial_strides[1][0],
                    spatial_strides[2][0],
                    spatial_strides[3][0],
                ),
                (
                    spatial_strides[0][1],
                    spatial_strides[1][1],
                    spatial_strides[2][1],
                    spatial_strides[3][1],
                ),
            ),
            stage_temporal_strides=((1, 1, 1, 1), (1, 1, 1, 1)),
            bottleneck=create_bottleneck_block,
            # Head configs.
            head=create_res_basic_head,
            head_pool=nn.AvgPool3d,
            head_pool_kernel_sizes=head_pool_kernel_sizes,
            head_activation=None,
            head_output_with_global_average=False,
        )

        self.post_act = get_head_act(cfg.MODEL.HEAD_ACT)

    def forward(self, x):
        x_slow = x[:, :, ::self.ALPHA, :, :]
        x = self.model([x_slow, x])

        # Performs fully convlutional inference.
        if not self.training:
            x = self.post_act(x)
            x = x.mean([2, 3, 4])

        x = x.view(x.shape[0], -1)
        return x



