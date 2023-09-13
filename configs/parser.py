import argparse
import sys
from fvcore.common.config import CfgNode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        dest="cfg_files",
        help="Path to the config files",
        default="configs/I3D.yaml",
    )
    parser.add_argument(
        "--opts",
        help="",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args, config_path=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    _C = CfgNode()

    # -----------------------------------------------------------------------------
    # Data options
    # -----------------------------------------------------------------------------
    _C.DATA = CfgNode()
    _C.DATA.DATA_PATH = 'data/'   # dataset file path
    _C.DATA.NUM_FRAMES = 16       # the length of clip
    _C.DATA.SAMPLING_RATE = 2     # frames/second, i.e., each second has two frames
    _C.DATA.INPUT_CHANNEL_NUM = [3]
    _C.DATA.TRAIN_CROP_SIZE = 256
    _C.DATA.TEST_CROP_SIZE = 256
    _C.DATA.MEAN = [0.485, 0.456, 0.406]
    _C.DATA.STD = [0.229, 0.224, 0.225]

    # -----------------------------------------------------------------------------
    # Model options
    # -----------------------------------------------------------------------------
    _C.MODEL = CfgNode()
    _C.MODEL.PRETRAIN = False
    _C.MODEL.PRETRAIN_FILE = ""
    _C.MODEL.ARCH = "i3d"
    _C.MODEL.MODEL_NAME = "I3D"
    _C.MODEL.NUM_CLASSES = 7
    _C.MODEL.SINGLE_PATHWAY_ARCH = [           # model architectures that has one single pathway
        "2d",
        "c2d",
        "i3d",
        "slow",
        "x3d",
        "mvit",
        "maskmvit",
    ]
    _C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"] # model architectures that has multiple pathways
    _C.MODEL.DROPOUT_RATE = 0.5                # dropout rate before final projection in the backbone
    _C.MODEL.HEAD_ACT = "softmax"              # activation layer for the output head

    # ---------------------------------------------------------------------------- #
    # Batch norm options
    # ---------------------------------------------------------------------------- #
    _C.BN = CfgNode()
    _C.BN.USE_PRECISE_STATS = False  # precise BN stats
    _C.BN.NUM_BATCHES_PRECISE = 200  # number of samples use to compute precise bn
    _C.BN.WEIGHT_DECAY = 0.0         # weight decay value that applies on BN
    _C.BN.NORM_TYPE = "batchnorm"    # norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
    _C.BN.NUM_SPLITS = 1             # Parameter for SubBatchNorm
    _C.BN.NUM_SYNC_DEVICES = 1       # Parameter for NaiveSyncBatchNorm
    _C.BN.GLOBAL_SYNC = False        # Parameter for NaiveSyncBatchNorm

    # -----------------------------------------------------------------------------
    # ResNet options
    # -----------------------------------------------------------------------------
    _C.RESNET = CfgNode()
    _C.RESNET.DEPTH = 50  # Number of weight layers.
    _C.RESNET.NUM_GROUPS = 1  # Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
    _C.RESNET.WIDTH_PER_GROUP = 64  # Width of each group (64 -> ResNet; 4 -> ResNeXt).
    _C.RESNET.INPLACE_RELU = True  # Apply relu in a inplace manner.
    _C.RESNET.ZERO_INIT_FINAL_BN = False  #  If true, initialize the gamma of the final BN of each block to zero.
    _C.RESNET.ZERO_INIT_FINAL_CONV = False  #  If true, initialize the final conv layer of each block to zero.
    _C.RESNET.STRIDE_1X1 = False  # Apply stride to 1x1 conv.
    _C.RESNET.TRANS_FUNC = "bottleneck_transform"  # Transformation function.
    _C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]
    _C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]  # Size of stride on different res stages.
    _C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]  # Size of dilation on different res stages.

    # ---------------------------------------------------------------------------- #
    # X3D  options
    # See https://arxiv.org/abs/2004.04730 for details about X3D Networks.
    # ---------------------------------------------------------------------------- #
    _C.X3D = CfgNode()
    _C.X3D.WIDTH_FACTOR = 1.0  # Width expansion factor.
    _C.X3D.DEPTH_FACTOR = 1.0  # Depth expansion factor.
    _C.X3D.BOTTLENECK_FACTOR = 1.0  # Bottleneck expansion factor for the 3x3x3 conv.
    _C.X3D.DIM_C5 = 2048  # Dimensions of the last linear layer before classificaiton.
    _C.X3D.DIM_C1 = 12   # Dimensions of the first 3x3 conv layer.
    _C.X3D.SCALE_RES2 = False  # Whether to scale the width of Res2, default is false.
    _C.X3D.BN_LIN5 = False  # Whether to use a BatchNorm (BN) layer before the classifier, default is false.
    _C.X3D.CHANNELWISE_3x3x3 = True  # Whether to use channelwise (=depthwise) convolution in the center (3x3x3) convolution operation of the residual blocks.

    # -----------------------------------------------------------------------------
    # SlowFast options
    # -----------------------------------------------------------------------------
    _C.SLOWFAST = CfgNode()
    _C.SLOWFAST.ALPHA = 4     # Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and Fast pathways.
    _C.SLOWFAST.BETA_INV = 8  # Corresponds to the inverse of the channel reduction ratio, $\beta$ between the Slow and Fast pathways.
    _C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2  # Ratio of channel dimensions between the Slow and Fast pathways.
    _C.SLOWFAST.FUSION_KERNEL_SZ = 5  # Kernel dimension used for fusing information from Fast pathway to Slow pathway.

    # ---------------------------------------------------------------------------- #
    # Training options.
    # ---------------------------------------------------------------------------- #
    _C.TRAIN = CfgNode()
    _C.TRAIN.ENABLE = True
    _C.TRAIN.BATCH_SIZE = 64
    _C.TRAIN.NUM_WORKERS = 8
    _C.TRAIN.CHECKPOINT_PATH = "checkpoints/"

    # ---------------------------------------------------------------------------- #
    # Optimizer options
    # ---------------------------------------------------------------------------- #
    _C.SOLVER = CfgNode()
    _C.SOLVER.OPTIMIZING_METHOD = "adam"  # Optimization method.
    _C.SOLVER.BASE_LR = 0.1  # Base learning rate.
    _C.SOLVER.COSINE_END_LR = 0.0  # Final learning rates for 'cosine' policy.
    _C.SOLVER.GAMMA = 0.1  # Exponential decay factor.
    _C.SOLVER.STEP_SIZE = 1  # Step size for 'exp' and 'cos' policies (in epochs).
    _C.SOLVER.STEPS = []  # Steps for 'steps_' policies (in epochs).
    _C.SOLVER.LRS = []  # Learning rates for 'steps_' policies.
    _C.SOLVER.EPOCHS = 300
    _C.SOLVER.WARMUP = False
    _C.SOLVER.WARMUP_EPOCHS = 0  # Gradually warm up the SOLVER.BASE_LR over this number of epochs.b
    _C.SOLVER.SCHEDULE = "cosine"
    _C.SOLVER.SCHEDULE_STEPS = [10, 20, 30]
    _C.SOLVER.SCHEDULE_GAMMA = 0.1
    _C.SOLVER.MOMENTUM = 0.9  # Momentum.
    _C.SOLVER.DAMPENING = 0.0  # Momentum dampening.
    _C.SOLVER.NESTEROV = True  # Nesterov momentum.
    _C.SOLVER.WEIGHT_DECAY = 1e-4  # L2 regularization.
    _C.SOLVER.WARMUP_FACTOR = 0.1  # Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
    _C.SOLVER.WARMUP_START_LR = 0.01  # The start learning rate of the warm up.
    _C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False  # Base learning rate is linearly scaled with NUM_SHARDS.
    _C.SOLVER.COSINE_AFTER_WARMUP = False  # If True, start from the peak cosine learning rate after warm up.
    _C.SOLVER.CLIP_GRAD_L2NORM = None  # Clip gradient at this norm before optimizer update
    _C.SOLVER.BETAS = (0.9, 0.999)  # Adam's beta

    _C.FEATURES = CfgNode()
    _C.FEATURES.CHECKPOINT = "checkpoints/i3d/save_xx.pth"
    _C.FEATURES.EXTRACT = False
    _C.FEATURES.FRAMES_PATH = "data/xxxxx/frames/"
    _C.FEATURES.STRIDE = 1

    _C.OUTPUT = 'output.json'
    _C.TENSORBOARD = False

    # Load config from cfg.
    if config_path is not None:
        _C.merge_from_file(config_path)

    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        _C.merge_from_list(args.opts)

    return _C