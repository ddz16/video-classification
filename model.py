import torch
from torch import nn

from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet
from models.ptv_model_builder import PTVResNet, PTVR2plus1D, PTVX3D, PTVSlowFast

def generate_model(opt):
    """
        mode: ['score', 'feature']
        model_name: ['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet']
        model_depth: [10, 18, 34, 50, 101, 152, 200]
        n_classes
        resnet_shortcut: [A, B]
        sample_size: 输入clip的帧的大小
        sample_duration: 输入clip的长度
        wide_resnet_k: wideresnet model才有的
        resnext_cardinality: resnext model才有的
    """
    assert opt.mode in ['score', 'feature']
    if opt.mode == 'score':
        last_fc = True
    elif opt.mode == 'feature':
        last_fc = False

    assert opt.model_name in ['resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet']

    if opt.model_name == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 18:
            model = resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 34:
            model = resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 50:
            model = resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                    sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                    last_fc=last_fc)
        elif opt.model_depth == 101:
            model = resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.model_depth == 152:
            model = resnet.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.model_depth == 200:
            model = resnet.resnet200(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
    elif opt.model_name == 'wideresnet':
        assert opt.model_depth in [50]

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, k=opt.wide_resnet_k,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
    elif opt.model_name == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        if opt.model_depth == 50:
            model = resnext.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                     sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                     last_fc=last_fc)
        elif opt.model_depth == 101:
            model = resnext.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                      last_fc=last_fc)
        elif opt.model_depth == 152:
            model = resnext.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut, cardinality=opt.resnext_cardinality,
                                      sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                      last_fc=last_fc)
    elif opt.model_name == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                            last_fc=last_fc)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                            last_fc=last_fc)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                            sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                            last_fc=last_fc)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                             last_fc=last_fc)
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                             last_fc=last_fc)
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut,
                                             sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                             last_fc=last_fc)
    elif opt.model_name == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        if opt.model_depth == 121:
            model = densenet.densenet121(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.model_depth == 169:
            model = densenet.densenet169(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.model_depth == 201:
            model = densenet.densenet201(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)
        elif opt.model_depth == 264:
            model = densenet.densenet264(num_classes=opt.n_classes,
                                         sample_size=opt.sample_size, sample_duration=opt.sample_duration,
                                         last_fc=last_fc)

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)  # 默认用全部的GPU

    return model


def generate_model_PTV(cfg):
    
    model_dict = {
        'I3D': PTVResNet,
        'R2plus1D': PTVR2plus1D,
        'X3D': PTVX3D,
        'SlowFast': PTVSlowFast,
    }

    model = model_dict[cfg.MODEL.MODEL_NAME](cfg)

    if cfg.MODEL.PRETRAIN:
        ckpt = torch.load(cfg.MODEL.PRETRAIN_FILE)["model_state"]
        pretrained_dict = {"model."+k: v for k, v in ckpt.items() if "proj" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
        print("missing_keys:", missing_keys)
        print("unexpected_keys:", unexpected_keys)

    return model