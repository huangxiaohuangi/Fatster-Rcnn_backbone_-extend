import os
import torch
import yaml
from torchvision.ops.misc import FrozenBatchNorm2d

from od.models.backbone.feature_pyramid_network import LastLevelMaxPool, BackboneWithFPN
from od.models.backbone.resnet import resnet50
from od.models.backbone.resnet_se import se_resnet50, at_se_resnet50


def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def resnet50_fpn_backbone(pretrain_path="",
                          norm_layer=FrozenBatchNorm2d,  # FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新
                          trainable_layers=3,
                          returned_layers=None,
                          extra_blocks=None, model_config=None):
    """
    搭建resnet50_fpn——backbone
    Args:
        pretrain_path: resnet50的预训练权重，如果不使用就默认为空
        norm_layer: 官方默认的是FrozenBatchNorm2d，即不会更新参数的bn层(因为如果batch_size设置的很小会导致效果更差，还不如不用bn层)
                    如果自己的GPU显存很大可以设置很大的batch_size，那么自己可以传入正常的BatchNorm2d层
                    (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        trainable_layers: 指定训练哪些层结构
        returned_layers: 指定哪些层的输出需要返回
        extra_blocks: 在输出的特征层基础上额外添加的层结构

    Returns:

    """


    if type(model_config) is str:
        model_config = yaml.load(open(model_config, 'r'))
    backbone_type = model_config['backbone']
    print(model_config)

    CoordAttention_flag = backbone_type['cor_att']
    if backbone_type['se']:
        resnet_backbone = se_resnet50(pretrained='se_resnet50')
        pretrain_path = './pre_model/se_resnet50.pth'
    elif backbone_type['se_att']:
        resnet_backbone = at_se_resnet50(pretrained='se_resnet50')
        pretrain_path = './pre_model/se_resnet50.pth'
    else:
        resnet_backbone = resnet50(backbone_type=backbone_type)
        pretrain_path = "./pre_model/resnet50.pth"
    print(resnet_backbone)
    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)

    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "{} is not exist.".format(pretrain_path)
        # 载入预训练权重
        # print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))
        model_dict = resnet_backbone.state_dict()  # 网络层的参数
        # 需要加载的预训练参数
        pretrained_dict = torch.load(pretrain_path)  # torch.load得到是字典，我们需要的是state_dict下的参数
        pretrained_dict = {k.replace('module.', ''): v for k, v in
                           pretrained_dict.items()}
        # 因为pretrained_dict得到module.conv1.weight，但是自己建的model无module，只是conv1.weight，所以改写下。
        # 删除pretrained_dict.items()中model所没有的东西
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预训练模型中，自己建的model有的参数
        model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
        resnet_backbone.load_state_dict(model_dict)  # model加载dict中的数据，更新网络的初始值

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # 如果要训练所有层结构的话，不要忘了conv1后还有一个bn1
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    # freeze layers
    # all train
    # for name, parameter in resnet_backbone.named_parameters():
    #     # 只训练不在layers_to_train列表中的层结构
    #     if all([not name.startswith(layer) for layer in layers_to_train]):
    #         parameter.requires_grad_(False)
    #     else:
    #         print(name)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]

    # 返回的特征层个数肯定大于0小于5
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    if CoordAttention_flag == False:
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        # return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    else:
        return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
        return_layers["coordAtt"] = '3'
        del return_layers["layer4"]

    # in_channel 为layer4的输出特征矩阵channel = 2048
    in_channels_stage2 = resnet_backbone.inplanes // 8  # 256
    # 记录resnet50提供给fpn的每个特征层channel
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # 通过fpn后得到的每个特征层的channel
    out_channels = 256
    # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    # in_channels_list = [256, 512, 1024, 2048]
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
