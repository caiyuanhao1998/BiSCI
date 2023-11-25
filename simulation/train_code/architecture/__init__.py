import torch
from .BiSRNet import BiSRNet
from .BBCU import BBCU
from .BiConnect import BiConnect
from .BirealNet import BirealNet
from .BTM import BTM
from .ReActNet import ReActNet
from .IRNet import IRNet
from .BNN import BNN


def model_generator(method, pretrained_model_path=None):
    if method == 'bisrnet':
        model = BiSRNet(in_channels=28, out_channels=28, n_feat=28, stage=1, num_blocks=[1,1,1]).cuda()
    elif method == 'bbcu':
        model = BBCU(in_channels=28, out_channels=28, n_feat=28, stage=1, num_blocks=[1,1,1]).cuda()
    elif method == 'biconnect':
        model = BiConnect(in_channels=28, out_channels=28, n_feat=28, stage=1, num_blocks=[1,1,1]).cuda()
    elif method == 'bireal':
        model = BirealNet(in_channels=28, out_channels=28, n_feat=28, stage=1, num_blocks=[1,1,1]).cuda()
    elif method == 'btm':
        model = BTM(in_channels=28, out_channels=28, n_feat=28, stage=1, num_blocks=[1,1,1]).cuda()
    elif method == 'reactnet':
        model = ReActNet(in_channels=28, out_channels=28, n_feat=28, stage=1, num_blocks=[1,1,1]).cuda()
    elif method == 'irnet':
        model = IRNet(in_channels=28, out_channels=28, n_feat=28, stage=1, num_blocks=[1,1,1]).cuda()
    elif method == 'bnn':
        model = BNN(in_channels=28, out_channels=28, n_feat=28, stage=1, num_blocks=[1,1,1]).cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=True)
    return model