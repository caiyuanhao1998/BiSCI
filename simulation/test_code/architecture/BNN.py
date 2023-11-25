import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import os
from pdb import set_trace as stx
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ['cuda_visible_device']='2'
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# --------------------------------------------- Binarized Basic Units -----------------------------------------------------------------
def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
    
class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out= out +self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
    
def Binaryconv3x3(in_planes, out_planes, stride=1,groups=1,bias=False):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1,groups=groups, bias=bias)

# def binaryconv3x3(in_planes, out_planes, stride=1,groups=1,bias=False):
#     """3x3 convolution with padding"""
#     return HardBinaryConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,groups=groups,bias=bias)

def Binaryconv1x1(in_planes, out_planes, stride=1,groups=1,bias=False):
    """1x1 convolution"""
    return BinarizeConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,groups=groups,bias=bias)


class FF_BICONV3(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1,bias=False):
        super(FF_BICONV3, self).__init__()

        self.conv1 = Binaryconv3x3(inplanes, planes, stride=stride,groups=groups,bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        # self.conv2 = Binaryconv3x3(planes, planes,stride=stride,groups=groups,bias=bias)
        #self.tanh2 = nn.Hardtanh(inplace=True)
        #self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        # out = self.conv2(out)

        out = out + residual

        #out = self.bn2(out)
        #out = self.tanh2(out)

        return out

           
class BinaryConv2d_Fusion_Decrease(nn.Module):
    '''
    空间尺寸不变且通道数减半 - Upsample 去掉上采样
    input: b,c,h,w
    output: b,c/2,h,w
    '''
    def __init__(self, inplanes, planes, stride=1,groups=1,bias=False):
        super(BinaryConv2d_Fusion_Decrease, self).__init__()

        self.conv1 = Binaryconv1x1(inplanes, planes, stride=stride,groups=groups,bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        # self.conv2 = Binaryconv1x1(planes, planes,stride=stride,groups=groups,bias=bias)
        #self.tanh2 = nn.Hardtanh(inplace=True)
        #self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h,w
        '''

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        # out = self.conv2(out)

        #out = self.bn2(out)
        #out = self.tanh2(out)

        return out
    
class BinaryConv2d_Fusion_Increase(nn.Module):
    '''
    空间尺寸不变且通道数翻倍 - Downsample 去掉下采样
    input: b,c,h,w
    output: b,2c,h,w
    '''
    def __init__(self, inplanes, planes, stride=1, groups=1,bias=False):
        super(BinaryConv2d_Fusion_Increase, self).__init__()

        self.conv1 = Binaryconv1x1(inplanes, planes, stride=stride,groups=groups,bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        # self.conv2 = Binaryconv1x1(planes, planes,stride=stride,groups=groups,bias=bias)
        #self.tanh2 = nn.Hardtanh(inplace=True)
        #self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h,w
        '''
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        # out = self.conv2(out)

        #out = self.bn2(out)
        #out = self.tanh2(out)

        return out
    

class BinaryConv2d_Down(nn.Module):
    '''
    降采样且通道数翻倍
    input: b,c,h,w
    output: b,c/2,2h,2w
    '''
    def __init__(self, inplanes, planes, stride=1,groups=1,bias=False):
        super(BinaryConv2d_Down, self).__init__()

        self.downsample = nn.Sequential(
                BinarizeConv2d(inplanes, planes,
                          kernel_size=1, stride=stride, groups=groups,bias=bias),
                nn.BatchNorm2d(planes),
            )
        
        self.conv1 = Binaryconv3x3(inplanes, planes, stride,groups=groups,bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        # self.conv2 = Binaryconv3x3(planes, planes,stride=1,groups=groups,bias=bias)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h/2,w/2
        '''

        residual = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        # out = self.conv2(out)

        residual = self.downsample(residual)

        out = out + residual

        out = self.bn2(out)
        out = self.tanh2(out)

        return out
    

class BinaryConv2d_Up(nn.Module):
    '''
    上采样且通道数减半
    input: b,c,h,w
    output: b,c/2,2h,2w
    '''
    def __init__(self, inplanes, planes, stride=1,  bias=False,groups=1):
        super(BinaryConv2d_Up, self).__init__()

        self.conv1 = Binaryconv3x3(inplanes, planes, stride=stride,groups=groups,bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        # self.conv2 = Binaryconv3x3(planes, planes,stride=stride,groups=groups,bias=bias)
        #self.tanh2 = nn.Hardtanh(inplace=True)
        #self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,c/2,2h,2w
        '''
        b,c,h,w = x.shape
        x = F.interpolate(x, scale_factor=2, mode='bilinear')

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        # out = self.conv2(out)

        #out = self.bn2(out)
        #out = self.tanh2(out)

        return out
    

# ---------------------------------------------------------- Binarized UNet------------------------------------------------------

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]



class FeedForward(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.net = nn.Sequential(
            BinaryConv2d_Fusion_Increase(dim, dim * mult, 1, bias=False),
            BinaryConv2d_Fusion_Increase(dim * mult, dim * mult * mult, 1, bias=False),
            GELU(),
            FF_BICONV3(dim * mult * mult, dim * mult * mult,1, groups=dim , bias=False),
            GELU(),
            BinaryConv2d_Fusion_Decrease(dim * mult * mult, dim * mult, 1, bias=False),
            BinaryConv2d_Fusion_Decrease(dim * mult, dim, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)



class BNN_block(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                PreNorm(dim, FeedForward(dim=dim))
            )

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for ff in self.blocks:
            # x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out



class BiUNet_BNN_body(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, stage=2, num_blocks=[2,4,4]):
        super(BiUNet_BNN_body, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)                           # 1-bit -> 32-bit
        # self.embedding = Binaryconv3x3(in_dim, self.dim, stride=1,bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                BNN_block(dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                BinaryConv2d_Down(dim_stage, dim_stage * 2, 2, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = BNN_block(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                BinaryConv2d_Up(dim_stage, dim_stage // 2, 1, bias=False),
                BinaryConv2d_Fusion_Decrease(dim_stage, dim_stage // 2, 1,bias=False),
                BNN_block(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        # self.mapping = Binaryconv3x3(in_dim, self.dim, stride=1, bias=False)                                  # 1-bit -> 32-bit
        
        #### activation function
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.BinAc = BinaryActivation()

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (BNN_block, FeaDownSample) in self.encoder_layers:
            # stx()
            fea = BNN_block(fea)
            fea_encoder.append(fea)
            # stx()
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, BNN_block) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            # stx()
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = BNN_block(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out



class BNN(nn.Module):
    '''
    Only 3 layers are 32-bit conv
    '''
    def __init__(self, in_channels=28, out_channels=28, n_feat=28, stage=3, num_blocks=[1,1,1]):
        super(BNN, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)       # 1-bit -> 32-bit
        modules_body = [BiUNet_BNN_body(dim=n_feat, stage=2, num_blocks=num_blocks) for _ in range(stage)]
        self.fution = nn.Conv2d(56, 28, 1, padding=0, bias=True)                                            # 1-bit -> 32-bit
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)     # 1-bit -> 32-bit
        init_model(self)

    def initial_x(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,256]
        :return: z: [b,28,256,256]
        """
        x = self.fution(torch.cat([y, Phi], dim=1))
        return x

    def forward(self, y, Phi=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        if Phi==None:
            Phi = torch.rand((1,28,256,256)).cuda()
            # Phi = torch.rand((1,28,256,256)).to(device)
        x = self.initial_x(y, Phi)
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)
        h = self.body(x)
        h = self.conv_out(h)
        h += x
        return h[:, :, :h_inp, :w_inp]


# if __name__ == "__main__":
#     from fvcore.nn import FlopCountAnalysis
#     inputs = torch.rand(1, 28, 256, 256).cuda()
#     model = BiUNet_BNN_5L(stage=1,num_blocks=[1,1,1]).cuda()
#     # inputs = torch.rand(1, 28, 256, 256).to(device)
#     # model = BiUNet(stage=1,num_blocks=[1,1,1]).to(device)
#     flops = FlopCountAnalysis(model, inputs)
#     n_param = sum([p.nelement() for p in model.parameters()])
#     print(f'GMac:{flops.total()}')
#     print(f'Params:{n_param}')