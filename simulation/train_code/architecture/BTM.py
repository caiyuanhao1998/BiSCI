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

def get_weight(module):
    std, mean = torch.std_mean(module.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
    weight = (module.weight - mean) / (std + module.eps)
    return weight

# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
            bias=False, gamma=1.0, eps=1e-5, use_layernorm=False):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps ** 2 if use_layernorm else eps
        self.use_layernorm = use_layernorm  # experimental, slightly faster/less GPU memory to hijack LN kernel

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        return self.gain * weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)

# class HardBinaryScaledStdConv2d(nn.Conv2d):

#     def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, gamma=1.0, eps=1e-5, groups=1, bias=True):
#         super(HardBinaryScaledStdConv2d, self).__init__(in_chn,
#             out_chn,
#             kernel_size,stride=stride,
#             padding=padding,
#             groups=groups,
#             bias=bias)

#         # self.shape = (out_chn, in_chn, kernel_size, kernel_size)
#         # self.weight = nn.Parameter(torch.rand(self.shape) * 0.001, requires_grad=True)

#         self.gain = nn.Parameter(torch.ones(out_chn, 1, 1, 1))
#         self.scale = gamma * self.weight[0].numel() ** -0.5
#         self.eps = eps 

#     def get_weight(self):

#         std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
#         weight = self.scale * (self.weight - mean) / (std + self.eps)

#         scaling_factor = torch.mean(torch.mean(torch.mean(abs(weight),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
#         scaling_factor = scaling_factor.detach()
#         binary_weights_no_grad = scaling_factor * torch.sign(weight)
#         cliped_weights = torch.clamp(weight, -1.0, 1.0)
#         binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

#         return self.gain * binary_weights

#     def forward(self, x):

#         return F.conv2d(x, self.get_weight(), stride=self.stride, padding=self.padding,groups=self.groups)

class HardBinaryScaledStdConv2d(nn.Conv2d):

    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, gamma=1.0, eps=1e-5, groups=1, bias=True):
        super(HardBinaryScaledStdConv2d, self).__init__(in_chn,
            out_chn,
            kernel_size,stride=stride,
            padding=padding,
            groups=groups,
            bias=bias)

        self.shape = (out_chn, in_chn//groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(self.shape) * 0.001, requires_grad=True)

        self.gain = nn.Parameter(torch.ones(out_chn, 1, 1, 1))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps 

    def get_weight(self):

        std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        weight = self.scale * (self.weight - mean) / (std + self.eps)

        scaling_factor = torch.mean(torch.mean(torch.mean(abs(weight),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(weight)
        cliped_weights = torch.clamp(weight, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        return self.gain * binary_weights

    def forward(self, x):

        return F.conv2d(x, self.get_weight(), stride=self.stride, padding=self.padding,groups=self.groups)
    
class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        cliped_ac = torch.clamp(x, -1.0, 1.0)
        out = out_forward.detach() - cliped_ac.detach() + cliped_ac

        return out

# class HardBinaryConv(nn.Module):
#     def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
#         super(HardBinaryConv, self).__init__()
#         self.stride = stride
#         self.padding = padding
#         self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
#         self.shape = (out_chn, in_chn, kernel_size, kernel_size)
#         #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
#         self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

#     def forward(self, x):
#         real_weights = self.weight
#         scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
#         scaling_factor = scaling_factor.detach()
#         binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
#         cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
#         binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
#         y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

#         return y
# --------------------------------------------- Binarized Basic Units -----------------------------------------------------------------




def binaryconv3x3(in_planes, out_planes, stride=1,groups=1,bias=False):
    """3x3 convolution with padding"""
    return HardBinaryScaledStdConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,groups=groups,bias=bias)


def binaryconv1x1(in_planes, out_planes, stride=1,groups=1,bias=False):
    """1x1 convolution"""
    return HardBinaryScaledStdConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,groups=groups,bias=bias)


# class HardBinaryConv(nn.Module):
#     def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1,groups=1):
#         super(HardBinaryConv, self).__init__()
#         self.stride = stride
#         self.padding = padding
#         self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
#         self.shape = (out_chn, in_chn, kernel_size, kernel_size)
#         self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
#         self.groups=groups

#     def forward(self, x):
#         real_weights = self.weights.view(self.shape)
#         scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
#         #print(scaling_factor, flush=True)
#         scaling_factor = scaling_factor.detach()
#         binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
#         cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
#         binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
#         #print(binary_weights, flush=True)
#         y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, groups=self.groups)

#         return y
    
# class HardBinaryConv(nn.Conv2d):
#     def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1,groups=1, bias=True):
#         super(HardBinaryConv, self).__init__(in_chn,
#             out_chn,
#             kernel_size,stride=stride,
#             padding=padding,
#             groups=groups,
#             bias=bias)   
        # self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        # self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        # self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    # def forward(self, x):
    #     # stx()
    #     # real_weights = self.weights.view(self.shape)
    #     real_weights = self.weight
    #     scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
    #     #print(scaling_factor, flush=True)
    #     scaling_factor = scaling_factor.detach()
    #     binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
    #     cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
    #     binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
    #     #print(binary_weights, flush=True)
    #     y = F.conv2d(x, binary_weights, self.bias,stride=self.stride, padding=self.padding, groups=self.groups)

    #     return y

class FF_BICONV3(nn.Module):
    def __init__(self, inplanes, planes, alpha, beta1, stride=1, groups=1,bias=False):
        super(FF_BICONV3, self).__init__()

        self.alpha = alpha
        self.beta1 = beta1

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3= binaryconv3x3(inplanes, inplanes, stride=stride,groups=groups,bias=bias)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.binary_activation = BinaryActivation()

    def forward(self, x):
        x = x*self.beta1 

        out1 = self.move11(x)

        out1 = self.binary_activation(out1)
        # stx()
        out1 = self.binary_3x3(out1)

        out1 = x + out1*self.alpha

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        return out1
           
class BinaryConv2d_Fusion_Decrease(nn.Module):
    '''
    空间尺寸不变且通道数减半 - Upsample 去掉上采样
    input: b,c,h,w
    output: b,c/2,h,w
    '''
    def __init__(self, inplanes, planes,alpha, beta1,stride=1,groups=1,bias=False):
        super(BinaryConv2d_Fusion_Decrease, self).__init__()
        self.alpha = alpha
        self.beta1 = beta1

        self.move11 = LearnableBias(inplanes)
        self.binary_1x1= binaryconv1x1(inplanes, planes, stride=stride,groups=groups,bias=bias)


        self.move12 = LearnableBias(planes)
        self.prelu1 = nn.PReLU(planes)
        self.move13 = LearnableBias(planes)

        self.binary_activation = BinaryActivation()

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h,w
        '''
        x = x*self.beta1

        out1 = self.move11(x)
        out1 = self.binary_activation(out1)

        out1 = self.binary_1x1(out1)

        out1 = out1*self.alpha


        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        return out1
    
class BinaryConv2d_Fusion_Increase(nn.Module):
    '''
    空间尺寸不变且通道数翻倍 - Downsample 去掉下采样
    input: b,c,h,w
    output: b,2c,h,w
    '''
    def __init__(self, inplanes, planes,alpha, beta1, stride=1, groups=1,bias=False):
        super(BinaryConv2d_Fusion_Increase, self).__init__()

        self.alpha = alpha
        self.beta1 = beta1

        self.move11 = LearnableBias(inplanes)
        self.binary_1x1= binaryconv1x1(inplanes, planes, stride=stride,groups=groups,bias=bias)


        self.move12 = LearnableBias(planes)
        self.prelu1 = nn.PReLU(planes)
        self.move13 = LearnableBias(planes)

        self.binary_activation = BinaryActivation()


    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h,w
        '''
        x = x*self.beta1

        out1 = self.move11(x)
        out1 = self.binary_activation(out1)

        out1 = self.binary_1x1(out1)

        out1 = out1*self.alpha


        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        return out1
    

class BinaryConv2d_Down(nn.Module):
    '''
    降采样且通道数翻倍
    input: b,c,h,w
    output: b,c/2,2h,2w
    '''
    def __init__(self, inplanes, planes, alpha, beta1,stride=1,groups=1,bias=False):
        super(BinaryConv2d_Down, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.alpha = alpha
        self.beta1 = beta1
        self.binary_activation = BinaryActivation()

        self.move21 = LearnableBias(inplanes)

        self.binary_pw_down1 = binaryconv3x3(inplanes, inplanes, stride=stride,groups=groups,bias=bias)
        self.binary_pw_down2 = binaryconv3x3(inplanes, inplanes, stride=stride,groups=groups,bias=bias)

        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h/2,w/2
        '''
        x = self.avg_pool(x)

        out = x*self.beta1

        out2 = self.move21(out)
        out2 = self.binary_activation(out2)

        out2_1 = self.binary_pw_down1(out2)
        out2_2 = self.binary_pw_down2(out2)

        out2_1 = out2_1*self.alpha + x
        out2_2 = out2_2*self.alpha + x
 
        out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2
    

class BinaryConv2d_Up(nn.Module):
    '''
    上采样且通道数减半
    input: b,c,h,w
    output: b,c/2,2h,2w
    '''
    def __init__(self, inplanes, planes, alpha, beta1,stride=1,  bias=False,groups=1):
        super(BinaryConv2d_Up, self).__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.binary_activation = BinaryActivation()


        self.move11 = LearnableBias(inplanes)
        self.binary_3x3= binaryconv3x3(inplanes, planes, stride=stride, groups=groups, bias=bias) #groups不行

        self.move12 = LearnableBias(planes)
        self.prelu1 = nn.PReLU(planes)
        self.move13 = LearnableBias(planes)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,c/2,2h,2w
        '''
        b,c,h,w = x.shape
        out = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = out*self.beta1 

        out1 = self.move11(x)

        out1 = self.binary_activation(out1)
        # stx()
        out1 = self.binary_3x3(out1)

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        return out1
    

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
            BinaryConv2d_Fusion_Increase(dim, dim * mult,0.2,1, 1, bias=False),
            BinaryConv2d_Fusion_Increase(dim * mult, dim * mult * mult,0.2, 1,1, bias=False),
            GELU(),
            FF_BICONV3(dim * mult * mult, dim * mult * mult,0.2,1,1, groups=dim , bias=False),
            GELU(),
            BinaryConv2d_Fusion_Decrease(dim * mult * mult, dim * mult,0.2,1,1, bias=False),
            BinaryConv2d_Fusion_Decrease(dim * mult, dim, 0.2,1,1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)



class BTM_block(nn.Module):
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



class BiUNet_BTM_body(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, stage=2, num_blocks=[2,4,4]):
        super(BiUNet_BTM_body, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        # self.embedding = BinaryConv2d(in_dim, self.dim, 3, 1, 1, bias=False)                           # 1-bit -> 32-bit
        self.act_embedding = BinaryActivation()
        self.embedding = binaryconv3x3(in_dim, self.dim, stride=1,bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                BTM_block(dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                BinaryConv2d_Down(dim_stage, dim_stage * 2, 0.2,1,1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = BTM_block(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                BinaryConv2d_Up(dim_stage, dim_stage // 2, 0.2,1,1, bias=False),
                BinaryConv2d_Fusion_Decrease(dim_stage, dim_stage // 2, 0.2,1,1,bias=False),
                BTM_block(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        # self.mapping = BinaryConv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.act_mapping = BinaryActivation()
        self.mapping = binaryconv3x3(in_dim, self.dim, stride=1, bias=False)                                  # 1-bit -> 32-bit
        
        #### activation function
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.BinAc = BinaryActivation()

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(self.act_mapping(x))

        # Encoder
        fea_encoder = []
        for (BTM_block, FeaDownSample) in self.encoder_layers:
            # stx()
            fea = BTM_block(fea)
            fea_encoder.append(fea)
            # stx()
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, BTM_block) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = BTM_block(fea)

        # Mapping
        out = self.mapping(self.act_mapping(fea)) + x

        return out



class BTM(nn.Module):
    '''
    Only 3 layers are 32-bit conv
    '''
    def __init__(self, in_channels=28, out_channels=28, n_feat=28, stage=3, num_blocks=[1,1,1]):
        super(BTM, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)       # 1-bit -> 32-bit
        modules_body = [BiUNet_BTM_body(dim=n_feat, stage=2, num_blocks=num_blocks) for _ in range(stage)]
        self.fution = nn.Conv2d(56, 28, 1, padding=0, bias=True)                                            # 1-bit -> 32-bit
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)     # 1-bit -> 32-bit

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
#     model = BiUNet_BTM_3L_v2(stage=1,num_blocks=[1,1,1]).cuda()
#     flops = FlopCountAnalysis(model, inputs)
#     n_param = sum([p.nelement() for p in model.parameters()])
#     print(f'GMac:{flops.total()}')
#     print(f'Params:{n_param}')