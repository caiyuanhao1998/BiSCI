import argparse

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")


# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='3')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')


# Saving specifications
parser.add_argument('--outf', type=str, default='./results/BiSRNet/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='bisrnet', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default='model_zoo/bisrnet.pth', help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT or None')

opt = parser.parse_args()

opt.input_setting = 'H'
opt.input_mask = 'Mask'
opt.scheduler = 'CosineAnnealingLR'

opt.mask_path = f"{opt.data_root}/TSA_simu_data/"
opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False
