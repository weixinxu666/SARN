import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from blocks import *




def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)



def conv5x5(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)



class BasicBlock_se_all(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(BasicBlock_se_all, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_se:
            self.se = SE( in_chnls=64, ratio=4 )
        else:
            self.se = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.se(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.se(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.se is None:
            out = self.se(out)

        out += residual
        out = self.relu(out)

        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out



class SARNet(nn.Module):
    def __init__(self):
        super(SARNet, self).__init__()
        self.fm = conv3x3(3, 64, stride=1)
        # self.bam1 = BAM(64 * 4)
        # self.bam1 = BAM(64)
        self.res_se1 = BasicBlock(64, 64)
        self.res_se2 = BasicBlock(64, 64)
        self.gen = conv3x3(64, 3, stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        fm = self.fm(x)
        out_res_1 = self.res_se1(fm)
        # out_bam_1 = self.bam1(out_res_1)
        out_res_2 = self.res_se2(out_res_1)
        gen = self.gen(out_res_2)
        out = self.sig(gen)
        out = out.permute(0, 2, 3, 1)
        return out


class SARNet_fuse(nn.Module):
    def __init__(self):
        super(SARNet_fuse, self).__init__()
        self.fm = conv3x3(3, 64, stride=1)
        # self.bam1 = BAM(64 * 4)
        # self.bam1 = BAM(64)
        self.res_se1 = BasicBlock(64, 64)
        self.res_se2 = BasicBlock(64, 64)
        self.gen = conv3x3(64*2, 3, stride=1)
        # self.gen2 = conv3x3(64, 32, stride=1)
        # self.gen3 = conv3x3(32, 3, stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        fm = self.fm(x)
        out_res_1 = self.res_se1(fm)
        # out_bam_1 = self.bam1(out_res_1)
        out_res_2 = self.res_se2(out_res_1)
        gen_input = torch.cat([out_res_2, fm], 1)
        gen = self.gen(gen_input)
        out = self.sig(gen)
        out = out.permute(0, 2, 3, 1)
        return out



class SARNet_fuse_se_all(nn.Module):
    def __init__(self):
        super(SARNet_fuse_se_all, self).__init__()
        self.fm = conv3x3(3, 64, stride=1)
        # self.bam1 = BAM(64 * 4)
        # self.bam1 = BAM(64)
        self.res_se1 = BasicBlock_se_all(64, 64)
        self.res_se2 = BasicBlock_se_all(64, 64)
        self.gen = conv3x3(64*2, 3, stride=1)
        # self.gen2 = conv3x3(64, 32, stride=1)
        # self.gen3 = conv3x3(32, 3, stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        fm = self.fm(x)
        out_res_1 = self.res_se1(fm)
        # out_bam_1 = self.bam1(out_res_1)
        out_res_2 = self.res_se2(out_res_1)
        gen_input = torch.cat([out_res_2, fm], 1)
        gen = self.gen(gen_input)
        out = self.sig(gen)
        out = out.permute(0, 2, 3, 1)
        return out


class SARNet_bam(nn.Module):
    def __init__(self):
        super(SARNet_bam, self).__init__()
        self.fm = conv3x3(3, 64, stride=1)
        # self.bam1 = BAM(64 * 4)
        self.bam1 = BAM(64)
        self.res_se1 = BasicBlock(64, 64)
        self.res_se2 = BasicBlock(64, 64)
        self.gen = conv3x3(64, 3, stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        fm = self.fm(x)
        out_res_1 = self.res_se1(fm)
        out_bam_1 = self.bam1(out_res_1)
        out_res_2 = self.res_se2(out_bam_1)
        gen = self.gen(out_res_2)
        out = self.sig(gen)
        out = out.permute(0, 2, 3, 1)
        return out



class SARNet_fuse_se_all_bam(nn.Module):
    def __init__(self):
        super(SARNet_fuse_se_all_bam, self).__init__()
        self.fm = conv3x3(3, 64, stride=1)
        # self.bam1 = BAM(64 * 4)
        self.bam1 = BAM(64)
        self.res_se1 = BasicBlock_se_all(64, 64)
        self.res_se2 = BasicBlock_se_all(64, 64)
        self.gen = conv3x3(64*2, 3, stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        fm = self.fm(x)
        out_res_1 = self.res_se1(fm)
        out_bam_1 = self.bam1(out_res_1)
        out_res_2 = self.res_se2(out_bam_1)
        gen_input = torch.cat([out_res_2, fm], 1)
        gen = self.gen(gen_input)
        out = self.sig(gen)
        out = out.permute(0, 2, 3, 1)
        return out


if __name__ == '__main__':
    img = torch.randn(4, 384, 384, 3)

    model = SARNet_fuse_se_all()
    out = model(img)
    print(out.shape)