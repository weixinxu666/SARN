import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from torchvision.models.vgg import vgg16


def l1_loss(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


class Blur(nn.Module):
    def __init__(self, nc):
        super(Blur, self).__init__()
        self.nc = nc
        kernel = gauss_kernel(kernlen=21, nsig=3, channels=self.nc)
        kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1).cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        if x.size(1) != self.nc:
            raise RuntimeError(
                "The channel of input [%d] does not match the preset channel [%d]" % (x.size(1), self.nc))
        x = F.conv2d(x, self.weight, stride=1, padding=10, groups=self.nc)
        return x


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x1, x2):
        return torch.sum(torch.pow((x1 - x2), 2)).div(2 * x1.size()[0])


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=
        1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k



class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, input, gt):
        perception_loss = self.mse_loss(self.loss_network(input), self.loss_network(gt))
        return perception_loss



class Sar_loss(nn.Module):

    def __init__(self):
        super(Sar_loss, self).__init__()
        self.perception_loss = PerceptionLoss()
        self.perception_loss = self.perception_loss.cuda()
        self.color_loss1 = L_color_loss()
        self.blur = Blur(3)

    def forward(self, out, gt):
        out = out.permute(0,3,1,2)
        gt = gt.permute(0,3,1,2)

        rc_loss = torch.mean(torch.pow((out - gt),2))
        p_loss = self.perception_loss(out, gt)
        loss = rc_loss + p_loss
        return loss


class Sar_loss_color(nn.Module):

    def __init__(self):
        super(Sar_loss_color, self).__init__()
        self.perception_loss = PerceptionLoss()
        self.perception_loss = self.perception_loss.cuda()
        self.color_loss = ColorLoss()
        self.blur = Blur(3)

    def forward(self, out, gt):
        out = out.permute(0,3,1,2).cuda()
        gt = gt.permute(0,3,1,2).cuda()

        rc_loss = torch.mean(torch.pow((out - gt),2))

        p_loss = self.perception_loss(out, gt)

        color_loss = self.color_loss(self.blur(out), self.blur(gt))

        loss = rc_loss + p_loss + 0.0001*color_loss
        return loss


class L_color_loss(nn.Module):
    def __init__(self):
        super(L_color_loss,self).__init__()

    def forward(self,x,gt):
        batch_size = x.size()[0]
        x_r=x[:,0,:,:]
        x_g = x[:, 1, :, :]
        x_b = x[:, 2, :, :]
        gt_r = gt[:, 0, :, :]
        gt_g = gt[:, 1, :, :]
        gt_b = gt[:, 2, :, :]
        l1_loss=torch.nn.MSELoss()
        color_loss=0.299*l1_loss(x_r,gt_r)+0.587*l1_loss(x_g,gt_g)+0.114*l1_loss(x_b,gt_b)

        return color_loss/batch_size
