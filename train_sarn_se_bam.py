import os
import torch
import torch.backends.cudnn as cudnn
import argparse
from model import *
from dataset_lol import TheDataset
from loss import *
import tqdm
from torch.utils.data import DataLoader
from glob import glob
from utils import calculate_psnr
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Use Device: {device}")


eval_every_epoch = 1

name_list = []


# Change the eval dir before training
low_data_name = glob('./data/eval/low/*')


for idx in range(len(low_data_name)-1):
    name = low_data_name[idx].split('/')[-1]
    name_list.append(name)


parser = argparse.ArgumentParser(description='RetinexNet args setting')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default='1', help='GPU idx')
parser.add_argument('--phase', dest='phase', default='train', help='train or lime_data_test')

parser.add_argument('--epoch', dest='epoch', type=int, default=300, help='number of total epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=6, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=384, help='patch size')
parser.add_argument('--workers', dest='workers', type=int, default=0, help='num workers of dataloader')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.0005, help='initial learning rate for adam')
# parser.add_argument('--save_interval', dest='save_interval', type=int, default=5, help='save model every # epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')

args = parser.parse_args()

if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)


# sar_net = SARNet_fuse_se_all()
sar_net = SARNet_fuse_se_all_bam()


if args.use_gpu:
    sar_net = sar_net.cuda()

    cudnn.benchmark = True
    cudnn.enabled = True

lr = args.start_lr * np.ones([args.epoch])
lr[20:] = lr[0] / 10.0


sarn_optim = torch.optim.Adam(sar_net.parameters(), lr=args.start_lr)


train_set = TheDataset(phase='train')
eval_set = TheDataset(phase='eval')


sarn_loss = Sar_loss()


def train():
    save_thres = 0

    sar_net.train()


    for epoch in range(args.epoch):
        times_per_epoch, sum_loss = 0, 0.

        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True, drop_last=True)
        sarn_optim.param_groups[0]['lr'] = lr[epoch]

        for data in tqdm.tqdm(dataloader):
            times_per_epoch += 1
            low_im, high_im = data
            low_im, high_im = low_im.cuda(), high_im.cuda()
            sarn_optim.zero_grad()
            out = sar_net(low_im)
            loss = sarn_loss(out, high_im)
            loss.backward()
            sarn_optim.step()
            sum_loss += loss
        print('epoch: ' + str(epoch) + ' | loss_sarn: ' + str(sum_loss / times_per_epoch))


        if (epoch + 1) % eval_every_epoch == 0:
            total_psnr = 0
            results_list = []
            eval_dataloader = DataLoader(eval_set, batch_size=2, shuffle=False,
                                    num_workers=args.workers, pin_memory=True, drop_last=True)

            for data in tqdm.tqdm(eval_dataloader):
                times_per_epoch += 1
                low_im, high_im = data
                low_im, high_im = low_im.cuda(), high_im.cuda()

                out = sar_net(low_im)

                out = out.cuda().data.cpu().numpy()
                high_im = high_im.cuda().data.cpu().numpy()
                out0 = out[0,:,:,:]
                out1 = out[1,:,:,:]
                high_im0 = high_im[0,:,:,:]
                high_im1 = high_im[1,:,:,:]
                psnr_val0 = calculate_psnr(out0, high_im0)
                psnr_val1 = calculate_psnr(out1, high_im1)
                psnr_val = (psnr_val0 + psnr_val1)/2

                total_psnr += psnr_val

                results_list.append(out0)
                results_list.append(out1)

            if save_thres < total_psnr / len(name_list):
                save_thres = total_psnr / len(name_list)
                # print('------PSNR------: ', total_psnr / len(low_data_name))
                torch.save(sar_net.state_dict(), args.ckpt_dir + '/sarn_fuse_se_all_bam.pth')
                print('-----Model saved-----')
                for i in range(len(name_list)):
                    cv2.imwrite('./results/lol/{}'.format(name_list[i]), cv2.cvtColor(np.squeeze(results_list[i]*255.0), cv2.COLOR_BGR2RGB))
                    print('Image saved: {}'.format(name_list[i]))


if __name__ == '__main__':
    train()
