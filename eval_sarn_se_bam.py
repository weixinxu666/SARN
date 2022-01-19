import os
import torch
import torch.backends.cudnn as cudnn
import argparse
from model import *
from tqdm import tqdm
import glob
import cv2
import tqdm
from torch.utils.data import DataLoader
from dataset_lol import TheDataset


dataset = 'lol'  # lol_real/lol_synthetic/DICM/LIME/MEF/NPE/Your own images

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Use Device: {device}")


parser = argparse.ArgumentParser(description='RetinexNet args setting')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default='3', help='GPU idx')

parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--save_dir', dest='save_dir', default='./my_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/my_test', help='directory for testing inputs')

args = parser.parse_args()


psnr_list = []
ssim_list = []
vif_list = []


# low_img_path = '/home/wxx/code/dataset/Test/{}'.format(dataset)   # lol_real/lol_synthetic/DICM/LIME/MEF/NPE/Your own images
low_img_path = './data/eval/low'

img_name = os.listdir(low_img_path)


name_list = []

low_data_name = glob.glob(low_img_path)
for idx in range(len(low_data_name)-1):
    name = low_data_name[idx].split('/')[-1]
    name_list.append(name)

eval_set = TheDataset(phase='eval')


@torch.no_grad()
def predict(model_path):
    sarn_net = SARNet_fuse_se_all_bam()

    if args.use_gpu:
        sarn_net = sarn_net.cuda()
        cudnn.benchmark = True
        cudnn.enabled = True

    sarn_net.load_state_dict(torch.load(model_path))
    sarn_net.to(device)

    results_list = []
    eval_dataloader = DataLoader(eval_set, batch_size=2, shuffle=False,
                                 num_workers=0, pin_memory=True, drop_last=True)

    for data in tqdm.tqdm(eval_dataloader):
        low_im, high_im = data
        low_im, high_im = low_im.cuda(), high_im.cuda()

        out = sarn_net(low_im)

        out = out.cuda().data.cpu().numpy()
        out0 = out[0, :, :, :]
        out1 = out[1, :, :, :]
        results_list.append(out0)
        results_list.append(out1)

    for i in range(len(img_name)):
        cv2.imwrite('./results/{}/{}'.format(dataset, img_name[i]), cv2.cvtColor(results_list[i]*255.0, cv2.COLOR_BGR2RGB))

    print('finished')



if __name__ == '__main__':

    predict(model_path='./checkpoint/sarn_fuse_se_all_bam.pth')

