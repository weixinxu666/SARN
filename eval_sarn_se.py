import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
from model import *
from tqdm import tqdm
import cv2
import time


dataset = 'lol'   # lol_real/lol_synthetic/DICM/LIME/MEF/NPE

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


parser = argparse.ArgumentParser(description='RetinexNet args setting')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--save_dir', dest='save_dir', default='./my_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/my_test', help='directory for testing inputs')

args = parser.parse_args()


psnr_list = []
ssim_list = []
vif_list = []


# low_img_path = '/home/wxx/code/dataset/Test/{}'.format(dataset)
low_img_path = './data/eval/low'


img_name = os.listdir(low_img_path)


@torch.no_grad()
def predict(model_path):
    # sarn_net = SARNet()
    # sarn_net = SARNet_fuse()
    sarn_net = SARNet_fuse_se_all()
    # sarn_net = SARNet_fuse_se_all_bam()

    if args.use_gpu:
        sarn_net = sarn_net.cuda()
        cudnn.benchmark = True
        cudnn.enabled = True
        sarn_net.load_state_dict(torch.load(model_path))
    else:
        sarn_net.load_state_dict(torch.load(model_path, map_location='cpu'))


    for name in tqdm(img_name):
        test_low_img = cv2.imread(os.path.join(low_img_path, name))
        test_low_img = np.array(test_low_img, dtype="float32") / 255.0

        input_tensor = torch.from_numpy(test_low_img)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.cuda()

        start_time = time.time()
        out_tensor = sarn_net(input_tensor)

        end_time = time.time()
        print('Time cost = {}'.format(end_time - start_time))

        out_tensor = out_tensor.squeeze(0)
        out = out_tensor.cpu().numpy()
        out = out*255.0

        w, h, _ = test_low_img.shape

        r_low_l_delta = cv2.resize(out, (h, w))

        assert r_low_l_delta.shape == test_low_img.shape

        cv2.imwrite('./results/{}/{}'.format(dataset, name), r_low_l_delta)

    print('finished')



if __name__ == '__main__':

    predict(model_path='./checkpoint/sarn_fuse_se_all.pth')
