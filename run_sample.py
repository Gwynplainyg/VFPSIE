import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from glob import glob
import numpy as np
from torch.utils.data import DataLoader
from core.models.VFPSIE import Model
import time
import argparse
import h5py

def read_Image(path_to_image):
    image = cv2.imread(path_to_image, 1)
    image = np.transpose(image, (2, 0, 1))
    image = image / 255.0
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image


def read_Event(path_to_event):
    grid_voxel = h5py.File(path_to_event, 'r')['data'][:]
    grid_voxel = torch.tensor(grid_voxel, dtype=torch.float32).unsqueeze(0)
    return grid_voxel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_folder_path', type=str, default='./sample_data')
    parser.add_argument('--save_output_dir', type=str, default='./output')
    parser.add_argument('--ckpt_path', type=str, default='pretrained_model/VFPSIE.pth')
    args = parser.parse_args()

    ckpt_path = args.ckpt_path

    # load model
    model = Model().cuda()
    raw_model = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in raw_model.items()})
    model = model.eval()

    # load dataset
    path_to_sample = args.sample_folder_path
    images_root = os.path.join(path_to_sample, "images")
    events_root = os.path.join(path_to_sample, "events")

    image_path_list = sorted(glob(os.path.join(images_root, "*.png")))
    event_path_list = sorted(glob(os.path.join(events_root, "*.hdf")))

    # run test
    path_to_output = args.save_output_dir
    if os.path.exists(path_to_output) == False:
        os.makedirs(path_to_output)
    img0 = read_Image(image_path_list[0]).cuda()
    print("image_path_list:{}".format(len(image_path_list)))

    with torch.no_grad():
        for i in range(len(event_path_list)):
            grid_voxel = read_Event(event_path_list[i])
            grid_voxel = grid_voxel.cuda()
            img_pred = model(img0, grid_voxel)
            image_name = image_path_list[i + 1].split('/')[-1]
            path_to_image = os.path.join(path_to_output, image_name)
            cv2.imwrite(path_to_image, np.array(img_pred[0].permute(1, 2, 0).cpu()) * 255.0)





















