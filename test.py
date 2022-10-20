import argparse
import os
import natsort
import torch
import numpy as np
from os import listdir
import torch.nn as nn
from utils import concat_rgb
from model import LiteISPNet_s as NET

parser = argparse.ArgumentParser(description="Test Script")
parser.add_argument('--input_dir', type=str, default='', help='input path with rgb image')
parser.add_argument('--output_dir', type=str, default='', help='output path with npy raw')
parser.add_argument('--checkpoint', type=str, default='ckpts/p20.pth', help='output path with npy raw')
parser.add_argument("--blevl", type=int, default=0, help="black level")
parser.add_argument("--wlevl", type=int, default=1023, help="black level")

opt = parser.parse_args()
os.makedirs(opt.output_dir, exist_ok=True)

print("===> Loading network parameters")
device = torch.device("cuda")
model = NET()
model.load_state_dict(torch.load(opt.checkpoint)['state_dict_model'])
model = nn.DataParallel(model)
model = model.to(device)
model = model.eval()

print("===> Loading input data")
full_rgb_img = concat_rgb(opt.input_dir)

print("===> Inference")
for rgb_name, rgb_img in full_rgb_img.items():
    img = torch.from_numpy(rgb_img.transpose(2, 0, 1)).unsqueeze(0)
    img = img.cuda()
    with torch.no_grad():
        output = model(img)
    out = output.detach().cpu().numpy().transpose((0, 2, 3, 1))[0]

    out = out * (opt.wlevl-opt.blevl) + opt.blevl
    out = out.round()
    out = np.clip(out, 0, opt.wlevl)
    out = out.astype(np.uint16)

    H, W, C = out.shape
    row_step, col_step = int(W / 8), int(H / 6)

    for i in range(6):
        for j in range(8):
            np.save(os.path.join(opt.output_dir, "{}_{}".format(rgb_name, 8 * i + j)),
                    out[i * col_step: (i + 1) * col_step, j * col_step: (j + 1) * col_step, :])

print("===> Done")

