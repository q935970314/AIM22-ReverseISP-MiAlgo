import os
import numpy as np
import cv2
from natsort import natsorted
from glob import glob


def load_img(filename, norm=True):
    img = cv2.imread(filename).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if norm:
        img = img / 255.
    return img


def concat_rgb(input_path):
    row, col = [], []
    data_lis = {}
    for img_path in natsorted(glob(os.path.join(input_path, "*.jpg"))):
        img = load_img(img_path)

        row.append(img)
        if len(row) == 8:
            col.append(np.concatenate(row, axis=1))
            row = []
        if len(col) == 6:
            out_img = np.concatenate(col, axis=0)
            col = []
            data_lis[img_path.split("/")[-1].split("_")[0]] = out_img
    return data_lis