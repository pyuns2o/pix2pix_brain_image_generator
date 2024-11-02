import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader

class Datasets(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.to_tensor = ToTensor()

        self.sub_ls = np.load(self.data_dir + "/subjects/r_{}_ls.npy".format(self.mode))

    def __len__(self):
        return len(self.sub_ls)

    def __getitem__(self, index):
        t1 = nib.load(self.data_dir + "/{}/T1w.nii.gz".format(self.sub_ls[index], self.sub_ls[index])).get_fdata()
        t2 = nib.load(self.data_dir + "/{}/T2w.nii.gz".format(self.sub_ls[index], self.sub_ls[index])).get_fdata()

        """ min-max """
        t1 = (t1 - t1.min()) / (t1.max() - t1.min())
        t2 = (t2 - t2.min()) / (t2.max() - t2.min())

        """ cropping """
        t1_img = np.zeros((256, 256, 256)); t2_img = np.zeros((256, 256, 256))
        t1_img[:, :, 25:228] = t1[17:-17, 17:-17, :]
        t2_img[:, :, 25:228] = t2[17:-17, 17:-17, :]

        t1_img = self.to_tensor(t1_img)
        t2_img = self.to_tensor(t2_img)

        if self.mode == 'train' or self.mode == 'valid':
            return {'t1_img': t1_img, 't2_img': t2_img}
        elif self.mode == 'test':
            return {'t1_img': t1_img, 't2_img': t2_img, 'sub_id': self.sub_ls[index]}

class ToTensor():
    def __call__(self, img):
        img = img[:, :, :, np.newaxis]
        img = np.transpose(img, (3, 0, 1, 2))
        img = torch.from_numpy(np.array(img)).float()

        return img

