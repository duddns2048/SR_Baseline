from torch.utils.data import Dataset
import pickle
import random
import numpy as np
import glob, os

class LR_dataset(Dataset):
    def __init__(self, dir_path = './COLON/TRAIN_SLICES/HR2', SR_factor=2, mode='train', transform = None):
        self.dir_path = dir_path
        self.HR_files = sorted(glob.glob(os.path.join(dir_path, '*.pt')))
        self.SR_factor = SR_factor
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.HR_files = self.HR_files[:97280] # train 190 case
        elif self.mode == 'valid':
            self.HR_files = self.HR_files[97280:128000] # valid 60 case
        else:
            self.HR_files = self.HR_files[128000:] # test 80 case
            

    def __len__(self):
        return len(self.HR_files)
    
    def __getitem__(self,idx):
        HR_file = self.HR_files[idx]
        f = pickle.load(open(HR_file, 'rb'))
        HR_img = f['image']
        spacing = f['spacing']

        start = random.randint(0, HR_img.shape[1]-60)

        HR_img = HR_img[:, start:start+60]
        img_lr = np.zeros_like(HR_img)

        for i in range(60//self.SR_factor):
            for j in range(self.SR_factor):
                img_lr[:,self.SR_factor*i+j] = HR_img[:,self.SR_factor*i]

        if self.transform:
            img_lr = self.transform(img_lr.astype(np.float32))
            HR_img = self.transform(HR_img.astype(np.float32))         

        return [img_lr, HR_img]