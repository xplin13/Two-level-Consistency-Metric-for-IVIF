import os
import torch
from torch.utils.data.dataset import Dataset
import cv2

class dataset_gray(Dataset):

    def __init__(self, data_dir, split, have_label, input_h=480, input_w=640 ,transform=[]):
        super(dataset_gray,self).__init__()

        assert split in ['train', 'test'], 'split must be "train"|"val"|"test"|"val_test"'

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]



        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.is_train  = have_label
        if 'train' not in  split and  split!='val'  and split!='test':
            self.n_data    = len(self.ir_name)
        else:
            self.n_data    = len(self.names)

    def read_image(self, name, folder_img):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder_img, name))
        image = cv2.imread(file_path,-1)
        [b,g,r,a] = cv2.split(image)
        img_rgb = cv2.merge([b,g,r])
        img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
        img_ir = a

        return img_gray, img_ir

    def read_image_test(self, name, folder_img_0, folder_img_1):
        file_path_0 = os.path.join(self.data_dir, '%s/VIS%s.png' % (folder_img_0, name))
        file_path_1 = os.path.join(self.data_dir, '%s/IR%s.png' % (folder_img_1, name))
        image_0 = cv2.imread(file_path_0)
        image_1 = cv2.imread(file_path_1)
        img_gray = cv2.cvtColor(image_0,cv2.COLOR_BGR2GRAY)
        img_ir = cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)

        return img_gray, img_ir

    def get_train_item(self, index):
        name  = self.names[index]
        image_gray, image_ir = self.read_image(name, 'images')
        if self.transform is not None:
            for func in self.transform:
                image_gray,image_ir = func(image_gray,image_ir)

        image = image_gray/255.
        image_ir = image_ir/255.


        return torch.tensor(image.copy()).unsqueeze(0).float(), torch.tensor(image_ir.copy()).unsqueeze(0).float(), name

    def get_test_item(self, index):
        name  = self.names[index]
        image, image_ir = self.read_image_test(name, 'vis', 'ir')
        image = cv2.resize(image,(self.input_w, self.input_h))/255
        image_ir = cv2.resize(image_ir,(self.input_w,self.input_h))/255

        return torch.tensor(image).unsqueeze(0).float(), torch.tensor(image_ir).unsqueeze(0).float(), name


    def __getitem__(self, index):
        if self.is_train is True:
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    def __len__(self):
        return self.n_data

