import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset


class ImageProcess(object):
    def __init__(self, users_img_file, train_data_file):
        super(ImageProcess, self).__init__()
        self.users_img_file = users_img_file
        self.train_data_file = train_data_file

    def rename_img(self):
        for user_file in os.listdir(self.users_img_file):
            for user_img in os.listdir(self.users_img_file + user_file):
                # number = random.randint(1000, 10000) 保存的时候就已经是数字名字了
                os.rename(self.users_img_file + user_file + '/' + user_img,
                          self.users_img_file + user_file + '/' + user_file + '/' + user_img)

    def resize_img(self, width=100, height=100):
        for user_file in os.listdir(self.users_img_file):
            for user_img in os.listdir(self.users_img_file + user_file):
                img_path = self.users_img_file + user_file + '/' + user_img
                img = Image.open(img_path)
                img = img.convert('L')
                resized_img = img.resize((width, height))
                resized_img.save(self.train_data_file + user_img)

    def img_to_np(self, img):
        return np.array(img)

    def loader(self):
        img_list = []
        img_label = []
        for train_img in os.listdir(self.train_data_file):
            train_op = Image.open(self.train_data_file + train_img)
            train_np = self.img_to_np(train_op)
            img_list.append(train_np)

            img_label.append(int(train_img.split('_')[0]))

        img_tensor = torch.from_numpy(np.array(img_list).reshape((-1, 1, 100, 100))) / 255.0
        # print(img_tensor.shape)
        img_lb_ter = torch.from_numpy(np.array(img_label)).long()
        # print(img_lb_ter.shape)
        train_dt = TensorDataset(img_tensor, img_lb_ter)
        train_loader = DataLoader(train_dt, batch_size=5, shuffle=True)
        return train_loader
