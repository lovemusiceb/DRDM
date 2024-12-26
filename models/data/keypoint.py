import os.path
import torchvision.transforms as transforms
from PIL import Image
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset



def get_transform():
    transform_list = []

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class KeyDataset(Dataset):
    def initialize(self,dataroot='/zy/diffusion5/dataset',phase='train',train_pairLst='/zy/diffusion5/dataset/fashion-resize-pairs-train.csv',test_pairLst='/zy/diffusion5/dataset/fashion-resize-pairs-test.csv'):
        self.phase=phase
        if phase == 'train':
            self.dir_P = os.path.join(dataroot, phase)
            self.dir_K = os.path.join(dataroot, phase + 'K')
            self.dir_conn_map = '/zy/diffusion5/dataset/semantic_merge3'
        elif phase == 'test':
            self.dir_P = os.path.join(dataroot, phase)
            self.dir_K = os.path.join(dataroot, "train" + 'K')
            self.dir_conn_map = '/zy/diffusion5/dataset/semantic_merge3'


        self.dir_SP = dataroot
        self.SP_input_nc = 20
        self.transform = get_transform()

        if phase == 'train':
            self.init_categories_train(train_pairLst)
        elif phase == 'test':
            self.init_categories_test(test_pairLst)


    def init_categories_train(self, train_pairLst):
        pairlist = pd.read_csv(train_pairLst)
        self.size = len(pairlist)
        self.imgs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            img = [pairlist.iloc[i]['from'], pairlist.iloc[i]['to']]
            self.imgs.append(img)
        print('Loading data train_pair finished ...')

    def init_categories_test(self, test_pairLst):
        pairlist = pd.read_csv(test_pairLst)
        self.size = len(pairlist)
        self.imgs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            img = [pairlist.iloc[i]['from'], pairlist.iloc[i]['to']]
            self.imgs.append(img)

        print('Loading data test_pair finished ...')

    def __getitem__(self, index):
        if self.phase == 'train':
            r1=torch.rand(1)
            r2=torch.rand(1)
            # person image
            P1_name, P2_name = self.imgs[index]
            if P1_name=="p1":
                P1= torch.zeros((3, 256,256))
                P11 = torch.zeros((3, 256, 256))

            else:
                if(r1<=0.1):
                    P1= torch.zeros((3, 256,256))
                    P11= torch.zeros((3, 256,256))
                    BP1=torch.zeros((18, 256,256))
                    BP2=torch.zeros((18, 256,256))
                else:
                    P1_path = os.path.join(self.dir_P, P1_name.replace('jpg','png'))
                    P1_img = Image.open(P1_path).convert('RGB')
                    P1_img = P1_img.resize((256, 256))
                    P1 = self.transform(P1_img)
                    P11_img = P1_img.resize((256, 256))
                    P11 = self.transform(P11_img)

                    BP2_path = os.path.join(self.dir_K, P2_name + '.npy')
                    BP2_img = np.load(BP2_path)
                    # PCM2_path = os.path.join("/zy/diffusion5/dataset/embo/dataset/pose_connection_map", P2_name + '.npy')
                    # PCM2_mask = np.load(PCM2_path)
                    BP2 = torch.from_numpy(BP2_img).float()
                    BP2 = BP2.transpose(2, 0)  # c,w,h
                    BP2 = BP2.transpose(2, 1)  # c,h,w
                    # PCM2_mask = torch.from_numpy(PCM2_mask).float()
                    # BP2 = torch.cat([BP2, PCM2_mask], 0)

                    BP1_path = os.path.join(self.dir_K, P1_name + '.npy')
                    BP1_img = np.load(BP1_path)
                    # PCM1_path = os.path.join("/zy/diffusion5/dataset/embo/dataset/pose_connection_map", P1_name + '.npy')
                    # PCM1_mask = np.load(PCM1_path)
                    BP1 = torch.from_numpy(BP1_img).float()
                    BP1 = BP1.transpose(2, 0)  # c,w,h
                    BP1 = BP1.transpose(2, 1)  # c,h,w
                    # PCM1_mask = torch.from_numpy(PCM1_mask).float()
                    # BP1 = torch.cat([BP1, PCM1_mask], 0)


            P2_path = os.path.join(self.dir_P, P2_name.replace('jpg','png'))
            P2_img = Image.open(P2_path).convert('RGB')
            P2_img = P2_img.resize((256, 256))
            P2 = self.transform(P2_img)
            if (r2<=-1 and r1>0.1):
                P1=P2


            # semantic
            if (r2<=-1 and r1>0.1):
                SP1_name = os.path.join( 'semantic_merge3',P2_name)
                SP1_path = os.path.join(self.dir_SP, SP1_name)
                SP1_path = SP1_path[:-4] + '.png'
                SP1_data = Image.open(SP1_path)
                SP1_data =np.array(SP1_data)
                SP1 = np.zeros((8, 256, 256), dtype='float32')
                SP1_20 = np.zeros((self.SP_input_nc, 256, 256), dtype='float32')
                for id in range(self.SP_input_nc):
                    SP1_20[id] = (SP1_data == id).astype('float32')
                SP1[0] = SP1_20[0]# 背景 √
                SP1[1] = SP1_20[9] + SP1_20[12]+ SP1_20[16] + SP1_20[17]+SP1_20[8] #裤子、裙子 、腿、袜子√
                SP1[2] = SP1_20[1]+SP1_20[2]  # 帽子、头发 √
                SP1[3] = SP1_20[3] #手套 √
                SP1[4] = SP1_20[13] + SP1_20[4] # 脸、眼镜 √
                SP1[5] = SP1_20[5] + SP1_20[6] + SP1_20[7] + SP1_20[10] + SP1_20[11] #上衣、连衣裙、外套、背带裤、围巾
                SP1[6] = SP1_20[14] + SP1_20[15] #手臂
                SP1[7] = SP1_20[18] + SP1_20[19] # 袜子、和鞋子
            else:
                SP1_name = os.path.join( 'semantic_merge3',P1_name)
                SP1_path = os.path.join(self.dir_SP, SP1_name)
                SP1_path = SP1_path[:-4] + '.png'
                SP1_data = Image.open(SP1_path)
                SP1_data =np.array(SP1_data)
                SP1 = np.zeros((8, 256, 256), dtype='float32')
                SP1_20 = np.zeros((self.SP_input_nc, 256, 256), dtype='float32')
                for id in range(self.SP_input_nc):
                    SP1_20[id] = (SP1_data == id).astype('float32')
                SP1[0] = SP1_20[0]# 背景 √
                SP1[1] = SP1_20[9] + SP1_20[12]+ SP1_20[16] + SP1_20[17]+SP1_20[8] #裤子、裙子 、腿、袜子√
                SP1[2] = SP1_20[1]+SP1_20[2]  # 帽子、头发 √
                SP1[3] = SP1_20[3] #手套 √
                SP1[4] = SP1_20[13] + SP1_20[4] # 脸、眼镜 √
                SP1[5] = SP1_20[5] + SP1_20[6] + SP1_20[7] + SP1_20[10] + SP1_20[11] #上衣、连衣裙、外套、背带裤、围巾
                SP1[6] = SP1_20[14] + SP1_20[15] #手臂
                SP1[7] = SP1_20[18] + SP1_20[19] # 袜子、和鞋子
                    




            return {'P1': P1, 'SP1': SP1, 'P2': P2, 'BP2': BP2, 'BP1': BP1, 'P1_path': P1_name, 'P2_path': P2_name,'P11':P11} # train ,P1 源人像1 SP1 原人像解析图 P2 目标人像 BP2 目标人像姿势

        elif self.phase == 'test':
            # person image
            P1_name, P2_name = self.imgs[index]
            P1_path = os.path.join(self.dir_P, P1_name.replace('jpg','png'))
            P1_img = Image.open(P1_path).convert('RGB')
            P2_path = os.path.join(self.dir_P, P2_name.replace('jpg','png'))
            P2_img = Image.open(P2_path).convert('RGB')
            P1_img = P1_img
            P2_img = P2_img
            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)
            # pose
            BP2_path = os.path.join(self.dir_K, P2_name + '.npy')
            BP2_img = np.load(BP2_path)
            # PCM2_path = os.path.join("/zy/diffusion5/dataset/embo/dataset/pose_connection_map", P2_name + '.npy')
            # PCM2_mask = np.load(PCM2_path)
            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0)  # c,w,h
            BP2 = BP2.transpose(2, 1)  # c,h,w
            # PCM2_mask = torch.from_numpy(PCM2_mask).float()
            # BP2 = torch.cat([BP2, PCM2_mask], 0)



            # semantic
            SP1_name = os.path.join( 'semantic_merge3',P1_name)
            SP1_path = os.path.join(self.dir_SP, SP1_name)
            SP1_path = SP1_path[:-4] + '.png'
            SP1_data = Image.open(SP1_path)
            SP1_data =np.array(SP1_data)
            SP1 = np.zeros((8, 256, 256), dtype='float32')
            SP1_20 = np.zeros((self.SP_input_nc, 256, 256), dtype='float32')
            for id in range(self.SP_input_nc):
                SP1_20[id] = (SP1_data == id).astype('float32')
            SP1[0] = SP1_20[0]# 背景 √
            SP1[1] = SP1_20[9] + SP1_20[12]+ SP1_20[16] + SP1_20[17]+SP1_20[8] #裤子、裙子 、腿、袜子√
            SP1[2] = SP1_20[1]+SP1_20[2]  # 帽子、头发 √
            SP1[3] = SP1_20[3] #手套 √
            SP1[4] = SP1_20[13] + SP1_20[4] # 脸、眼镜 √
            SP1[5] = SP1_20[5] + SP1_20[6] + SP1_20[7] + SP1_20[10] + SP1_20[11] #上衣、连衣裙、外套、背带裤、围巾
            SP1[6] = SP1_20[14] + SP1_20[15] #手臂
            SP1[7] = SP1_20[18] + SP1_20[19] # 袜子、和鞋子
                    

            return {'P1': P1, 'SP1': SP1, 'P2': P2, 'BP2': BP2, 'P1_path': P1_name, 'P2_path': P2_name,'P11':P1,'BP1': BP2,} # test ,P1 源人像1 SP1 原人像解析图 P2 目标人像 BP2 目标人像姿势

    def __len__(self):
        return self.size

    def name(self):
        return 'KeyDataset'

    def split_name(self, str, type):
        list = []
        list.append(type)
        if (str[len('fashion'):len('fashion') + 2] == 'WO'):
            lenSex = 5
        else:
            lenSex = 3
        list.append(str[len('fashion'):len('fashion') + lenSex])
        idx = str.rfind('id0')
        list.append(str[len('fashion') + len(list[1]):idx])
        id = str[idx:idx + 10]
        list.append(id[:2] + '_' + id[2:])
        pose = str[idx + 10:]
        list.append(pose[:4] + '_' + pose[4:])

        head = ''
        for path in list:
            head = os.path.join(head, path)
        return head
