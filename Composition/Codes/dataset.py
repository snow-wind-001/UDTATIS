from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path, use_virtual=False):
        self.train_path = data_path
        self.use_virtual = use_virtual
        
        if use_virtual:
            print("Using virtual dataset for training")
            self.length = 10  # 创建10个虚拟样本
        else:
            self.datas = OrderedDict()
            datas = glob.glob(os.path.join(self.train_path, '*'))
            for data in sorted(datas):
                data_name = data.split('/')[-1]
                # 支持原始目录结构和新的目录结构
                if data_name in ['warp1', 'warp2', 'mask1', 'mask2']:
                    self.datas[data_name] = {}
                    self.datas[data_name]['path'] = data
                    self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg')) + glob.glob(os.path.join(data, '*.png'))
                    self.datas[data_name]['image'].sort()
            
            # 检查数据是否存在
            if not all(key in self.datas for key in ['warp1', 'warp2', 'mask1', 'mask2']):
                print("Warning: Missing required directories in data path")
                print(f"Found directories: {list(self.datas.keys())}")
                print("Required directories: warp1, warp2, mask1, mask2")
                print("Creating virtual dataset as fallback")
                self.use_virtual = True
                self.length = 10
            elif len(self.datas.get('warp1', {}).get('image', [])) == 0:
                print("Warning: No images found in warp1 directory")
                print("Creating virtual dataset as fallback")
                self.use_virtual = True
                self.length = 10
            else:
                print(f"Found data directories: {list(self.datas.keys())}")
                print(f"Found {len(self.datas['warp1']['image'])} images in warp1")
                self.length = len(self.datas['warp1']['image'])

    def __getitem__(self, index):
        if self.use_virtual:
            return self._get_virtual_item(index)
        else:
            return self._get_real_item(index)
    
    def _get_virtual_item(self, index):
        # 生成虚拟warp图像
        height, width = 512, 512
        warp1 = np.random.rand(height, width, 3).astype(np.float32)
        warp2 = np.random.rand(height, width, 3).astype(np.float32)
        
        # 调整到[-1, 1]范围
        warp1 = warp1 * 2 - 1
        warp2 = warp2 * 2 - 1
        
        # 生成虚拟mask
        mask1 = np.zeros((height, width, 1), dtype=np.float32)
        mask2 = np.zeros((height, width, 1), dtype=np.float32)
        
        # 在随机位置创建mask区域
        x1 = np.random.randint(0, width//2)
        y1 = np.random.randint(0, height//2)
        x2 = np.random.randint(width//2, width)
        y2 = np.random.randint(height//2, height)
        mask1[y1:y2, x1:x2, 0] = 1.0
        
        x1 = np.random.randint(0, width//2)
        y1 = np.random.randint(0, height//2)
        x2 = np.random.randint(width//2, width)
        y2 = np.random.randint(height//2, height)
        mask2[y1:y2, x1:x2, 0] = 1.0
        
        # 转换为tensor格式
        warp1 = np.transpose(warp1, [2, 0, 1])
        warp2 = np.transpose(warp2, [2, 0, 1])
        mask1 = np.transpose(mask1, [2, 0, 1])
        mask2 = np.transpose(mask2, [2, 0, 1])
        
        warp1_tensor = torch.tensor(warp1)
        warp2_tensor = torch.tensor(warp2)
        mask1_tensor = torch.tensor(mask1)
        mask2_tensor = torch.tensor(mask2)
        
        # 随机交换以增加多样性
        if random.randint(0, 1) == 0:
            return (warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
        else:
            return (warp2_tensor, warp1_tensor, mask2_tensor, mask1_tensor)
    
    def _get_real_item(self, index):
        # load image1
        warp1 = cv2.imread(self.datas['warp1']['image'][index])
        warp1 = warp1.astype(dtype=np.float32)
        warp1 = (warp1 / 127.5) - 1.0
        warp1 = np.transpose(warp1, [2, 0, 1])

        # load image2
        warp2 = cv2.imread(self.datas['warp2']['image'][index])
        warp2 = warp2.astype(dtype=np.float32)
        warp2 = (warp2 / 127.5) - 1.0
        warp2 = np.transpose(warp2, [2, 0, 1])

        # load mask1
        mask1 = cv2.imread(self.datas['mask1']['image'][index])
        mask1 = mask1.astype(dtype=np.float32)
        mask1 = np.expand_dims(mask1[:,:,0], 2) / 255
        mask1 = np.transpose(mask1, [2, 0, 1])

        # load mask2
        mask2 = cv2.imread(self.datas['mask2']['image'][index])
        mask2 = mask2.astype(dtype=np.float32)
        mask2 = np.expand_dims(mask2[:,:,0], 2) / 255
        mask2 = np.transpose(mask2, [2, 0, 1])

        # convert to tensor
        warp1_tensor = torch.tensor(warp1)
        warp2_tensor = torch.tensor(warp2)
        mask1_tensor = torch.tensor(mask1)
        mask2_tensor = torch.tensor(mask2)

        if_exchange = random.randint(0,1)
        if if_exchange == 0:
            return (warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
        else:
            return (warp2_tensor, warp1_tensor, mask2_tensor, mask1_tensor)

    def __len__(self):
        return self.length

class TestDataset(Dataset):
    def __init__(self, data_path, use_virtual=False):
        self.test_path = data_path
        self.use_virtual = use_virtual
        
        if use_virtual:
            print("Using virtual dataset for testing")
            self.length = 5  # 创建5个虚拟测试样本
        else:
            self.datas = OrderedDict()

            datas = glob.glob(os.path.join(self.test_path, '*'))
            for data in sorted(datas):
                data_name = data.split('/')[-1]
                # 支持原始目录结构和新的目录结构
                if data_name in ['warp1', 'warp2', 'mask1', 'mask2']:
                    self.datas[data_name] = {}
                    self.datas[data_name]['path'] = data
                    self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg')) + glob.glob(os.path.join(data, '*.png'))
                    self.datas[data_name]['image'].sort()
            
            # 检查数据是否存在
            if not all(key in self.datas for key in ['warp1', 'warp2', 'mask1', 'mask2']):
                print("Warning: Missing required directories in data path")
                print(f"Found directories: {list(self.datas.keys())}")
                print("Required directories: warp1, warp2, mask1, mask2")
                print("Creating virtual dataset as fallback")
                self.use_virtual = True
                self.length = 5
            elif len(self.datas.get('warp1', {}).get('image', [])) == 0:
                print("Warning: No images found in warp1 directory")
                print("Creating virtual dataset as fallback")
                self.use_virtual = True
                self.length = 5
            else:
                print(f"Found data directories: {list(self.datas.keys())}")
                print(f"Found {len(self.datas['warp1']['image'])} images in warp1")
                self.length = len(self.datas['warp1']['image'])

    def __getitem__(self, index):
        if self.use_virtual:
            return self._get_virtual_item(index)
        else:
            return self._get_real_item(index)
    
    def _get_virtual_item(self, index):
        # 生成虚拟warp图像
        height, width = 512, 512
        warp1 = np.random.rand(height, width, 3).astype(np.float32)
        warp2 = np.random.rand(height, width, 3).astype(np.float32)
        
        # 调整到[-1, 1]范围
        warp1 = warp1 * 2 - 1
        warp2 = warp2 * 2 - 1
        
        # 生成虚拟mask
        mask1 = np.zeros((height, width, 1), dtype=np.float32)
        mask2 = np.zeros((height, width, 1), dtype=np.float32)
        
        # 在随机位置创建mask区域
        x1 = np.random.randint(0, width//2)
        y1 = np.random.randint(0, height//2)
        x2 = np.random.randint(width//2, width)
        y2 = np.random.randint(height//2, height)
        mask1[y1:y2, x1:x2, 0] = 1.0
        
        x1 = np.random.randint(0, width//2)
        y1 = np.random.randint(0, height//2)
        x2 = np.random.randint(width//2, width)
        y2 = np.random.randint(height//2, height)
        mask2[y1:y2, x1:x2, 0] = 1.0
        
        # 转换为tensor格式
        warp1 = np.transpose(warp1, [2, 0, 1])
        warp2 = np.transpose(warp2, [2, 0, 1])
        mask1 = np.transpose(mask1, [2, 0, 1])
        mask2 = np.transpose(mask2, [2, 0, 1])
        
        warp1_tensor = torch.tensor(warp1)
        warp2_tensor = torch.tensor(warp2)
        mask1_tensor = torch.tensor(mask1)
        mask2_tensor = torch.tensor(mask2)
        
        return (warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
    
    def _get_real_item(self, index):
        # load image1
        warp1 = cv2.imread(self.datas['warp1']['image'][index])
        warp1 = warp1.astype(dtype=np.float32)
        warp1 = (warp1 / 127.5) - 1.0
        warp1 = np.transpose(warp1, [2, 0, 1])

        # load image2
        warp2 = cv2.imread(self.datas['warp2']['image'][index])
        warp2 = warp2.astype(dtype=np.float32)
        warp2 = (warp2 / 127.5) - 1.0
        warp2 = np.transpose(warp2, [2, 0, 1])

        # load mask1
        mask1 = cv2.imread(self.datas['mask1']['image'][index])
        mask1 = mask1.astype(dtype=np.float32)
        mask1 = np.expand_dims(mask1[:,:,0], 2) / 255
        mask1 = np.transpose(mask1, [2, 0, 1])

        # load mask2
        mask2 = cv2.imread(self.datas['mask2']['image'][index])
        mask2 = mask2.astype(dtype=np.float32)
        mask2 = np.expand_dims(mask2[:,:,0], 2) / 255
        mask2 = np.transpose(mask2, [2, 0, 1])

        # convert to tensor
        warp1_tensor = torch.tensor(warp1)
        warp2_tensor = torch.tensor(warp2)
        mask1_tensor = torch.tensor(mask1)
        mask2_tensor = torch.tensor(mask2)

        return (warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

    def __len__(self):
        return self.length


