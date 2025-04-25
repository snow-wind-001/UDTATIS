from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 512
        self.train_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input1' or data_name == 'input2' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input1 = cv2.imread(self.datas['input1']['image'][index])
        input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        # load image2
        input2 = cv2.imread(self.datas['input2']['image'][index])
        input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)
        
        #print("fasdf")
        if_exchange = random.randint(0,1)
        if if_exchange == 0:
            #print(if_exchange)
            return (input1_tensor, input2_tensor)
        else:
            #print(if_exchange)
            return (input2_tensor, input1_tensor)

    def __len__(self):

        return len(self.datas['input1']['image'])

class TestDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input1' or data_name == 'input2' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input1 = cv2.imread(self.datas['input1']['image'][index])
        #input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        # load image2
        input2 = cv2.imread(self.datas['input2']['image'][index])
        #input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        return (input1_tensor, input2_tensor)

    def __len__(self):

        return len(self.datas['input1']['image'])



class UDISDataset(Dataset):
    def __init__(self, root_dir, is_train=True, use_virtual=False, width=512, height=512):
        self.width = width
        self.height = height
        self.root_dir = root_dir
        self.is_train = is_train
        self.use_virtual = use_virtual
        
        if use_virtual:
            print("Using virtual dataset for testing")
            self.length = 10  # 创建10个虚拟样本
        else:
            # 实际数据集路径检查
            img1_dir = os.path.join(root_dir, 'img1')
            img2_dir = os.path.join(root_dir, 'img2')
            
            if not os.path.exists(img1_dir) or not os.path.exists(img2_dir):
                print(f"Warning: Directory {img1_dir} or {img2_dir} not found!")
                print(f"Creating virtual dataset as fallback")
                self.use_virtual = True
                self.length = 10
            else:
                # 获取实际数据集长度
                img1_files = glob.glob(os.path.join(img1_dir, '*.jpg')) + glob.glob(os.path.join(img1_dir, '*.png'))
                if len(img1_files) == 0:
                    print(f"Warning: No images found in {img1_dir}")
                    print(f"Creating virtual dataset as fallback")
                    self.use_virtual = True
                    self.length = 10
                else:
                    self.img1_files = sorted(img1_files)
                    self.img2_files = sorted(glob.glob(os.path.join(img2_dir, '*.jpg')) + glob.glob(os.path.join(img2_dir, '*.png')))
                    self.length = len(self.img1_files)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if self.use_virtual:
            # 创建虚拟图像数据
            # 创建一个随机噪声图像作为img1
            img1 = np.random.rand(self.height, self.width, 3).astype(np.float32)
            
            # 为了模拟真实场景，img2应该是img1的变形版本
            # 这里简单地添加一点偏移和噪声
            offset_x = np.random.randint(-20, 20)
            offset_y = np.random.randint(-20, 20)
            
            img2 = np.roll(np.roll(img1, offset_x, axis=1), offset_y, axis=0)
            # 添加一些额外的噪声
            img2 += np.random.normal(0, 0.1, img2.shape).astype(np.float32)
            
            # 确保值在合理范围内
            img1 = np.clip(img1, 0, 1)
            img2 = np.clip(img2, 0, 1)
            
            # 转换到[-1, 1]范围
            img1 = img1 * 2 - 1
            img2 = img2 * 2 - 1
            
            # 转换到tensor格式
            img1 = torch.tensor(np.transpose(img1, [2, 0, 1]))
            img2 = torch.tensor(np.transpose(img2, [2, 0, 1]))
            
            # 创建目标数据
            # 在实际的训练中，这些会从数据集中加载或通过某种方式计算得到
            h_points = np.random.rand(4, 2).astype(np.float32) * 0.1  # 单应性变换的四个角点偏移
            mesh = np.random.rand((12+1)*(12+1), 2).astype(np.float32) * 0.1  # 网格变形，使用13x13网格
            
            homography = torch.tensor(h_points)
            mesh = torch.tensor(mesh)
            
            target = {
                'homography': homography,
                'mesh': mesh
            }
            
            return img1, img2, target
        else:
            # 从实际数据集加载
            img1_path = self.img1_files[index]
            img2_path = self.img2_files[index % len(self.img2_files)]  # 防止索引越界
            
            # 读取图像
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            # 调整大小
            img1 = cv2.resize(img1, (self.width, self.height))
            img2 = cv2.resize(img2, (self.width, self.height))
            
            # 转换为float并归一化到[-1, 1]
            img1 = img1.astype(np.float32) / 127.5 - 1.0
            img2 = img2.astype(np.float32) / 127.5 - 1.0
            
            # 通道转换
            img1 = np.transpose(img1, [2, 0, 1])
            img2 = np.transpose(img2, [2, 0, 1])
            
            # 转换为tensor
            img1 = torch.tensor(img1)
            img2 = torch.tensor(img2)
            
            # 创建随机目标数据用于训练
            # 实际中应该基于图像对之间的关系计算这些值
            h_points = np.random.rand(4, 2).astype(np.float32) * 0.1
            mesh = np.random.rand((12+1)*(12+1), 2).astype(np.float32) * 0.1  # 使用13x13网格
            
            homography = torch.tensor(h_points)
            mesh = torch.tensor(mesh)
            
            target = {
                'homography': homography,
                'mesh': mesh
            }
            
            return img1, img2, target


# 创建虚拟数据目录的函数
def create_virtual_data_directories(base_path):
    """
    创建用于测试的虚拟数据目录结构
    """
    # 创建Warp用的目录
    warp_train_path = os.path.join(base_path, "UDIS-D", "training")
    warp_test_path = os.path.join(base_path, "UDIS-D", "testing")
    
    # 创建Composition用的目录
    comp_train_path = os.path.join(base_path, "warp_results", "training")
    comp_test_path = os.path.join(base_path, "warp_results", "testing")
    
    # 创建所有必要的子目录
    for path in [warp_train_path, warp_test_path]:
        os.makedirs(os.path.join(path, "img1"), exist_ok=True)
        os.makedirs(os.path.join(path, "img2"), exist_ok=True)
    
    for path in [comp_train_path, comp_test_path]:
        os.makedirs(os.path.join(path, "warp1"), exist_ok=True)
        os.makedirs(os.path.join(path, "warp2"), exist_ok=True)
        os.makedirs(os.path.join(path, "mask1"), exist_ok=True)
        os.makedirs(os.path.join(path, "mask2"), exist_ok=True)
    
    # 创建一些虚拟图像用于测试
    create_virtual_images(os.path.join(warp_train_path, "img1"), 5)
    create_virtual_images(os.path.join(warp_train_path, "img2"), 5)
    create_virtual_images(os.path.join(warp_test_path, "img1"), 3)
    create_virtual_images(os.path.join(warp_test_path, "img2"), 3)
    
    create_virtual_images(os.path.join(comp_train_path, "warp1"), 5)
    create_virtual_images(os.path.join(comp_train_path, "warp2"), 5)
    create_virtual_masks(os.path.join(comp_train_path, "mask1"), 5)
    create_virtual_masks(os.path.join(comp_train_path, "mask2"), 5)
    create_virtual_images(os.path.join(comp_test_path, "warp1"), 3)
    create_virtual_images(os.path.join(comp_test_path, "warp2"), 3)
    create_virtual_masks(os.path.join(comp_test_path, "mask1"), 3)
    create_virtual_masks(os.path.join(comp_test_path, "mask2"), 3)
    
    return {
        "warp_train": warp_train_path,
        "warp_test": warp_test_path,
        "comp_train": comp_train_path,
        "comp_test": comp_test_path
    }

def create_virtual_images(directory, count, width=512, height=512):
    """
    在指定目录中创建一些随机图像
    """
    for i in range(count):
        # 创建随机噪声图像
        img = np.random.rand(height, width, 3) * 255
        img = img.astype(np.uint8)
        
        # 添加一些简单的形状以模拟真实图像
        cv2.rectangle(img, 
                     (np.random.randint(width//4), np.random.randint(height//4)), 
                     (np.random.randint(width//2, width), np.random.randint(height//2, height)), 
                     (np.random.randint(255), np.random.randint(255), np.random.randint(255)), 
                     -1)
        
        cv2.circle(img, 
                  (np.random.randint(width), np.random.randint(height)), 
                  np.random.randint(20, 100),
                  (np.random.randint(255), np.random.randint(255), np.random.randint(255)), 
                  -1)
        
        # 保存图像
        cv2.imwrite(os.path.join(directory, f"image_{i+1:06d}.jpg"), img)

def create_virtual_masks(directory, count, width=512, height=512):
    """
    在指定目录中创建一些随机掩码
    """
    for i in range(count):
        # 创建随机二值掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 添加一个随机形状的区域
        x1 = np.random.randint(0, width//3)
        y1 = np.random.randint(0, height//3)
        x2 = np.random.randint(2*width//3, width)
        y2 = np.random.randint(2*height//3, height)
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # 添加一些随机噪声
        noise = np.random.randint(0, 2, (height, width)) * 255
        mask = cv2.bitwise_or(mask, noise & np.random.randint(0, 2) * 255)
        
        # 扩展到3通道(虽然只用第一个通道)
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        
        # 保存掩码
        cv2.imwrite(os.path.join(directory, f"mask_{i+1:06d}.jpg"), mask_3ch)



