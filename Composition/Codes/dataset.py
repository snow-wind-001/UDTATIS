from torch.utils.data import Dataset
import numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random
import time
import traceback
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


def safe_mask_to_tensor(mask_array):
    """
    安全地将掩码数组转换为张量，处理各种可能的维度和稀疏问题
    
    参数:
        mask_array: numpy数组形式的掩码
        
    返回:
        适当维度和格式的torch张量掩码
    """
    try:
        # 确保是3维的
        if len(mask_array.shape) == 2:
            mask_array = mask_array[:, :, np.newaxis]
            
        # 转置为通道优先格式
        if mask_array.shape[2] == 1:
            mask_tensor_array = np.transpose(mask_array, [2, 0, 1])
        else:
            # 如果是多通道的，只取第一个通道
            mask_tensor_array = mask_array[:, :, 0:1]
            mask_tensor_array = np.transpose(mask_tensor_array, [2, 0, 1])
            
        # 转换为张量
        mask_tensor = torch.from_numpy(mask_tensor_array)
        
        # 检查是否是稀疏张量并处理
        if hasattr(mask_tensor, 'is_sparse') and mask_tensor.is_sparse:
            mask_tensor = mask_tensor.to_dense()
            
        return mask_tensor
    except Exception as e:
        print(f"掩码转换错误: {e}")
        # 创建一个安全的后备掩码
        h, w = mask_array.shape[:2]
        return torch.ones((1, h, w), dtype=torch.float32)


class TrainDataset(Dataset):
    def __init__(self, data_path, image_size=512, augment=True, norm_type='imagenet', is_test=False, use_virtual=False, overlap_based_stitching=False):
        """
        Composition训练数据集
        
        参数:
            data_path: 数据路径
            image_size: 输入图像尺寸，默认修改为512
            augment: 是否使用数据增强
            norm_type: 归一化类型，'imagenet'或'normal'
            is_test: 是否为测试模式
            use_virtual: 是否使用虚拟合成数据
            overlap_based_stitching: 是否使用基于重叠区域的切片与拼接
        """
        self.data_path = data_path
        self.image_size = image_size  # 修改默认尺寸为512
        self.augment = augment
        self.norm_type = norm_type
        self.is_test = is_test
        self.use_virtual = use_virtual
        self.overlap_based_stitching = overlap_based_stitching
        
        # 获取数据文件列表
        self.warp1_files = sorted(glob.glob(os.path.join(data_path, 'warp1', '*.*')))
        self.warp2_files = sorted(glob.glob(os.path.join(data_path, 'warp2', '*.*')))
        self.mask1_files = sorted(glob.glob(os.path.join(data_path, 'mask1', '*.*')))
        self.mask2_files = sorted(glob.glob(os.path.join(data_path, 'mask2', '*.*'))) 
        
        # 确保数据列表非空且长度匹配
        assert len(self.warp1_files) > 0, f"No files found in {os.path.join(data_path, 'warp1')}"
        assert len(self.warp1_files) == len(self.warp2_files) == len(self.mask1_files) == len(self.mask2_files), \
               "Number of files in warp1, warp2, mask1, mask2 directories should be the same"
        
        self.num_samples = len(self.warp1_files)
        print(f"找到 {self.num_samples} 对训练样本")
        print(f"使用重叠区域切片: {self.overlap_based_stitching}")
        
        # 定义预处理变换
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            self._get_normalize_transform()
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def _get_normalize_transform(self):
        """根据norm_type获取归一化变换"""
        if self.norm_type == 'imagenet':
            # ImageNet标准化
            return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            # [-1, 1]标准化
            return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    def find_overlap_region(self, mask1, mask2, threshold=100):
        """
        查找两个掩码图像的重叠区域
        
        参数:
            mask1, mask2: PIL图像掩码
            threshold: 重叠区域像素数量阈值
        
        返回:
            overlap_bbox: 重叠区域边界框 (x1, y1, x2, y2)，若无重叠则返回None
        """
        # 确保两个掩码尺寸一致
        if mask1.size != mask2.size:
            # 统一尺寸到较小的一个，以确保不会丢失mask2的数据
            common_size = (min(mask1.size[0], mask2.size[0]), min(mask1.size[1], mask2.size[1]))
            mask1 = mask1.resize(common_size, Image.LANCZOS)
            mask2 = mask2.resize(common_size, Image.LANCZOS)
        
        # 转换为numpy数组
        mask1_np = np.array(mask1) / 255.0
        mask2_np = np.array(mask2) / 255.0
        
        # 计算重叠区域
        overlap = mask1_np * mask2_np
        overlap_pixels = (overlap > 0.1).sum()
        
        # 如果重叠区域太小，认为没有重叠
        if overlap_pixels < threshold:
            # 在没有明显重叠的情况下，使用中心区域
            h, w = mask1_np.shape[:2]
            return (w//4, h//4, 3*w//4, 3*h//4)  # 返回中间区域
            
        # 找到重叠区域的边界框
        y_indices, x_indices = np.where(overlap > 0.1)
        if len(y_indices) == 0 or len(x_indices) == 0:
            # 如果找不到重叠像素，使用中心区域
            h, w = mask1_np.shape[:2]
            return (w//4, h//4, 3*w//4, 3*h//4)
        
        x1, y1 = np.min(x_indices), np.min(y_indices)
        x2, y2 = np.max(x_indices), np.max(y_indices)
        
        # 扩展边界框以包含更多上下文
        h, w = mask1_np.shape[:2]
        border = max(h, w) // 8  # 边界扩展大小
        
        x1 = max(0, x1 - border)
        y1 = max(0, y1 - border)
        x2 = min(w - 1, x2 + border)
        y2 = min(h - 1, y2 + border)
        
        # 确保边界框足够大
        min_size = 128
        if x2 - x1 < min_size:
            center = (x1 + x2) // 2
            half_size = min_size // 2
            x1 = max(0, center - half_size)
            x2 = min(w - 1, center + half_size)
        
        if y2 - y1 < min_size:
            center = (y1 + y2) // 2
            half_size = min_size // 2
            y1 = max(0, center - half_size)
            y2 = min(h - 1, center + half_size)
        
        return (x1, y1, x2, y2)

    def crop_around_overlap(self, img1, img2, mask1, mask2, target_size=None):
        """
        根据重叠区域裁剪并调整两张图像的大小
        
        参数:
            img1, img2: PIL输入图像
            mask1, mask2: PIL掩码图像
            target_size: 目标输出尺寸，如果没有指定则使用self.image_size
            
        返回:
            裁剪并调整大小后的图像和掩码
        """
        if target_size is None:
            target_size = (self.image_size, self.image_size)
        
        # 制作临时副本用于查找重叠区域，防止修改原始掩码
        temp_mask1 = mask1.copy()
        temp_mask2 = mask2.copy()
        
        # 找到重叠区域
        # 首先确保两个临时掩码有相同的尺寸用于查找重叠
        common_size = (min(temp_mask1.size[0], temp_mask2.size[0]), 
                      min(temp_mask1.size[1], temp_mask2.size[1]))
        temp_mask1_resized = temp_mask1.resize(common_size, Image.LANCZOS)
        temp_mask2_resized = temp_mask2.resize(common_size, Image.LANCZOS)
        
        overlap_bbox = self.find_overlap_region(temp_mask1_resized, temp_mask2_resized)
        
        if overlap_bbox is None:
            # 如果没有找到重叠区域，简单地调整整个图像大小
            img1_resized = img1.resize(target_size, Image.LANCZOS)
            img2_resized = img2.resize(target_size, Image.LANCZOS)
            mask1_resized = mask1.resize(target_size, Image.LANCZOS)
            mask2_resized = mask2.resize(target_size, Image.LANCZOS)
            return img1_resized, img2_resized, mask1_resized, mask2_resized
        
        # 获取重叠边界框的坐标和尺寸
        x1, y1, x2, y2 = overlap_bbox
        
        # 将重叠区域扩展为与目标尺寸相同的宽高比
        target_w, target_h = target_size
        overlap_w, overlap_h = x2 - x1, y2 - y1
        
        # 计算额外需要的宽度和高度
        target_ratio = target_w / target_h
        current_ratio = overlap_w / overlap_h
        
        if current_ratio > target_ratio:
            # 当前区域太宽，需要增加高度
            new_overlap_h = int(overlap_w / target_ratio)
            new_overlap_w = overlap_w
            center_y = (y1 + y2) // 2
            half_h = new_overlap_h // 2
            
            # 确保扩展后的边界框不超出图像范围
            if mask1.size[1] > mask2.size[1]:
                h_max = mask2.size[1]
            else:
                h_max = mask1.size[1]
                
            y1_new = max(0, center_y - half_h)
            y2_new = min(h_max - 1, center_y + half_h)
            
            x1_new, x2_new = x1, x2
        else:
            # 当前区域太高，需要增加宽度
            new_overlap_w = int(overlap_h * target_ratio)
            new_overlap_h = overlap_h
            center_x = (x1 + x2) // 2
            half_w = new_overlap_w // 2
            
            # 确保扩展后的边界框不超出图像范围
            if mask1.size[0] > mask2.size[0]:
                w_max = mask2.size[0]
            else:
                w_max = mask1.size[0]
                
            x1_new = max(0, center_x - half_w)
            x2_new = min(w_max - 1, center_x + half_w)
            
            y1_new, y2_new = y1, y2
        
        # 分别裁剪两张图像和掩码
        # 确保裁剪框不超出原始图像尺寸
        def safe_crop(img, bbox):
            x1, y1, x2, y2 = bbox
            w, h = img.size
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # 确保裁剪区域有效
            if x2 <= x1 or y2 <= y1:
                # 无效的裁剪区域，返回整个图像的缩放
                return img.resize(target_size, Image.LANCZOS)
            
            return img.crop((x1, y1, x2, y2)).resize(target_size, Image.LANCZOS)
        
        # 应用安全裁剪
        bbox = (x1_new, y1_new, x2_new, y2_new)
        img1_cropped = safe_crop(img1, bbox)
        img2_cropped = safe_crop(img2, bbox)
        mask1_cropped = safe_crop(mask1, bbox)
        mask2_cropped = safe_crop(mask2, bbox)
        
        return img1_cropped, img2_cropped, mask1_cropped, mask2_cropped
    
    def get_overlap_based_crop(self, img1, img2, mask1, mask2):
        """
        基于重叠区域的智能裁剪，保留原始尺寸信息用于后续恢复
        
        参数:
            img1, img2: PIL图像对象
            mask1, mask2: PIL掩码对象
            
        返回:
            裁剪后的图像、掩码和裁剪信息
        """
        # 保存原始尺寸
        orig_size1 = img1.size
        orig_size2 = img2.size
        
        # 找到重叠区域
        overlap_bbox = self.find_overlap_region(mask1, mask2)
        x1, y1, x2, y2 = overlap_bbox
        
        # 计算应该裁剪的区域大小，确保为32的倍数
        width = x2 - x1
        height = y2 - y1
        
        # 确保裁剪区域至少为256x256
        if width < 256:
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - 128)
            x2 = min(img1.size[0], center_x + 128)
            width = x2 - x1
        
        if height < 256:
            center_y = (y1 + y2) // 2
            y1 = max(0, center_y - 128)
            y2 = min(img1.size[1], center_y + 128)
            height = y2 - y1
        
        # 调整为32的倍数
        target_width = ((width + 31) // 32) * 32
        target_height = ((height + 31) // 32) * 32
        
        # 确保不超过原始图像大小
        target_width = min(target_width, img1.size[0])
        target_height = min(target_height, img1.size[1])
        
        # 计算应该扩展的区域
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        new_x1 = max(0, center_x - target_width // 2)
        new_y1 = max(0, center_y - target_height // 2)
        new_x2 = min(img1.size[0], new_x1 + target_width)
        new_y2 = min(img1.size[1], new_y1 + target_height)
        
        # 调整x1,y1，确保裁剪区域正好是目标大小
        if new_x2 - new_x1 != target_width:
            if new_x2 == img1.size[0]:  # 如果右边界已经到最大
                new_x1 = max(0, new_x2 - target_width)
            else:
                new_x2 = min(img1.size[0], new_x1 + target_width)
        
        if new_y2 - new_y1 != target_height:
            if new_y2 == img1.size[1]:  # 如果下边界已经到最大
                new_y1 = max(0, new_y2 - target_height)
        else:
                new_y2 = min(img1.size[1], new_y1 + target_height)
        
        # 裁剪图像和掩码
        crop_bbox = (new_x1, new_y1, new_x2, new_y2)
        img1_crop = img1.crop(crop_bbox)
        img2_crop = img2.crop(crop_bbox)
        mask1_crop = mask1.crop(crop_bbox)
        mask2_crop = mask2.crop(crop_bbox)
        
        # 确保裁剪后尺寸为目标尺寸
        if img1_crop.size != (target_width, target_height):
            img1_crop = img1_crop.resize((target_width, target_height), Image.LANCZOS)
            img2_crop = img2_crop.resize((target_width, target_height), Image.LANCZOS)
            mask1_crop = mask1_crop.resize((target_width, target_height), Image.LANCZOS)
            mask2_crop = mask2_crop.resize((target_width, target_height), Image.LANCZOS)
        
        # 如果需要调整到最终的512x512尺寸
        final_crops = (img1_crop, img2_crop, mask1_crop, mask2_crop)
        if target_width != 512 or target_height != 512:
            img1_crop = img1_crop.resize((512, 512), Image.LANCZOS)
            img2_crop = img2_crop.resize((512, 512), Image.LANCZOS)
            mask1_crop = mask1_crop.resize((512, 512), Image.LANCZOS)
            mask2_crop = mask2_crop.resize((512, 512), Image.LANCZOS)
            final_crops = (img1_crop, img2_crop, mask1_crop, mask2_crop)
        
        # 返回裁剪信息
        crop_info = {
            'x1': new_x1, 'y1': new_y1, 'x2': new_x2, 'y2': new_y2,
            'orig_size1': orig_size1, 'orig_size2': orig_size2,
            'crop_width': target_width, 'crop_height': target_height
        }
        
        return final_crops, crop_info
    
    def prepare_original_size_info(self, warp1_path, warp2_path, mask1_path, mask2_path):
        """
        加载原始图像并获取尺寸信息，不进行裁剪或缩放
        
        参数:
            warp1_path, warp2_path, mask1_path, mask2_path: 图像和掩码文件路径
            
        返回:
            原始图像和掩码，以及尺寸信息
        """
        # 打开图像
        warp1 = Image.open(warp1_path).convert('RGB')
        warp2 = Image.open(warp2_path).convert('RGB')
        mask1 = Image.open(mask1_path).convert('L')
        mask2 = Image.open(mask2_path).convert('L')
        
        # 获取重叠区域但不进行裁剪
        overlap_bbox = self.find_overlap_region(mask1, mask2)
        
        # 保存原始尺寸和重叠区域信息
        orig_info = {
            'orig_size1': warp1.size,
            'orig_size2': warp2.size,
            'overlap': overlap_bbox
        }
        
        return warp1, warp2, mask1, mask2, orig_info

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        # 加载图像
        warp1_path = self.warp1_files[index]
        warp2_path = self.warp2_files[index]
        mask1_path = self.mask1_files[index]
        mask2_path = self.mask2_files[index]
        
        try:
            # 根据模式选择处理方法
            if self.overlap_based_stitching:
                # 打开图像
                warp1 = Image.open(warp1_path).convert('RGB')
                warp2 = Image.open(warp2_path).convert('RGB')
                mask1 = Image.open(mask1_path).convert('L')
                mask2 = Image.open(mask2_path).convert('L')
                
                # 基于重叠区域进行裁剪
                (warp1, warp2, mask1, mask2), crop_info = self.get_overlap_based_crop(
                    warp1, warp2, mask1, mask2
                )
                
                # 应用数据增强（如果启用）
                if self.augment and not self.is_test:
                    # 随机水平翻转
                    if random.random() > 0.5:
                        warp1 = TF.hflip(warp1)
                        warp2 = TF.hflip(warp2)
                        mask1 = TF.hflip(mask1)
                        mask2 = TF.hflip(mask2)
                    
                    # 随机亮度和对比度变化（保留几何关系的增强）
                    if random.random() > 0.5:
                        brightness_factor = random.uniform(0.8, 1.2)
                        contrast_factor = random.uniform(0.8, 1.2)
                        warp1 = TF.adjust_brightness(warp1, brightness_factor)
                        warp1 = TF.adjust_contrast(warp1, contrast_factor)
                        warp2 = TF.adjust_brightness(warp2, brightness_factor)
                        warp2 = TF.adjust_contrast(warp2, contrast_factor)
                
                # 转换为张量
                warp1_tensor = self.transform_rgb(warp1)
                warp2_tensor = self.transform_rgb(warp2)
                mask1_tensor = self.transform_mask(mask1)
                mask2_tensor = self.transform_mask(mask2)
                
                # 返回样本数据，包含裁剪信息
                return {
                    'warp1': warp1_tensor,
                    'warp2': warp2_tensor,
                    'mask1': mask1_tensor,
                    'mask2': mask2_tensor,
                    'path': os.path.basename(warp1_path),
                    'crop_info': crop_info,
                    'original_paths': {
                        'warp1_path': warp1_path,
                        'warp2_path': warp2_path,
                        'mask1_path': mask1_path,
                        'mask2_path': mask2_path
                    }
                }
            else:
                # 使用原有的处理方式
                # 打开图像
                warp1 = Image.open(warp1_path).convert('RGB')
                warp2 = Image.open(warp2_path).convert('RGB')
                mask1 = Image.open(mask1_path).convert('L')
                mask2 = Image.open(mask2_path).convert('L')
                
                # 应用智能裁剪
                warp1, warp2, mask1, mask2 = self.crop_around_overlap(
                    warp1, warp2, mask1, mask2, 
                    target_size=(self.image_size, self.image_size)
                )
                
                # 应用数据增强（如果启用）
                if self.augment and not self.is_test:
                    # 随机水平翻转
                    if random.random() > 0.5:
                        warp1 = TF.hflip(warp1)
                        warp2 = TF.hflip(warp2)
                        mask1 = TF.hflip(mask1)
                        mask2 = TF.hflip(mask2)
                    
                    # 随机旋转
                    if random.random() > 0.5:
                        angle = random.uniform(-10, 10)
                        warp1 = TF.rotate(warp1, angle)
                        warp2 = TF.rotate(warp2, angle)
                        mask1 = TF.rotate(mask1, angle)
                        mask2 = TF.rotate(mask2, angle)
                    
                    # 随机亮度和对比度变化
                    if random.random() > 0.5:
                        brightness_factor = random.uniform(0.8, 1.2)
                        contrast_factor = random.uniform(0.8, 1.2)
                        warp1 = TF.adjust_brightness(warp1, brightness_factor)
                        warp1 = TF.adjust_contrast(warp1, contrast_factor)
                        warp2 = TF.adjust_brightness(warp2, brightness_factor)
                        warp2 = TF.adjust_contrast(warp2, contrast_factor)
                
                # 转换为张量
                warp1_tensor = self.transform_rgb(warp1)
                warp2_tensor = self.transform_rgb(warp2)
                mask1_tensor = self.transform_mask(mask1)
                mask2_tensor = self.transform_mask(mask2)
                
                # 返回样本数据
                return {
                    'warp1': warp1_tensor,
                    'warp2': warp2_tensor,
                    'mask1': mask1_tensor,
                    'mask2': mask2_tensor,
                    'path': os.path.basename(warp1_path)
                }
        
        except Exception as e:
            print(f"Error loading sample {index}: {e}")
            # 发生错误时返回第一个样本或随机选择另一个样本
            rand_idx = random.randint(0, self.__len__() - 1)
            return self.__getitem__(rand_idx)

class TestDataset(Dataset):
    def __init__(self, data_path, use_virtual=False, image_size=512, augment=False, norm_type='imagenet', is_test=True, overlap_based_stitching=False):
        """
        测试数据集
        
        参数:
            data_path: 数据路径
            use_virtual: 是否使用虚拟数据
            image_size: 图像尺寸，默认修改为512
            augment: 是否使用数据增强（测试时通常关闭）
            norm_type: 归一化类型
            is_test: 是否为测试模式
            overlap_based_stitching: 是否使用基于重叠区域的切片与拼接
        """
        self.data_path = data_path
        self.image_size = image_size  # 修改默认尺寸为512
        self.augment = augment
        self.norm_type = norm_type
        self.is_test = is_test
        self.use_virtual = use_virtual
        self.overlap_based_stitching = overlap_based_stitching
        
        # 获取数据文件列表
        self.warp1_files = sorted(glob.glob(os.path.join(data_path, 'warp1', '*.*')))
        self.warp2_files = sorted(glob.glob(os.path.join(data_path, 'warp2', '*.*')))
        self.mask1_files = sorted(glob.glob(os.path.join(data_path, 'mask1', '*.*')))
        self.mask2_files = sorted(glob.glob(os.path.join(data_path, 'mask2', '*.*')))
        
        # 确保数据列表非空且长度匹配
        assert len(self.warp1_files) > 0, f"No files found in {os.path.join(data_path, 'warp1')}"
        assert len(self.warp1_files) == len(self.warp2_files) == len(self.mask1_files) == len(self.mask2_files), \
               "Number of files in warp1, warp2, mask1, mask2 directories should be the same"
        
        self.num_samples = len(self.warp1_files)
        print(f"找到 {self.num_samples} 对测试样本")
        print(f"使用重叠区域切片: {self.overlap_based_stitching}")
        
        # 定义预处理变换
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            self._get_normalize_transform()
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def _get_normalize_transform(self):
        """根据norm_type获取归一化变换"""
        if self.norm_type == 'imagenet':
            return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # 使用与TrainDataset相同的方法
    find_overlap_region = TrainDataset.find_overlap_region
    crop_around_overlap = TrainDataset.crop_around_overlap
    get_overlap_based_crop = TrainDataset.get_overlap_based_crop
    prepare_original_size_info = TrainDataset.prepare_original_size_info

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        # 加载图像
        warp1_path = self.warp1_files[index]
        warp2_path = self.warp2_files[index]
        mask1_path = self.mask1_files[index]
        mask2_path = self.mask2_files[index]
        
        try:
            # 根据模式选择处理方法
            if self.overlap_based_stitching:
                # 打开图像
                warp1 = Image.open(warp1_path).convert('RGB')
                warp2 = Image.open(warp2_path).convert('RGB')
                mask1 = Image.open(mask1_path).convert('L')
                mask2 = Image.open(mask2_path).convert('L')
                
                # 基于重叠区域进行裁剪
                (warp1, warp2, mask1, mask2), crop_info = self.get_overlap_based_crop(
                    warp1, warp2, mask1, mask2
                )
                
                # 转换为张量
                warp1_tensor = self.transform_rgb(warp1)
                warp2_tensor = self.transform_rgb(warp2)
                mask1_tensor = self.transform_mask(mask1)
                mask2_tensor = self.transform_mask(mask2)
                
                # 返回样本数据，包含裁剪信息
                return {
                    'warp1': warp1_tensor,
                    'warp2': warp2_tensor,
                    'mask1': mask1_tensor,
                    'mask2': mask2_tensor,
                    'path': os.path.basename(warp1_path),
                    'crop_info': crop_info,
                    'original_paths': {
                        'warp1_path': warp1_path,
                        'warp2_path': warp2_path,
                        'mask1_path': mask1_path,
                        'mask2_path': mask2_path
                    }
                }
            else:
                # 使用原有的处理方式
                # 打开图像
                warp1 = Image.open(warp1_path).convert('RGB')
                warp2 = Image.open(warp2_path).convert('RGB')
                mask1 = Image.open(mask1_path).convert('L')
                mask2 = Image.open(mask2_path).convert('L')
                
                # 应用智能裁剪而不是简单的resize
                warp1, warp2, mask1, mask2 = self.crop_around_overlap(
                    warp1, warp2, mask1, mask2, 
                    target_size=(self.image_size, self.image_size)
                )
                
                # 转换为张量
                warp1_tensor = self.transform_rgb(warp1)
                warp2_tensor = self.transform_rgb(warp2)
                mask1_tensor = self.transform_mask(mask1)
                mask2_tensor = self.transform_mask(mask2)
                
                return {
                    'warp1': warp1_tensor,
                    'warp2': warp2_tensor,
                    'mask1': mask1_tensor,
                    'mask2': mask2_tensor,
                    'path': os.path.basename(warp1_path)
                }
        
        except Exception as e:
            print(f"Error loading test sample {index}: {e}")
            # 发生错误时返回第一个样本或随机选择另一个
            rand_idx = random.randint(0, self.__len__() - 1)
            return self.__getitem__(rand_idx)


