import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import cv2
from PIL import Image

class FeatureVisualizer:
    """用于可视化网络特征图的工具类"""
    
    def __init__(self, save_dir='output', max_images=8):
        """
        初始化可视化工具
        
        参数:
            save_dir: 保存图像的目录
            max_images: 一个批次中最多可视化的图像数量
        """
        self.save_dir = save_dir
        self.max_images = max_images
        os.makedirs(save_dir, exist_ok=True)
    
    def tensor_to_numpy(self, tensor):
        """将张量转换为numpy数组"""
        # 将张量移动到CPU并转换为numpy
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu()
            # 如果是单通道，增加通道维度
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(1)
            # 确保范围在[0,1]之间
            if tensor.min() < 0:
                tensor = (tensor + 1) / 2  # 假设范围是[-1,1]
            elif tensor.max() > 1:
                tensor = tensor / 255.0  # 假设范围是[0,255]
            return tensor.numpy()
        return tensor
    
    def visualize_tensor(self, tensor, name, step=0, normalize=True, nrow=None, original_sizes=None, preserve_resolution=False):
        """
        可视化并保存张量
        
        参数:
            tensor: 要可视化的张量 [B, C, H, W]
            name: 保存图像的名称
            step: 当前步骤/迭代次数
            normalize: 是否将像素值标准化到[0,1]
            nrow: 每行显示的图像数量
            original_sizes: 原始图像尺寸列表 [(width, height), ...]
            preserve_resolution: 是否保留原始分辨率
        """
        if tensor is None:
            return
            
        # 转换为numpy
        tensor = self.tensor_to_numpy(tensor)
        
        # 限制批次大小
        b = min(tensor.shape[0], self.max_images)
        tensor = tensor[:b]
        
        # 如果保留原始分辨率且提供了原始尺寸
        if preserve_resolution and original_sizes and len(original_sizes) >= b:
            for i in range(b):
                # 获取当前图像
                img = tensor[i].transpose(1, 2, 0)
                
                # 如果是单通道图像，转换为RGB
                if img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                
                # 确保像素值在[0,1]范围内
                if normalize and img.max() > 1.0:
                    img = img / 255.0
                
                # 转换为PIL图像并调整回原始尺寸
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                orig_width, orig_height = original_sizes[i]
                pil_img = pil_img.resize((orig_width, orig_height), Image.BICUBIC)
                
                # 保存图像
                filename = f"{name}_{i}_{step}.png"
                filepath = os.path.join(self.save_dir, filename)
                pil_img.save(filepath)
                print(f"以原始分辨率({orig_width}x{orig_height})保存图像到: {filepath}")
            
            return
        
        # 创建网格
        if nrow is None:
            nrow = int(np.ceil(np.sqrt(b)))
        
        # 使用make_grid生成网格图像
        grid = make_grid(torch.from_numpy(tensor), nrow=nrow, normalize=normalize, padding=2)
        grid = grid.numpy().transpose((1, 2, 0))
        
        # 保存图像
        filename = f"{name}_{step}.png"
        filepath = os.path.join(self.save_dir, filename)
        
        # 如果是单通道图像，转换为RGB
        if grid.shape[2] == 1:
            grid = np.repeat(grid, 3, axis=2)
        
        plt.figure(figsize=(12, 12))
        plt.imshow(grid)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"保存图像到: {filepath}")
        return filepath
    
    def visualize_feature_maps(self, feature_maps, name, step=0, max_features=64):
        """
        可视化特征图(多通道)
        
        参数:
            feature_maps: 特征图张量 [B, C, H, W]
            name: 保存图像的名称
            step: 当前步骤/迭代次数
            max_features: 最多显示的特征图数量
        """
        if feature_maps is None or not isinstance(feature_maps, torch.Tensor):
            return
            
        # 转换到CPU
        feature_maps = feature_maps.detach().cpu()
        
        # 只获取第一个批次的特征图
        if feature_maps.dim() == 4:
            feature_maps = feature_maps[0]  # [C, H, W]
        
        # 限制特征数量
        c = min(feature_maps.shape[0], max_features)
        feature_maps = feature_maps[:c]
        
        # 标准化每个特征图
        normalized_maps = []
        for i in range(c):
            feature_map = feature_maps[i].numpy()
            if feature_map.max() > feature_map.min():  # 避免除零
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            normalized_maps.append(feature_map)
        
        # 创建网格
        nrows = int(np.ceil(np.sqrt(c)))
        ncols = int(np.ceil(c / nrows))
        
        plt.figure(figsize=(ncols * 2, nrows * 2))
        for i in range(c):
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(normalized_maps[i], cmap='viridis')
            plt.axis('off')
        
        plt.tight_layout()
        filename = f"{name}_features_{step}.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"保存特征图到: {filepath}")
        return filepath
    
    def visualize_masks(self, masks, name, step=0, colormap=cv2.COLORMAP_JET):
        """
        可视化掩码并应用颜色映射
        
        参数:
            masks: 掩码张量 [B, 1, H, W] 或 [B, H, W]
            name: 保存图像的名称
            step: 当前步骤/迭代次数
            colormap: OpenCV颜色映射
        """
        if masks is None:
            return
            
        # 转换为numpy
        masks = self.tensor_to_numpy(masks)
        
        # 打印原始掩码形状用于调试
        print(f"掩码形状: {masks.shape}")
        
        # 确保掩码为3维 [B, H, W]
        if masks.ndim == 4:
            if masks.shape[1] == 1:
                # [B, 1, H, W] -> [B, H, W]
                masks = masks.squeeze(1)
            elif masks.shape[1] == 3:
                # 如果是3通道，取平均值转为单通道
                masks = masks.mean(axis=1)
        
        # 标准化到[0,1]
        if np.max(masks) > 1.0 or np.min(masks) < 0.0:
            masks = (masks - np.min(masks)) / (np.max(masks) - np.min(masks) + 1e-8)
        
        # 转换为uint8以应用颜色映射
        masks_uint8 = (masks * 255).astype(np.uint8)
        
        # 限制批次大小
        b = min(masks.shape[0], self.max_images)
        masks_uint8 = masks_uint8[:b]
        
        # 应用颜色映射并创建可视化
        colored_masks = []
        for i in range(b):
            mask = masks_uint8[i]
            
            # 确保mask是单通道2D数组
            if mask.ndim != 2:
                print(f"警告: 掩码 {i} 的维度是 {mask.ndim}，而不是2。形状: {mask.shape}")
                if mask.ndim == 3:
                    if mask.shape[0] == 1:
                        # [1, H, W] -> [H, W]
                        mask = mask[0]
                    elif mask.shape[0] == 3:
                        # 如果是3通道，取平均值
                        mask = mask.mean(axis=0).astype(np.uint8)
                    else:
                        # 未知格式，使用第一个通道
                        mask = mask[0]
                else:
                    # 创建空白掩码
                    print(f"无法处理掩码维度 {mask.ndim}，使用空白掩码")
                    mask = np.zeros((masks_uint8.shape[1], masks_uint8.shape[2]), dtype=np.uint8)
            
            try:
                # 应用颜色映射
                colored_mask = cv2.applyColorMap(mask, colormap)
                # 转换BGR到RGB
                colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
                colored_masks.append(torch.from_numpy(colored_mask).permute(2, 0, 1) / 255.0)
            except Exception as e:
                print(f"应用颜色映射时出错: {e}，掩码形状: {mask.shape}，类型: {mask.dtype}")
                # 使用备用可视化方法
                plt.figure()
                plt.imshow(mask, cmap='jet')
                plt.axis('off')
                filename = f"{name}_mask_{i}_{step}.png"
                filepath = os.path.join(self.save_dir, filename)
                plt.savefig(filepath)
                plt.close()
                
                # 创建伪彩色掩码以添加到网格
                colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                colored_mask[:,:,0] = mask  # 设置红色通道
                colored_masks.append(torch.from_numpy(colored_mask).permute(2, 0, 1) / 255.0)
        
        # 如果没有有效的掩码，直接返回
        if not colored_masks:
            print(f"警告: 没有有效的掩码可视化")
            return
        
        # 创建网格
        colored_masks_tensor = torch.stack(colored_masks)
        grid = make_grid(colored_masks_tensor, nrow=int(np.ceil(np.sqrt(b))), padding=2)
        grid = grid.numpy().transpose((1, 2, 0))
        
        # 保存图像
        filename = f"{name}_masks_{step}.png"
        filepath = os.path.join(self.save_dir, filename)
        
        plt.figure(figsize=(12, 12))
        plt.imshow(grid)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"保存掩码到: {filepath}")
        return filepath
    
    def side_by_side_comparison(self, images1, images2, name, titles=None, step=0):
        """
        创建两组图像的并排比较
        
        参数:
            images1: 第一组图像张量 [B, C, H, W]
            images2: 第二组图像张量 [B, C, H, W]
            name: 保存图像的名称
            titles: 两个图像组的标题
            step: 当前步骤/迭代次数
        """
        if images1 is None or images2 is None:
            return
            
        # 转换为numpy
        images1 = self.tensor_to_numpy(images1)
        images2 = self.tensor_to_numpy(images2)
        
        # 限制批次大小
        b = min(min(images1.shape[0], images2.shape[0]), self.max_images)
        images1 = images1[:b]
        images2 = images2[:b]
        
        # 设置默认标题
        if titles is None:
            titles = ['Images 1', 'Images 2']
        
        # 创建并排比较
        fig, axes = plt.subplots(b, 2, figsize=(10, 5 * b))
        
        # 如果只有一个批次，调整axes结构
        if b == 1:
            axes = np.expand_dims(axes, axis=0)
        
        for i in range(b):
            # 显示第一组图像
            img1 = images1[i].transpose(1, 2, 0)
            if img1.shape[2] == 1:  # 单通道
                img1 = np.repeat(img1, 3, axis=2)
            axes[i, 0].imshow(img1)
            axes[i, 0].set_title(f"{titles[0]} {i+1}")
            axes[i, 0].axis('off')
            
            # 显示第二组图像
            img2 = images2[i].transpose(1, 2, 0)
            if img2.shape[2] == 1:  # 单通道
                img2 = np.repeat(img2, 3, axis=2)
            axes[i, 1].imshow(img2)
            axes[i, 1].set_title(f"{titles[1]} {i+1}")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        filename = f"{name}_comparison_{step}.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"保存比较图到: {filepath}")
        return filepath 