#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
import glob
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import cv2
import time
import math

# 添加项目根目录到路径
sys.path.append('.')

def tensor_to_image(tensor):
    """
    将PyTorch张量转换为NumPy图像
    
    参数:
        tensor: PyTorch张量 [C,H,W] 或 [B,C,H,W]
        
    返回:
        numpy图像 [H,W,C], 值范围为0-255的uint8
    """
    if tensor.dim() == 4:  # [B,C,H,W]
        tensor = tensor[0]  # 只取第一个样本
    
    # 移到CPU
    tensor = tensor.detach().cpu()
    
    # 如果是单通道图像，复制到3通道
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    
    # 转为HWC排列并转为NumPy
    img = tensor.permute(1, 2, 0).numpy()
    
    # 规范化到0-1范围
    if img.min() < 0 or img.max() > 1:
        img = (img + 1.0) / 2.0  # 假设范围是[-1,1]
    
    img = np.clip(img, 0, 1)
    
    # 转为0-255范围
    img = (img * 255).astype(np.uint8)
    
    return img

def find_overlap_region(mask1, mask2, threshold=100):
    """
    找到两个掩码的重叠区域，重点关注黑色边缘部分
    
    参数:
        mask1, mask2: 掩码图像
        threshold: 像素阈值，用于确定重叠区域
    
    返回:
        overlap_bbox: 重叠区域的边界框 (x1, y1, x2, y2)
        overlap_width: 重叠区域的宽度
    """
    # 确保掩码是NumPy数组
    if isinstance(mask1, Image.Image):
        mask1 = np.array(mask1)
    if isinstance(mask2, Image.Image):
        mask2 = np.array(mask2)
    
    # 确保掩码是二值图像，黑色区域为1（表示重叠），白色区域为0（表示有效内容）
    mask1_bin = (mask1 < 50).astype(np.uint8)  # 黑色区域为1
    mask2_bin = (mask2 < 50).astype(np.uint8)  # 黑色区域为1
    
    # 检查形状并处理不匹配的情况
    if mask1_bin.shape != mask2_bin.shape:
        print(f"  掩码形状不匹配: mask1={mask1_bin.shape}, mask2={mask2_bin.shape}")
        
        # 将两个掩码调整为相同大小以便进行重叠计算
        # 使用较小的尺寸进行操作
        target_size = (min(mask1_bin.shape[0], mask2_bin.shape[0]), 
                      min(mask1_bin.shape[1], mask2_bin.shape[1]))
        
        # 裁剪掩码以匹配尺寸
        mask1_bin = mask1_bin[:target_size[0], :target_size[1]]
        mask2_bin = mask2_bin[:target_size[0], :target_size[1]]
        
        print(f"  调整后的掩码形状: {mask1_bin.shape}")
    
    # 计算重叠区域 - 两个掩码都是黑色的区域
    overlap = mask1_bin * mask2_bin
    
    # 找到重叠区域的边界
    y_indices, x_indices = np.where(overlap > 0)
    
    if len(y_indices) < threshold:
        # 重叠区域太小或不存在，尝试查找mask1的黑色边缘区域
        print("  黑色重叠区域太小，尝试查找mask1的黑色边缘区域")
        
        # 尝试查找mask1中黑色区域的边缘
        if np.sum(mask1_bin) > 0:
            # 使用膨胀和腐蚀操作找到边缘
            kernel = np.ones((5,5), np.uint8)
            dilated = cv2.dilate(mask1_bin, kernel, iterations=1)
            eroded = cv2.erode(mask1_bin, kernel, iterations=1)
            edge = dilated - eroded
            
            y_indices, x_indices = np.where(edge > 0)
            if len(y_indices) >= threshold:
                print("  使用mask1的黑色边缘区域")
            else:
                # 还是找不到足够的边缘像素，尝试直接使用mask1的黑色区域
                y_indices, x_indices = np.where(mask1_bin > 0)
                if len(y_indices) < threshold:
                    print("  无法找到足够的黑色边缘像素，尝试不同方法")
                    return None, 0
        else:
            print("  没有黑色区域，使用整个图像")
        return None, 0
    
    # 计算边界框
    x1, y1 = np.min(x_indices), np.min(y_indices)
    x2, y2 = np.max(x_indices), np.max(y_indices)
    
    # 计算重叠宽度
    overlap_width = x2 - x1 + 1
    
    # 扩展边界框，确保包含足够的黑色边缘部分
    border_expand = min(100, overlap_width // 2)  # 边界扩展量增加到100
    x1 = max(0, x1 - border_expand)
    y1 = max(0, y1 - border_expand)
    x2 = min(mask1_bin.shape[1] - 1, x2 + border_expand)
    y2 = min(mask1_bin.shape[0] - 1, y2 + border_expand)
    
    print(f"  找到黑色边缘区域: ({x1},{y1})-({x2},{y2})，宽度={overlap_width}像素")
    
    return (x1, y1, x2, y2), overlap_width

# 确保裁剪区域不会超出图像边界
def safe_crop(img, bbox):
    w, h = img.size
    x1, y1, x2, y2 = bbox
    
    # 确保在图像边界内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    return img.crop((x1, y1, x2, y2))

def crop_around_overlap(img1, img2, mask1, mask2, target_size=(512, 512)):
    """
    围绕重叠区域进行智能裁剪，重点关注黑色边缘区域
    
    参数:
        img1, img2: 输入图像
        mask1, mask2: 输入掩码
        target_size: 目标尺寸 (宽, 高)
    
    返回:
        裁剪后的图像和掩码以及裁剪信息
    """
    # 检查图像尺寸
    if img1.size != img2.size or img1.size != mask1.size or img2.size != mask2.size:
        print(f"  图像尺寸不一致: img1={img1.size}, img2={img2.size}, mask1={mask1.size}, mask2={mask2.size}")
        
        # 先将所有图像和掩码调整为相同尺寸以方便处理
        # 使用第一张图像的尺寸作为基准
        base_size = img1.size
        
        if img2.size != base_size:
            print(f"  调整 img2 尺寸从 {img2.size} 到 {base_size}")
            img2 = img2.resize(base_size, Image.LANCZOS)
        
        if mask1.size != base_size:
            print(f"  调整 mask1 尺寸从 {mask1.size} 到 {base_size}")
            mask1 = mask1.resize(base_size, Image.LANCZOS)
            
        if mask2.size != base_size:
            print(f"  调整 mask2 尺寸从 {mask2.size} 到 {base_size}")
            mask2 = mask2.resize(base_size, Image.LANCZOS)
    
    # 保存原始尺寸信息
    orig_size1 = img1.size
    orig_size2 = img2.size
    
    # 查找重叠区域 - 特别关注黑色边缘
    overlap_bbox, overlap_width = find_overlap_region(mask1, mask2)
    
    # 创建裁剪信息字典，用于后续恢复原始尺寸
    crop_info = {
        'orig_size1': orig_size1,
        'orig_size2': orig_size2
    }
    
    # 如果找不到重叠区域，尝试替代策略
    if overlap_bbox is None:
        print("  未找到黑色重叠区域，尝试沿着mask1的黑色边缘进行裁剪")
        
        # 转换mask1为NumPy数组
        mask1_array = np.array(mask1)
        mask1_bin = (mask1_array < 50).astype(np.uint8)  # 黑色区域为1
        
        # 检查是否存在黑色区域
        if np.sum(mask1_bin) > 0:
            # 查找黑色区域边缘
            kernel = np.ones((5,5), np.uint8)
            dilated = cv2.dilate(mask1_bin, kernel, iterations=1)
            eroded = cv2.erode(mask1_bin, kernel, iterations=1)
            edge = dilated - eroded
            
            y_indices, x_indices = np.where(edge > 0)
            
            if len(y_indices) > 0:
                # 使用黑色区域边缘进行裁剪
                x1, y1 = np.min(x_indices), np.min(y_indices)
                x2, y2 = np.max(x_indices), np.max(y_indices)
                
                # 扩展边界确保包含整个边缘
                border_expand = 100  # 增加边界扩展量
                x1 = max(0, x1 - border_expand)
                y1 = max(0, y1 - border_expand)
                x2 = min(mask1_array.shape[1] - 1, x2 + border_expand)
                y2 = min(mask1_array.shape[0] - 1, y2 + border_expand)
                
                print(f"  使用mask1的黑色边缘区域: ({x1},{y1})-({x2},{y2})")
                
                # 保存边界信息
                overlap_bbox = (x1, y1, x2, y2)
                overlap_width = x2 - x1 + 1
                crop_info['overlap_bbox'] = overlap_bbox
                
                # 裁剪
                img1_crop = safe_crop(img1, overlap_bbox)
                mask1_crop = safe_crop(mask1, overlap_bbox)
                img2_crop = safe_crop(img2, overlap_bbox)
                mask2_crop = safe_crop(mask2, overlap_bbox)
                
                # 调整大小到目标尺寸
                img1_crop = img1_crop.resize(target_size, Image.LANCZOS)
                mask1_crop = mask1_crop.resize(target_size, Image.LANCZOS)
                img2_crop = img2_crop.resize(target_size, Image.LANCZOS)
                mask2_crop = mask2_crop.resize(target_size, Image.LANCZOS)
                
                # 设置裁剪区域的坐标
                crop_info['x1'] = x1
                crop_info['y1'] = y1
                crop_info['x2'] = x2
                crop_info['y2'] = y2
                crop_info['target_size'] = target_size
                
                # 计算裁剪区域，以包含整个重叠区域
                crop_width = target_size[0]
                crop_height = target_size[1]
                
                # 检查重叠区域的宽高比
                overlap_height = y2 - y1 + 1
                aspect_ratio = overlap_width / overlap_height
                target_aspect_ratio = crop_width / crop_height
                
                print(f"  原始裁剪区域: 宽度={overlap_width}, 高度={overlap_height}, 宽高比={aspect_ratio:.2f}")
                print(f"  目标裁剪区域: 宽度={crop_width}, 高度={crop_height}, 宽高比={target_aspect_ratio:.2f}")
                
                return img1_crop, img2_crop, mask1_crop, mask2_crop, crop_info
            else:
                print("  无法找到mask1的黑色边缘区域，使用整个图像")
    else:
        print("  mask1没有黑色区域，使用整个图像")
    
    # 正常裁剪流程 - 有找到重叠区域
        x1, y1, x2, y2 = overlap_bbox
        
    # 保存重叠边界信息
    crop_info['overlap_bbox'] = overlap_bbox
    crop_info['x1'] = x1
    crop_info['y1'] = y1
    crop_info['x2'] = x2
    crop_info['y2'] = y2
    crop_info['target_size'] = target_size
    
    # 计算裁剪区域，以包含整个重叠区域
    crop_width = target_size[0]
    crop_height = target_size[1]
        
    # 检查重叠区域的宽高比
    overlap_height = y2 - y1 + 1
    aspect_ratio = overlap_width / overlap_height
    target_aspect_ratio = crop_width / crop_height
    
    print(f"  原始裁剪区域: 宽度={overlap_width}, 高度={overlap_height}, 宽高比={aspect_ratio:.2f}")
    print(f"  目标裁剪区域: 宽度={crop_width}, 高度={crop_height}, 宽高比={target_aspect_ratio:.2f}")
    
    # 根据目标宽高比调整裁剪区域
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # 尝试最大化包含黑色边缘区域，同时维持目标宽高比
    if aspect_ratio > target_aspect_ratio:
        # 太宽，需要增加高度
        new_height = int(overlap_width / target_aspect_ratio)
        half_height = new_height // 2
        y1_new = max(0, center_y - half_height)
        y2_new = min(orig_size1[1], center_y + half_height)
        x1_new, x2_new = x1, x2
    else:
        # 太高，需要增加宽度
        new_width = int(overlap_height * target_aspect_ratio)
        half_width = new_width // 2
        x1_new = max(0, center_x - half_width)
        x2_new = min(orig_size1[0], center_x + half_width)
        y1_new, y2_new = y1, y2
    
    # 更新裁剪区域信息
    crop_info['x1'] = x1_new
    crop_info['y1'] = y1_new
    crop_info['x2'] = x2_new
    crop_info['y2'] = y2_new
    
    # 裁剪图像
    img1_crop = safe_crop(img1, (x1_new, y1_new, x2_new, y2_new))
    mask1_crop = safe_crop(mask1, (x1_new, y1_new, x2_new, y2_new))
    img2_crop = safe_crop(img2, (x1_new, y1_new, x2_new, y2_new))
    mask2_crop = safe_crop(mask2, (x1_new, y1_new, x2_new, y2_new))
    
    # 调整大小到目标尺寸
    img1_crop = img1_crop.resize(target_size, Image.LANCZOS)
    mask1_crop = mask1_crop.resize(target_size, Image.LANCZOS)
    img2_crop = img2_crop.resize(target_size, Image.LANCZOS)
    mask2_crop = mask2_crop.resize(target_size, Image.LANCZOS)
    
    print(f"  最终裁剪区域: ({x1_new},{y1_new})-({x2_new},{y2_new})")
    
    return img1_crop, img2_crop, mask1_crop, mask2_crop, crop_info

def main():
    parser = argparse.ArgumentParser(description='增强版Composition测试脚本')
    parser.add_argument('--model_path', type=str, default='Composition/model/model_latest.pth', 
                        help='模型路径')
    parser.add_argument('--test_data', type=str, default='data/UDIS-D/composition_data/test',
                        help='测试数据路径')
    parser.add_argument('--output_dir', type=str, default='test_composition_results',
                        help='输出结果目录')
    parser.add_argument('--limit', type=int, default=10,
                        help='测试图像数量限制，设为-1表示处理全部图像')
    parser.add_argument('--gpu', type=int, default=0,
                        help='指定使用的GPU ID')
    parser.add_argument('--sample_steps', type=int, default=100, help='扩散采样步数')
    parser.add_argument('--image_size', type=int, default=512,
                        help='输入图像的处理分辨率')
    parser.add_argument('--overlap_based_stitching', action='store_true',
                        help='启用基于重叠区域的智能裁剪与拼接')
    parser.add_argument('--restore_original_size', action='store_true',
                        help='将结果恢复到原始图像尺寸')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='生成可视化结果')
    
    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available():
        if args.gpu >= 0 and args.gpu < torch.cuda.device_count():
            device = torch.device(f'cuda:{args.gpu}')
            print(f"使用GPU: {torch.cuda.get_device_name(args.gpu)}")
        else:
            print(f"指定的GPU ID {args.gpu} 无效，使用默认GPU")
            device = torch.device('cuda')
    else:
        print("CUDA不可用，使用CPU")
        device = torch.device('cpu')
    
    # 创建输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子目录
    result_dir = os.path.join(output_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    panorama_dir = os.path.join(output_dir, "panorama")
    os.makedirs(panorama_dir, exist_ok=True)
    
    if args.visualize:
        vis_dir = os.path.join(output_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
    
    # 保存测试参数
    with open(os.path.join(output_dir, "test_config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    try:
        from Composition.Codes.enhanced_network import ImprovedDiffusionComposition
        
        # 创建模型实例
        model = ImprovedDiffusionComposition(
            num_timesteps=1000,
            beta_schedule='linear',
            image_size=args.image_size,
            base_channels=64,
            attention_resolutions=[16, 8],
            dropout=0.0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            num_res_blocks=2,
            heads=4,
            use_scale_shift_norm=True
        ).to(device)
        
        # 加载模型权重
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                # 检查点是训练期间保存的整个状态
                model.load_state_dict(checkpoint['model'])
            else:
                # 尝试其他可能的键
                possible_keys = ['state_dict', 'net', 'network', 'model_weights']
                for key in possible_keys:
                    if key in checkpoint:
                        model.load_state_dict(checkpoint[key])
                        break
                else:
                    # 如果没有找到任何已知键，尝试直接加载
                    try:
                        model.load_state_dict(checkpoint)
                    except Exception as load_err:
                        print(f"无法加载模型权重: {load_err}")
                        raise
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print(f"模型加载成功，设置为评估模式")
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 查找测试图像
    warp1_dir = os.path.join(args.test_data, "warp1")
    warp2_dir = os.path.join(args.test_data, "warp2")
    mask1_dir = os.path.join(args.test_data, "mask1")
    mask2_dir = os.path.join(args.test_data, "mask2")
    
    warp1_files = sorted(glob.glob(os.path.join(warp1_dir, "*.*")))
    warp2_files = sorted(glob.glob(os.path.join(warp2_dir, "*.*")))
    mask1_files = sorted(glob.glob(os.path.join(mask1_dir, "*.*")))
    mask2_files = sorted(glob.glob(os.path.join(mask2_dir, "*.*")))
    
    # 确保文件列表非空且长度一致
    if not warp1_files or not warp2_files or not mask1_files or not mask2_files:
        print(f"无法找到测试图像，请检查路径: {args.test_data}")
        return
    
    if len(warp1_files) != len(warp2_files) or len(warp1_files) != len(mask1_files) or len(warp1_files) != len(mask2_files):
        print(f"警告: 文件数量不一致 - warp1: {len(warp1_files)}, warp2: {len(warp2_files)}, mask1: {len(mask1_files)}, mask2: {len(mask2_files)}")
        min_len = min(len(warp1_files), len(warp2_files), len(mask1_files), len(mask2_files))
        warp1_files = warp1_files[:min_len]
        warp2_files = warp2_files[:min_len]
        mask1_files = mask1_files[:min_len]
        mask2_files = mask2_files[:min_len]
    
    # 限制测试图像数量
    if args.limit > 0 and args.limit < len(warp1_files):
        warp1_files = warp1_files[:args.limit]
        warp2_files = warp2_files[:args.limit]
        mask1_files = mask1_files[:args.limit]
        mask2_files = mask2_files[:args.limit]
    
    # 预处理变换
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    mask_preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    
    print(f"开始处理 {len(warp1_files)} 对图像...")
    
    # 记录处理时间和评估指标
    total_time = 0
    success_count = 0
    results = []
    
    # 处理每对图像
    for i, (warp1_path, warp2_path, mask1_path, mask2_path) in enumerate(tqdm(zip(warp1_files, warp2_files, mask1_files, mask2_files), total=len(warp1_files))):
        try:
            file_basename = os.path.splitext(os.path.basename(warp1_path))[0]
            print(f"\n处理图像对 {i+1}/{len(warp1_files)}: {file_basename}")
            
            # 加载图像
            warp1 = Image.open(warp1_path).convert('RGB')
            warp2 = Image.open(warp2_path).convert('RGB')
            mask1 = Image.open(mask1_path).convert('L')
            mask2 = Image.open(mask2_path).convert('L')
            
            # 记录原始尺寸
            original_size = warp1.size
            print(f"  原始图像尺寸: {original_size}")
            
            # 预处理图像
            if args.overlap_based_stitching:
                print("  使用基于重叠区域的智能裁剪")
                crop_info = {}
                # 检测重叠区域
                overlap_bbox, overlap_width = find_overlap_region(mask1, mask2)
                if overlap_bbox:
                    x1, y1, x2, y2 = overlap_bbox
                    print(f"  找到重叠区域: 位置=({x1},{y1})-({x2},{y2}), 宽度={overlap_width}px")
                    crop_info['overlap_bbox'] = overlap_bbox
                
                # 智能裁剪
                warp1_proc, warp2_proc, mask1_proc, mask2_proc, crop_info = crop_around_overlap(
                    warp1, warp2, mask1, mask2, target_size=(args.image_size, args.image_size)
                )
                
                # 保存裁剪信息
                crop_info['original_size'] = original_size
                crop_info['target_size'] = (args.image_size, args.image_size)
                crop_info['orig_size1'] = warp1.size  # 添加原始尺寸信息
                crop_info['orig_size2'] = warp2.size  # 添加原始尺寸信息
            else:
                print("  使用直接缩放")
                warp1_proc = warp1.resize((args.image_size, args.image_size), Image.LANCZOS)
                warp2_proc = warp2.resize((args.image_size, args.image_size), Image.LANCZOS)
                mask1_proc = mask1.resize((args.image_size, args.image_size), Image.LANCZOS)
                mask2_proc = mask2.resize((args.image_size, args.image_size), Image.LANCZOS)
                crop_info = None
            
            # 转换为张量
            warp1_tensor = preprocess(warp1_proc).unsqueeze(0).to(device)
            warp2_tensor = preprocess(warp2_proc).unsqueeze(0).to(device)
            mask1_tensor = mask_preprocess(mask1_proc).unsqueeze(0).to(device)
            mask2_tensor = mask_preprocess(mask2_proc).unsqueeze(0).to(device)
            
            # 使用模型进行处理
            if args.overlap_based_stitching and crop_info:
                # 使用safe_forward_with_original_size方法处理裁剪后的图像
                print("  使用safe_forward_with_original_size方法（恢复到原始尺寸）")
                input_dict = {
                    'crop_info': crop_info,
                    'original_paths': {
                        'warp1': warp1_path,
                        'warp2': warp2_path,
                        'mask1': mask1_path,
                        'mask2': mask2_path
                    }
                }
                
                start_time = time.time()
                print(f"使用 {args.sample_steps} 步扩散采样生成warp2模板")
                
                # 使用带有原始尺寸恢复的前向传播
                mask, result, panorama = model.safe_forward_with_original_size(
                    warp1=warp1_tensor,
                    warp2=warp2_tensor,
                    mask1=mask1_tensor,
                    mask2=mask2_tensor,
                    input_dict=input_dict,
                    restore_original=args.restore_original_size
                )
                
                if mask is None or result is None:
                    print(f"处理图像对 {file_basename} 时出错: 模型返回了空结果")
                    failed_count += 1
                    continue
                    
                # 保存生成的全景图
                if panorama is not None:
                    # 确保存储目录存在
                    os.makedirs(panorama_dir, exist_ok=True)
                    
                    # 保存增强版全景图
                    panorama_img = tensor_to_image(panorama)
                    panorama_path = os.path.join(panorama_dir, f"{file_basename}_panorama_enhanced.png")
                    cv2.imwrite(panorama_path, panorama_img)
                    
                    # 创建一个基本的全景图版本（简单拼接warp1和warp2）
                    # 这是为了对比增强模型与简单拼接的差异
                    try:
                        # 加载原始图像
                        warp1_img = cv2.imread(warp1_path)
                        warp2_img = cv2.imread(warp2_path)
                        
                        if warp1_img is not None and warp2_img is not None:
                            # 确保两个图像高度相同
                            h1, w1 = warp1_img.shape[:2]
                            h2, w2 = warp2_img.shape[:2]
                            
                            if h1 != h2:
                                # 调整高度
                                target_height = max(h1, h2)
                                if h1 < target_height:
                                    warp1_img = cv2.resize(warp1_img, (int(w1 * target_height / h1), target_height))
                                if h2 < target_height:
                                    warp2_img = cv2.resize(warp2_img, (int(w2 * target_height / h2), target_height))
                            
                            # 获取裁剪信息中的重叠区域
                            overlap_bbox = crop_info.get('overlap_bbox')
                            if overlap_bbox:
                                x1, y1, x2, y2 = overlap_bbox
                                overlap_width = x2 - x1
                            else:
                                # 默认使用10%的重叠
                                overlap_width = int(min(w1, w2) * 0.1)
                            
                            # 创建基本全景图
                            basic_panorama_width = w1 + w2 - overlap_width
                            
                            # 确保宽度计算不会出错
                            if w1 + w2 - overlap_width > max(w1, w2):
                                # 创建画布
                                basic_panorama = np.zeros((max(warp1_img.shape[0], warp2_img.shape[0]), 
                                                         basic_panorama_width, 3), dtype=np.uint8)
                                
                                # 放置warp1
                                if warp1_img.shape[1] <= basic_panorama_width:
                                    basic_panorama[:warp1_img.shape[0], :warp1_img.shape[1]] = warp1_img
                                else:
                                    # 如果太宽，裁剪
                                    basic_panorama[:warp1_img.shape[0], :basic_panorama_width] = warp1_img[:, :basic_panorama_width]
                                
                                # 创建渐变过渡区域
                                for x in range(overlap_width):
                                    alpha = x / overlap_width  # 从0到1的过渡
                                    pos = w1 - overlap_width + x
                                    if 0 <= pos < w1 and 0 <= x < w2 and pos < basic_panorama_width:
                                        # 确保在有效区域内
                                        warp1_slice = warp1_img[:warp1_img.shape[0], pos].reshape(warp1_img.shape[0], 1, 3)
                                        warp2_slice = warp2_img[:warp2_img.shape[0], x].reshape(warp2_img.shape[0], 1, 3)
                                        
                                        # 处理高度不同的情况
                                        common_height = min(warp1_img.shape[0], warp2_img.shape[0])
                                        basic_panorama[:common_height, pos] = (
                                            warp1_slice[:common_height, 0] * (1 - alpha) + 
                                            warp2_slice[:common_height, 0] * alpha
                                        )
                                
                                # 放置warp2其余部分
                                warp2_start = w1 - overlap_width + overlap_width  # w1
                                if warp2_start < basic_panorama_width:
                                    # 计算要复制的warp2宽度
                                    copy_width = min(w2 - overlap_width, basic_panorama_width - warp2_start)
                                    # 确保不会越界
                                    if copy_width > 0:
                                        basic_panorama[:warp2_img.shape[0], warp2_start:warp2_start+copy_width] = (
                                            warp2_img[:, overlap_width:overlap_width+copy_width]
                                        )
                                
                            # 保存基本全景图
                            basic_panorama_path = os.path.join(panorama_dir, f"{file_basename}_panorama_basic.png")
                            cv2.imwrite(basic_panorama_path, basic_panorama)
                    except Exception as e:
                        print(f"  创建基本全景图时出错: {e}")
                else:
                    # 使用常规前向传播
                    start_time = time.time()
                    mask, result = model.safe_forward(
                        warp1=warp1_tensor,
                        warp2=warp2_tensor,
                        mask1=mask1_tensor,
                        mask2=mask2_tensor
                    )
            
            # 计算处理时间
            process_time = time.time() - start_time
            total_time += process_time
            
            # 转换结果为图像
            mask_img = tensor_to_image(mask)
            result_img = tensor_to_image(result)
            
            # 保存结果
            print(f"  保存结果: {file_basename}, 处理时间: {process_time:.2f}秒")
            cv2.imwrite(os.path.join(result_dir, f"{file_basename}.png"), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(result_dir, f"{file_basename}_mask.png"), mask_img)
            
            # 生成可视化结果
            if args.visualize:
                try:
                    # 获取图像尺寸
                    warp1_img = tensor_to_image(warp1_tensor)
                    warp2_img = tensor_to_image(warp2_tensor)
                    
                    # 检查结果图像尺寸是否需要调整
                    if result_img.shape[:2] != warp1_img.shape[:2]:
                        print(f"  调整结果图像从 {result_img.shape[:2]} 到 {warp1_img.shape[:2]}")
                        result_img = cv2.resize(result_img, (warp1_img.shape[1], warp1_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    
                    # 创建可视化展示
                    img_height, img_width = warp1_img.shape[:2]
                    vis_width = img_width * 3
                    vis_height = img_height
                    
                    visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
                    
                    # 放置warp1、warp2和result
                    visualization[:, :img_width] = warp1_img
                    
                    # 确保warp2尺寸与warp1相同
                    if warp2_img.shape[:2] != warp1_img.shape[:2]:
                        warp2_img = cv2.resize(warp2_img, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
                    
                    visualization[:, img_width:img_width*2] = warp2_img
                    visualization[:, img_width*2:] = result_img
                    
                    # 添加文字标签
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(visualization, "Warp 1", (10, 30), font, 1, (255, 255, 255), 2)
                    cv2.putText(visualization, "Warp 2", (img_width + 10, 30), font, 1, (255, 255, 255), 2)
                    cv2.putText(visualization, "Result", (img_width*2 + 10, 30), font, 1, (255, 255, 255), 2)
                    
                    # 保存可视化
                    cv2.imwrite(os.path.join(vis_dir, f"{file_basename}_vis.png"), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                except Exception as vis_e:
                    print(f"  创建可视化结果时出错: {vis_e}")
                    import traceback
                    traceback.print_exc()
            
            # 更新处理信息
            success_count += 1
            results.append({
                'filename': file_basename,
                'process_time': process_time,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"处理图像对 {file_basename} 时出错: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'filename': file_basename,
                'status': 'error',
                'error': str(e)
            })
    
    # 打印统计信息
    avg_time = total_time / max(success_count, 1)
    print(f"\n处理完成!")
    print(f"成功处理: {success_count}/{len(warp1_files)} 图像对")
    print(f"平均处理时间: {avg_time:.2f}秒")
    print(f"结果保存在: {output_dir}")
    
    # 保存处理报告
    with open(os.path.join(output_dir, "processing_report.json"), "w") as f:
        json.dump({
            'total_images': len(warp1_files),
            'success_count': success_count,
            'average_time': avg_time,
            'results': results
        }, f, indent=4)

if __name__ == "__main__":
    main() 