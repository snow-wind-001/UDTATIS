#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_composition')

# 获取项目路径
last_path = Path(__file__).parent.parent.absolute()

def save_image(img, path):
    """保存图像到指定路径，处理不同格式的输入"""
    # 确保像素值在0-255范围内
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = img * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存图像 - 修复单通道图像处理
    if len(img.shape) == 3 and img.shape[2] == 3:  # 彩色图像
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:  # 单通道图像或其他格式
        cv2.imwrite(path, img)

def visualize_results(input1, input2, stitched, mask1, mask2, save_path):
    """可视化结果，生成一个包含所有图像的网格"""
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 显示输入图像
    axes[0, 0].imshow(input1)
    axes[0, 0].set_title('Input Image 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(input2)
    axes[0, 1].set_title('Input Image 2')
    axes[0, 1].axis('off')
    
    # 显示拼接结果
    axes[0, 2].imshow(stitched)
    axes[0, 2].set_title('Stitched Image')
    axes[0, 2].axis('off')
    
    # 显示掩码
    if len(mask1.shape) == 2:
        mask1_display = mask1
    else:
        mask1_display = mask1[:,:,0]
        
    if len(mask2.shape) == 2:
        mask2_display = mask2
    else:
        mask2_display = mask2[:,:,0]
        
    axes[1, 0].imshow(mask1_display, cmap='gray')
    axes[1, 0].set_title('Mask 1')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask2_display, cmap='gray')
    axes[1, 1].set_title('Mask 2')
    axes[1, 1].axis('off')
    
    # 创建一个简单的混合展示
    if len(mask1.shape) == 2:
        blended = input1 * mask1[:,:,np.newaxis] + input2 * (1-mask1)[:,:,np.newaxis]
    else:
        blended = input1 * mask1 + input2 * (1-mask1)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    axes[1, 2].imshow(blended)
    axes[1, 2].set_title('Simple Blend')
    axes[1, 2].axis('off')
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(stitched, input1, input2):
    """计算评估指标 (PSNR 和 SSIM)"""
    # 计算PSNR
    psnr1 = psnr(input1, stitched)
    psnr2 = psnr(input2, stitched)
    
    # 计算SSIM
    ssim1 = ssim(input1, stitched, channel_axis=2)  # 更新为新版API
    ssim2 = ssim(input2, stitched, channel_axis=2)
    
    return {
        'psnr': (psnr1 + psnr2) / 2,
        'ssim': (ssim1 + ssim2) / 2
    }

def simple_composition(warp1, warp2, mask1, mask2):
    """使用简单的线性混合进行图像合成"""
    # 确保mask1和mask2是正确的形状 (H,W) 或 (H,W,1)
    if len(mask1.shape) == 2:
        mask1_3d = mask1[:,:,np.newaxis]
    else:
        mask1_3d = mask1
        
    if len(mask2.shape) == 2:
        mask2_3d = mask2[:,:,np.newaxis]
    else:
        mask2_3d = mask2
    
    # 简单线性混合
    stitched = warp1 * mask1_3d + warp2 * (1 - mask1_3d)
    stitched = np.clip(stitched, 0, 255).astype(np.uint8)
    
    return stitched

def test_single_pair(warp1_path, warp2_path, mask1_path, mask2_path, output_dir):
    """测试单对图像"""
    logger.info(f"处理图像: {os.path.basename(warp1_path)} 和 {os.path.basename(warp2_path)}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    warp1 = cv2.imread(warp1_path)
    warp1 = cv2.cvtColor(warp1, cv2.COLOR_BGR2RGB)
    
    warp2 = cv2.imread(warp2_path)
    warp2 = cv2.cvtColor(warp2, cv2.COLOR_BGR2RGB)
    
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    mask1 = mask1.astype(np.float32) / 255.0
    
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
    mask2 = mask2.astype(np.float32) / 255.0
    
    logger.info(f"图像尺寸: {warp1.shape}, 掩码尺寸: {mask1.shape}")
    
    # 执行简单合成
    stitched = simple_composition(warp1, warp2, mask1, mask2)
    
    # 保存结果
    save_image(warp1, os.path.join(output_dir, "input1.jpg"))
    save_image(warp2, os.path.join(output_dir, "input2.jpg"))
    save_image(mask1*255, os.path.join(output_dir, "mask1.jpg"))
    save_image(mask2*255, os.path.join(output_dir, "mask2.jpg"))
    save_image(stitched, os.path.join(output_dir, "composition.jpg"))
    
    # 可视化结果
    visualize_results(
        warp1, warp2, stitched, mask1, mask2,
        os.path.join(output_dir, "visualization.png")
    )
    
    # 计算评估指标
    metrics = calculate_metrics(stitched, warp1, warp2)
    
    # 输出评估指标
    logger.info(f'PSNR: {metrics["psnr"]:.2f}')
    logger.info(f'SSIM: {metrics["ssim"]:.4f}')
    
    # 将结果写入文件
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f'PSNR: {metrics["psnr"]:.2f}\n')
        f.write(f'SSIM: {metrics["ssim"]:.4f}\n')
    
    logger.info(f"结果已保存到: {output_dir}")
    return metrics

def main():
    """主函数，解析命令行参数并运行测试"""
    parser = argparse.ArgumentParser(description='简单图像融合测试脚本')
    
    # 必需参数
    parser.add_argument('--warp1', type=str, required=True, help='第一张输入图像路径')
    parser.add_argument('--warp2', type=str, required=True, help='第二张输入图像路径')
    parser.add_argument('--mask1', type=str, required=True, help='第一张图像掩码路径')
    parser.add_argument('--mask2', type=str, required=True, help='第二张图像掩码路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    try:
        # 运行测试
        start_time = time.time()
        test_single_pair(
            args.warp1, args.warp2, args.mask1, args.mask2, args.output_dir
        )
        elapsed_time = time.time() - start_time
        logger.info(f"测试完成，总耗时: {elapsed_time:.2f} 秒")
    except Exception as e:
        logger.error(f"测试过程中出错: {e}", exc_info=True)

if __name__ == "__main__":
    main() 