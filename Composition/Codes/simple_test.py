#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
from network import build_model, ImprovedDiffusionComposition
from dataset import TestDataset
import os
import numpy as np
import cv2
from tqdm import tqdm
import glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import logging
import time
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('simple_test')

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
    
    # 保存图像
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img)

def visualize_results(input1, input2, stitched, denoised, mask1, mask2, save_path):
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
    
    # 显示去噪结果
    axes[1, 0].imshow(denoised)
    axes[1, 0].set_title('Denoised Image')
    axes[1, 0].axis('off')
    
    # 显示掩码
    axes[1, 1].imshow(mask1, cmap='gray')
    axes[1, 1].set_title('Learned Mask 1')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mask2, cmap='gray')
    axes[1, 2].set_title('Learned Mask 2')
    axes[1, 2].axis('off')
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(stitched, denoised, input1, input2):
    """计算评估指标 (PSNR 和 SSIM)"""
    # 计算PSNR
    psnr1 = psnr(input1, stitched)
    psnr2 = psnr(input2, stitched)
    psnr_denoised1 = psnr(input1, denoised)
    psnr_denoised2 = psnr(input2, denoised)
    
    # 计算SSIM
    ssim1 = ssim(input1, stitched, channel_axis=2)  # 更新为新版API
    ssim2 = ssim(input2, stitched, channel_axis=2)
    ssim_denoised1 = ssim(input1, denoised, channel_axis=2)
    ssim_denoised2 = ssim(input2, denoised, channel_axis=2)
    
    return {
        'psnr': (psnr1 + psnr2) / 2,
        'psnr_denoised': (psnr_denoised1 + psnr_denoised2) / 2,
        'ssim': (ssim1 + ssim2) / 2,
        'ssim_denoised': (ssim_denoised1 + ssim_denoised2) / 2
    }

def load_model(model_path, device):
    """加载模型并处理不同格式的权重文件"""
    # 扩散参数设置
    diffusion_params = {
        'num_timesteps': 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02
    }
    
    # 初始化模型
    net = ImprovedDiffusionComposition(diffusion_params=diffusion_params)
    if torch.cuda.is_available() and device != 'cpu':
        net = net.cuda()
    
    # 加载模型权重
    if os.path.isdir(model_path):
        # 查找最新的checkpoint
        ckpt_list = glob.glob(os.path.join(model_path, "*.pth"))
        if not ckpt_list:
            raise FileNotFoundError(f"目录 {model_path} 中没有找到.pth文件")
        
        ckpt_list.sort()
        checkpoint_path = ckpt_list[-1]
        logger.info(f"加载最新模型: {checkpoint_path}")
    elif os.path.isfile(model_path):
        # 直接加载指定的模型文件
        checkpoint_path = model_path
        logger.info(f"加载指定模型: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"模型路径 {model_path} 不存在")
    
    # 加载模型权重
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            net.load_state_dict(checkpoint['model'])
            logger.info(f"从检查点加载模型状态")
        else:
            net.load_state_dict(checkpoint)
            logger.info(f"加载模型权重")
        
        return net
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        raise

def run_test(args):
    """运行测试流程"""
    # 设置设备
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 设置路径
    model_path = args.pretrained_path
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    
    # 记录配置信息
    logger.info(f"测试数据路径: {args.test_path}")
    logger.info(f"批大小: {args.batch_size}")
    logger.info(f"结果保存路径: {result_path}")
    
    # 创建输出目录
    output_dirs = {
        'learn_mask1': os.path.join(result_path, 'learned_mask1'),
        'learn_mask2': os.path.join(result_path, 'learned_mask2'),
        'composition': os.path.join(result_path, 'composition'),
        'denoised': os.path.join(result_path, 'denoised'),
        'visualization': os.path.join(result_path, 'visualization')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 创建数据集和数据加载器
    try:
        test_data = TestDataset(data_path=args.test_path)
        test_loader = DataLoader(
            dataset=test_data, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False, 
            drop_last=False,
            pin_memory=True
        )
        logger.info(f"加载了 {len(test_data)} 个测试样本")
    except Exception as e:
        logger.error(f"加载测试数据集时出错: {e}")
        raise
    
    # 加载模型
    net = load_model(model_path, device)
    net.eval()
    
    # 评估指标
    total_metrics = {
        'psnr': 0,
        'psnr_denoised': 0,
        'ssim': 0,
        'ssim_denoised': 0
    }
    num_samples = 0
    
    # 开始计时
    start_time = time.time()
    logger.info("开始测试...")
    
    with torch.no_grad():
        for i, batch_value in enumerate(tqdm(test_loader, desc="处理样本")):
            try:
                # 获取输入数据
                warp1_tensor = batch_value[0].float()
                warp2_tensor = batch_value[1].float()
                mask1_tensor = batch_value[2].float()
                mask2_tensor = batch_value[3].float()
                
                if device != 'cpu':
                    warp1_tensor = warp1_tensor.to(device, non_blocking=True)
                    warp2_tensor = warp2_tensor.to(device, non_blocking=True)
                    mask1_tensor = mask1_tensor.to(device, non_blocking=True)
                    mask2_tensor = mask2_tensor.to(device, non_blocking=True)
                
                # 运行模型
                batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
                
                # 获取结果
                for batch_idx in range(warp1_tensor.size(0)):
                    # 提取单个样本的结果
                    stitched_image = ((batch_out['stitched_image'][batch_idx]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                    learned_mask1 = (batch_out['learned_mask1'][batch_idx]*255).cpu().detach().numpy().transpose(1,2,0)
                    learned_mask2 = (batch_out['learned_mask2'][batch_idx]*255).cpu().detach().numpy().transpose(1,2,0)
                    denoised = ((batch_out['denoised'][batch_idx]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                    input1 = ((warp1_tensor[batch_idx]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                    input2 = ((warp2_tensor[batch_idx]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                    
                    # 计算当前样本的索引
                    sample_idx = i * args.batch_size + batch_idx + 1
                    
                    # 保存结果
                    save_image(learned_mask1, os.path.join(output_dirs['learn_mask1'], f"{sample_idx:06d}.jpg"))
                    save_image(learned_mask2, os.path.join(output_dirs['learn_mask2'], f"{sample_idx:06d}.jpg"))
                    save_image(stitched_image, os.path.join(output_dirs['composition'], f"{sample_idx:06d}.jpg"))
                    save_image(denoised, os.path.join(output_dirs['denoised'], f"{sample_idx:06d}.jpg"))
                    
                    # 可视化结果
                    visualize_results(
                        input1, input2, stitched_image, denoised,
                        learned_mask1, learned_mask2,
                        os.path.join(output_dirs['visualization'], f"{sample_idx:06d}.png")
                    )
                    
                    # 计算评估指标
                    metrics = calculate_metrics(stitched_image, denoised, input1, input2)
                    for key in total_metrics:
                        total_metrics[key] += metrics[key]
                    num_samples += 1
                    
                    # 输出当前样本的指标
                    if (sample_idx % 10 == 0) or (sample_idx == len(test_data)):
                        logger.info(f'样本 {sample_idx}/{len(test_data)} PSNR: {metrics["psnr"]:.2f} SSIM: {metrics["ssim"]:.4f} '
                                   f'PSNR(denoised): {metrics["psnr_denoised"]:.2f} SSIM(denoised): {metrics["ssim_denoised"]:.4f}')
            
            except Exception as e:
                logger.error(f"处理批次 {i} 时出错: {e}")
                continue
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    logger.info(f"测试完成，总耗时: {elapsed_time:.2f} 秒")
    
    # 计算平均评估指标
    for key in total_metrics:
        total_metrics[key] /= max(num_samples, 1)
    
    # 输出平均评估指标
    logger.info('\n平均指标:')
    logger.info(f'PSNR: {total_metrics["psnr"]:.2f}')
    logger.info(f'SSIM: {total_metrics["ssim"]:.4f}')
    logger.info(f'PSNR(denoised): {total_metrics["psnr_denoised"]:.2f}')
    logger.info(f'SSIM(denoised): {total_metrics["ssim_denoised"]:.4f}')
    
    # 将结果写入文件
    with open(os.path.join(result_path, 'metrics.txt'), 'w') as f:
        f.write('平均指标:\n')
        f.write(f'PSNR: {total_metrics["psnr"]:.2f}\n')
        f.write(f'SSIM: {total_metrics["ssim"]:.4f}\n')
        f.write(f'PSNR(denoised): {total_metrics["psnr_denoised"]:.2f}\n')
        f.write(f'SSIM(denoised): {total_metrics["ssim_denoised"]:.4f}\n')
        f.write(f'总样本数: {num_samples}\n')
        f.write(f'总耗时: {elapsed_time:.2f} 秒\n')
    
    return total_metrics

def test_single_pair(warp1_path, warp2_path, mask1_path, mask2_path, model_path, output_dir, device='cuda:0'):
    """测试单对图像"""
    # 加载模型
    net = load_model(model_path, device)
    net.eval()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载和预处理图像
    warp1 = cv2.imread(warp1_path)
    warp1 = cv2.cvtColor(warp1, cv2.COLOR_BGR2RGB)
    warp1 = warp1.astype(np.float32) / 127.5 - 1.0
    
    warp2 = cv2.imread(warp2_path)
    warp2 = cv2.cvtColor(warp2, cv2.COLOR_BGR2RGB)
    warp2 = warp2.astype(np.float32) / 127.5 - 1.0
    
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    mask1 = mask1.astype(np.float32) / 255.0
    
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
    mask2 = mask2.astype(np.float32) / 255.0
    
    # 转换为张量并确保尺寸一致
    warp1_tensor = torch.from_numpy(warp1.transpose(2, 0, 1)).unsqueeze(0).float()
    warp2_tensor = torch.from_numpy(warp2.transpose(2, 0, 1)).unsqueeze(0).float()
    mask1_tensor = torch.from_numpy(mask1).unsqueeze(0).unsqueeze(0).float()
    mask2_tensor = torch.from_numpy(mask2).unsqueeze(0).unsqueeze(0).float()
    
    # 移动到指定设备
    warp1_tensor = warp1_tensor.to(device)
    warp2_tensor = warp2_tensor.to(device)
    mask1_tensor = mask1_tensor.to(device)
    mask2_tensor = mask2_tensor.to(device)
    
    # 运行模型
    with torch.no_grad():
        batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
    
    # 获取结果
    stitched_image = ((batch_out['stitched_image'][0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
    learned_mask1 = (batch_out['learned_mask1'][0]*255).cpu().detach().numpy().transpose(1,2,0)
    learned_mask2 = (batch_out['learned_mask2'][0]*255).cpu().detach().numpy().transpose(1,2,0)
    denoised = ((batch_out['denoised'][0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
    input1 = ((warp1_tensor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
    input2 = ((warp2_tensor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
    
    # 保存结果
    save_image(learned_mask1, os.path.join(output_dir, "learned_mask1.jpg"))
    save_image(learned_mask2, os.path.join(output_dir, "learned_mask2.jpg"))
    save_image(stitched_image, os.path.join(output_dir, "composition.jpg"))
    save_image(denoised, os.path.join(output_dir, "denoised.jpg"))
    
    # 可视化结果
    visualize_results(
        input1, input2, stitched_image, denoised,
        learned_mask1, learned_mask2,
        os.path.join(output_dir, "visualization.png")
    )
    
    # 计算评估指标
    metrics = calculate_metrics(stitched_image, denoised, input1, input2)
    
    # 输出评估指标
    print(f'PSNR: {metrics["psnr"]:.2f}')
    print(f'SSIM: {metrics["ssim"]:.4f}')
    print(f'PSNR(denoised): {metrics["psnr_denoised"]:.2f}')
    print(f'SSIM(denoised): {metrics["ssim_denoised"]:.4f}')
    
    # 将结果写入文件
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f'PSNR: {metrics["psnr"]:.2f}\n')
        f.write(f'SSIM: {metrics["ssim"]:.4f}\n')
        f.write(f'PSNR(denoised): {metrics["psnr_denoised"]:.2f}\n')
        f.write(f'SSIM(denoised): {metrics["ssim_denoised"]:.4f}\n')
    
    return metrics

def main():
    """主函数，解析命令行参数并运行测试"""
    parser = argparse.ArgumentParser(description='图像融合测试脚本')
    
    # 基本参数
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 表示使用CPU)')
    parser.add_argument('--batch_size', type=int, default=1, help='批大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作线程数')
    parser.add_argument('--test_path', type=str, default=None, help='测试数据路径')
    parser.add_argument('--result_path', type=str, default=None, help='结果保存路径')
    parser.add_argument('--pretrained_path', type=str, default=None, help='预训练模型路径')
    
    # 单图像对测试参数
    parser.add_argument('--single_test', action='store_true', help='测试单对图像')
    parser.add_argument('--warp1', type=str, default=None, help='第一张输入图像路径')
    parser.add_argument('--warp2', type=str, default=None, help='第二张输入图像路径')
    parser.add_argument('--mask1', type=str, default=None, help='第一张图像掩码路径')
    parser.add_argument('--mask2', type=str, default=None, help='第二张图像掩码路径')
    parser.add_argument('--output_dir', type=str, default=None, help='单图像对测试的输出目录')
    
    args = parser.parse_args()
    
    # 设置默认值
    if args.pretrained_path is None:
        args.pretrained_path = os.path.join(last_path, 'model')
    
    if args.result_path is None:
        args.result_path = os.path.join(last_path, 'results')
    
    # 根据模式选择运行方式
    if args.single_test:
        # 检查所需参数
        required_args = ['warp1', 'warp2', 'mask1', 'mask2', 'output_dir']
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        
        if missing_args:
            parser.error(f"单图像对测试模式需要以下参数: {', '.join(missing_args)}")
        
        # 运行单图像对测试
        device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu"
        test_single_pair(
            args.warp1, args.warp2, args.mask1, args.mask2,
            args.pretrained_path, args.output_dir, device
        )
    else:
        # 检查所需参数
        if args.test_path is None:
            parser.error("测试数据路径 (--test_path) 必须提供")
        
        # 运行批处理测试
        run_test(args)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"测试过程中出错: {e}", exc_info=True) 