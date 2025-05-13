import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import cv2
from tqdm import tqdm
import logging

# 导入本地模块
from network import ImprovedDiffusionComposition, StableDiffusionConditioner
from dataset import CompositionDataset
from loss import CompositionLoss, GradientConsistencyLoss

def evaluate(model, val_loader, device, step, writer, num_samples=5, num_steps=200, save_dir=None):
    """
    评估模型，生成样本并计算FID得分
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        step: 当前训练步数
        writer: TensorBoard writer
        num_samples: 保存的样本数量
        num_steps: 扩散采样步数，增加到200
        save_dir: 保存目录
    """
    model.eval()
    
    # 创建保存目录
    if save_dir is None:
        save_dir = os.path.join('results', 'samples', f'step_{step}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 开始评估
    print(f"生成验证样本，使用{num_steps}步采样")
    
    # ... existing code ...

def train(config):
    """训练函数"""
    # ... existing code ...
    
    # 训练循环
    step = 0
    for epoch in range(config.start_epoch, config.num_epochs):
        model.train()
        
        # ... existing code ...
        
        # 验证
        if epoch % config.val_freq == 0 or epoch == config.num_epochs - 1:
            print(f"Epoch {epoch}, 运行验证...")
            # 提高采样步数，使用200步DDIM采样获得更高质量结果
            evaluate(
                model, val_loader, device, step, writer, 
                num_samples=5, 
                num_steps=200,  # 增加采样步数到200
                save_dir=os.path.join(config.output_dir, f'samples_epoch_{epoch}')
            )
            # 保存检查点
            save_checkpoint(model, optimizer, epoch, step, os.path.join(config.output_dir, f'ckpt_epoch_{epoch}.pth'))
        
        # ... existing code ...

def process_options():
    """处理命令行选项"""
    parser = argparse.ArgumentParser(description='训练图像融合扩散模型')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='improved_diffusion', help='模型类型')
    parser.add_argument('--time_steps', type=int, default=1000, help='扩散时间步数')
    parser.add_argument('--beta_schedule', type=str, default='cosine', help='beta调度类型')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率')
    parser.add_argument('--val_freq', type=int, default=5, help='验证频率(轮)')
    parser.add_argument('--save_freq', type=int, default=10, help='保存频率(轮)')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('--sample_steps', type=int, default=200, help='采样步数')  # 增加到200步
    
    # ... existing code ... 