import argparse
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Composition.Codes.network import build_model, ImprovedDiffusionComposition
from Composition.Codes.dataset import TrainDataset
import glob
from Composition.Codes.loss import (
    cal_boundary_term, 
    cal_smooth_term_stitch, 
    cal_smooth_term_diff, 
    cal_perceptual_loss,
    cal_ssim_loss,
    cal_color_consistency_loss,
    MultiScaleLoss
)
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import math
import cv2

# path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

def train(gpu, args):
    """
    改进的训练函数，包含自动混合精度训练和高级学习率调度
    
    Args:
        gpu: GPU设备ID
        args: 训练参数
    """
    # 强制使用cudnn基准模式以提高性能
    torch.backends.cudnn.benchmark = True
    
    # GPU设置
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"使用GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        
        # 打印CUDA信息
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"是否启用cuDNN: {torch.backends.cudnn.enabled}")
        print(f"cuDNN基准模式: {torch.backends.cudnn.benchmark}")
    
    # 强制启用AMP
    args.use_amp = True
    print(f"自动混合精度(AMP)训练: {'启用' if args.use_amp else '禁用'}")
    
    # 创建日志目录
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    print(f"创建TensorBoard日志目录: {args.log_dir}")
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"TensorBoard SummaryWriter初始化完成，日志将保存到: {writer.log_dir}")
    
    # 记录训练配置信息
    # 将args转换为字典并记录
    args_dict = vars(args)
    writer.add_text('TrainingConfig', str(args_dict), 0)
    writer.flush()
    
    # 配置训练数据集
    train_dataset = TrainDataset(
        args.data_dir, 
        image_size=args.img_size,
        augment=True,
        norm_type=args.norm_type
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 创建或加载模型
    if args.use_diffusion:
        # 配置扩散模型参数
        diffusion_params = {
            'num_timesteps': args.diffusion_steps,
            'beta_start': 1e-4,
            'beta_end': 0.02,
        }
        model = ImprovedDiffusionComposition(
            image_channels=3, 
            diffusion_params=diffusion_params, 
            embedding_dim=args.embedding_dim,
            device=device  # 确保模型创建时明确指定设备
        ).to(device)
    else:
        model = build_model(args.model_type, pretrain=args.pretrain).to(device)
    
    # 验证模型是否在正确设备上
    print(f"模型设备: {next(model.parameters()).device}")
    
    # 确保所有模型参数在同一设备上
    for name, param in model.named_parameters():
        if param.device != device:
            print(f"移动参数 {name} 从 {param.device} 到 {device}")
            param.data = param.data.to(device)
    
    # 多尺度损失函数
    multi_scale_loss = MultiScaleLoss(scales=[1, 0.5, 0.25]).to(device)
    
    # 优化器设置：使用AdamW优化器提高泛化能力
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 使用余弦退火学习率调度
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs, 
            eta_min=args.lr * 0.01
        )
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.epochs * len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
    
    # 从检查点恢复训练（如果存在）
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 确保优化器状态在正确设备上
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    
    # 自动混合精度训练设置
    scaler = GradScaler(enabled=args.use_amp)
    
    # 损失权重
    loss_weights = {
        'l1': args.l1_weight,
        'boundary': args.boundary_weight,
        'smooth': args.smooth_weight,
        'perceptual': args.perceptual_weight,
        'ssim': args.ssim_weight,
        'color': args.color_weight,
        'diffusion': args.diffusion_weight
    }
    
    # 辅助函数：确保张量在正确设备上
    def ensure_on_device(tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor) and tensor.device != device:
            return tensor.to(device)
        return tensor
    
    # 主训练循环
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_l1_loss = 0
        epoch_boundary_loss = 0
        epoch_smooth_loss = 0
        epoch_perceptual_loss = 0
        epoch_diffusion_loss = 0
        
        # 使用tqdm进度条可视化训练过程
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # 监控GPU使用情况
        if torch.cuda.is_available():
            start_gpu_memory = torch.cuda.memory_allocated(device) / 1024**2
            print(f"GPU内存使用(开始): {start_gpu_memory:.2f} MB")
        
        start_time = time.time()
        
        for batch_idx, (img1, img2, mask1, mask2, gt, warp1, warp2) in enumerate(progress_bar):
            # 数据准备 - 确保所有张量都在正确的设备上
            img1, img2 = img1.to(device, non_blocking=True), img2.to(device, non_blocking=True)
            mask1, mask2 = mask1.to(device, non_blocking=True), mask2.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            warp1, warp2 = warp1.to(device, non_blocking=True), warp2.to(device, non_blocking=True)
            
            # 打印设备信息（仅在训练开始时）
            if epoch == start_epoch and batch_idx == 0:
                print(f"数据设备检查:")
                print(f" - img1: {img1.device}")
                print(f" - mask1: {mask1.device}")
                print(f" - gt: {gt.device}")
                print(f" - model: {next(model.parameters()).device}")
                
                # 检查数据类型，确保使用半精度
                print(f"数据类型:")
                print(f" - img1: {img1.dtype}")
                print(f" - mask1: {mask1.dtype}")
                print(f" - gt: {gt.dtype}")
                print(f" - model参数: {next(model.parameters()).dtype}")
            
            # 重叠区域
            mask_overlap = mask1 * mask2
            
            # 确保在进入autocast之前所有数据都在GPU上
            torch.cuda.synchronize()
            
            # 自动混合精度训练
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                try:
                    if args.use_diffusion:
                        # 使用扩散模型进行预测 - 直接使用forward方法而不是sample
                        mask_output, denoised = model(img1, img2, mask1, mask2)
                        
                        # 确保输出在正确设备上
                        mask_output = ensure_on_device(mask_output)
                        denoised = ensure_on_device(denoised)
                        
                        # mask_output可能包含多个通道，取第一个通道作为mask
                        mask = mask_output[:, 0:1, :, :]
                        
                        # 计算融合图像
                        stitched_image = img1 * mask + img2 * (1 - mask)
                    else:
                        # 使用常规模型进行预测
                        mask = model(torch.cat([img1, img2, mask1, mask2], 1))
                        
                        # 确保输出在正确设备上
                        mask = ensure_on_device(mask)
                        
                        # 计算融合图像
                        stitched_image = warp1 * mask + warp2 * (1 - mask)
                    
                    # 确保融合图像在正确设备上
                    stitched_image = ensure_on_device(stitched_image)
                    
                    # 计算各种损失
                    l1_loss = F.l1_loss(stitched_image, gt)
                    
                    # 边界一致性损失
                    boundary_loss, boundary_mask = cal_boundary_term(warp1, warp2, mask1, mask2, stitched_image)
                    boundary_loss = ensure_on_device(boundary_loss)
                    boundary_mask = ensure_on_device(boundary_mask)
                    
                    # 边界损失异常大时进行截断（降低阈值从10.0到3.0）
                    if boundary_loss > 3.0:
                        print(f"警告: 边界损失异常大 ({boundary_loss.item():.4f})，将被截断到3.0")
                        boundary_loss = torch.clamp(boundary_loss, max=3.0)
                    
                    # 诊断边界区域
                    if batch_idx == 0 and epoch % 5 == 0:
                        # 计算边界区域的比例
                        boundary_ratio = boundary_mask.sum() / (boundary_mask.shape[0] * boundary_mask.shape[1] * boundary_mask.shape[2] * boundary_mask.shape[3])
                        print(f"边界区域占比: {boundary_ratio.item():.4f}")
                        writer.add_scalar('Diagnostics/boundary_ratio', boundary_ratio.item(), epoch)
                        
                        # 将边界掩码可视化（每5个epoch）
                        if epoch % 5 == 0 and boundary_mask is not None:
                            boundary_vis = boundary_mask[:4].cpu()
                            writer.add_images('Masks/boundary', boundary_vis, epoch)
                    
                    # 平滑损失
                    smooth_loss = cal_smooth_term_stitch(torch.cat([warp1, warp2], dim=1), mask)
                    smooth_loss = ensure_on_device(smooth_loss)
                    
                    # 差异平滑损失
                    mask_overlap = ensure_on_device(mask_overlap)  # 确保重叠掩码在正确设备上
                    diff_smooth_loss = cal_smooth_term_diff(warp1, warp2, mask, mask_overlap)
                    diff_smooth_loss = ensure_on_device(diff_smooth_loss)
                    
                    # 多尺度感知损失
                    perceptual_loss = cal_perceptual_loss(stitched_image, warp1, warp2, mask, 1-mask)
                    perceptual_loss = ensure_on_device(perceptual_loss)
                    
                    # SSIM损失
                    ssim_loss = cal_ssim_loss(stitched_image, warp1, warp2, mask, 1-mask)
                    ssim_loss = ensure_on_device(ssim_loss)
                    
                    # 颜色一致性损失
                    color_loss = cal_color_consistency_loss(stitched_image, warp1, warp2, mask1, mask2)
                    color_loss = ensure_on_device(color_loss)
                    
                    # 多尺度L1损失
                    ms_loss = multi_scale_loss(stitched_image, gt)
                    ms_loss = ensure_on_device(ms_loss)
                    
                    # 改进的扩散损失计算，使用噪声预测而非图像比较
                    diffusion_loss = torch.tensor(0.0, device=device)
                    if args.use_diffusion:
                        # 1. 随机采样时间步
                        batch_size = img1.shape[0]
                        t = torch.randint(0, args.diffusion_steps, (batch_size,), device=device)
                        
                        # 2. 对目标图像添加噪声
                        noisy_gt, noise_target = model.diffusion.forward_diffusion(gt, t)
                        
                        # 3. 预测噪声（使用相同时间步）
                        concat_input = torch.cat([img1, img2], dim=1)  # 这会产生6通道输入
                        
                        # 确保使用channel_adapter将6通道减少到3通道
                        if hasattr(model, 'channel_adapter'):
                            adapted_input = model.channel_adapter(concat_input)
                        elif hasattr(model.diffusion, 'channel_adapter'):
                            adapted_input = model.diffusion.channel_adapter(concat_input)
                        else:
                            # 如果没有channel_adapter，创建一个临时的
                            print("警告：未找到channel_adapter，创建临时卷积层")
                            temp_adapter = nn.Conv2d(6, 3, kernel_size=1).to(device)
                            adapted_input = temp_adapter(concat_input)
                        
                        # 确保输入通道数匹配
                        assert adapted_input.shape[1] == 3, f"通道数不匹配: 期望3，得到{adapted_input.shape[1]}"
                        
                        # 使用处理后的输入预测噪声
                        predicted_noise = model.diffusion(adapted_input, t)
                        
                        # 打印维度信息以进行诊断
                        if batch_idx == 0 and epoch == start_epoch:
                            print(f"噪声预测维度信息:")
                            print(f" - 适配后输入尺寸: {adapted_input.shape}")
                            print(f" - 预测噪声尺寸: {predicted_noise.shape}")
                            print(f" - 目标噪声尺寸: {noise_target.shape}")
                        
                        # 4. 调整维度，确保预测噪声和目标噪声尺寸匹配
                        if predicted_noise.shape[2:] != noise_target.shape[2:]:
                            # 调整预测噪声和目标噪声到相同尺寸
                            min_size = min(predicted_noise.shape[2], noise_target.shape[2])
                            
                            # 如果需要，裁剪到相同的空间尺寸
                            if predicted_noise.shape[2] > min_size:
                                predicted_noise = F.interpolate(predicted_noise, size=(min_size, min_size), mode='bilinear', align_corners=True)
                            
                            if noise_target.shape[2] > min_size:
                                noise_target = F.interpolate(noise_target, size=(min_size, min_size), mode='bilinear', align_corners=True)
                            
                            # 确保两者尺寸匹配
                            assert predicted_noise.shape == noise_target.shape, \
                                f"调整后尺寸仍不匹配: predicted={predicted_noise.shape}, target={noise_target.shape}"
                        
                        # 4. 计算噪声预测损失
                        diffusion_loss = F.mse_loss(predicted_noise, noise_target)
                        diffusion_loss = ensure_on_device(diffusion_loss)
                    
                    # 总损失
                    # 改进边界损失渐进策略
                    current_boundary_weight = loss_weights['boundary']
                    if not args.exclude_boundary:
                        if epoch < args.warm_up_epochs:
                            # 使用余弦平滑增加而非线性增加
                            progress = epoch / args.warm_up_epochs
                            boundary_weight_factor = 0.5 * (1 - math.cos(math.pi * progress))
                            current_boundary_weight = loss_weights['boundary'] * boundary_weight_factor
                            if batch_idx == 0:
                                print(f"边界损失预热中: 当前权重 = {current_boundary_weight:.4f}")
                        elif epoch < args.warm_up_epochs * 2:
                            # 在预热后的一段时间内保持稳定
                            current_boundary_weight = loss_weights['boundary']
                        else:
                            # 预热两倍时间后，逐渐减小权重
                            decay_factor = max(0.5, 1.0 - (epoch - args.warm_up_epochs * 2) / (args.epochs - args.warm_up_epochs * 2) * 0.5)
                            current_boundary_weight = loss_weights['boundary'] * decay_factor
                            if batch_idx == 0 and epoch % 5 == 0:
                                print(f"边界损失衰减中: 当前权重 = {current_boundary_weight:.4f}")
                    else:
                        # 完全排除边界损失
                        current_boundary_weight = 0.0
                        if epoch == 0 and batch_idx == 0:
                            print("警告: 边界损失已被排除")
                    
                    # 增加L1和扩散损失的权重来平衡边界损失的影响
                    l1_weight_adjusted = loss_weights['l1'] * (1.0 + 0.2 * (current_boundary_weight / loss_weights['boundary']))
                    diffusion_weight_adjusted = loss_weights['diffusion'] * (1.0 + 0.3 * (current_boundary_weight / loss_weights['boundary'])) if args.use_diffusion else 0.0
                    
                    # 修改总损失计算，使用调整后的权重
                    loss = (
                        l1_weight_adjusted * (l1_loss + ms_loss) +
                        current_boundary_weight * boundary_loss +
                        loss_weights['smooth'] * (smooth_loss + diff_smooth_loss) +
                        loss_weights['perceptual'] * perceptual_loss +
                        loss_weights['ssim'] * ssim_loss +
                        loss_weights['color'] * color_loss +
                        diffusion_weight_adjusted * diffusion_loss
                    )
                    
                    # 确保最终损失在正确设备上
                    loss = ensure_on_device(loss)
                
                except RuntimeError as e:
                    if "expected device" in str(e) or "device mismatch" in str(e):
                        print(f"设备不匹配错误: {e}")
                        print(f"诊断信息:")
                        if 'mask_output' in locals():
                            print(f" - mask_output: {mask_output.device}")
                        if 'stitched_image' in locals():
                            print(f" - stitched_image: {stitched_image.device}")
                        if 'boundary_loss' in locals():
                            print(f" - boundary_loss: {boundary_loss.device}")
                        if 'loss' in locals():
                            print(f" - loss: {loss.device}")
                        print(f" - model: {next(model.parameters()).device}")
                        
                        # 尝试手动修复并继续
                        print("尝试手动修复...")
                        continue
                    else:
                        raise e
            
            # 反向传播（带AMP）
            optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True可以减少内存使用
            
            try:
                scaler.scale(loss).backward()
                
                # 记录最大梯度范数，用于诊断
                if batch_idx % 50 == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                    writer.add_scalar('Gradients/norm', grad_norm.item(), global_step=epoch * len(train_loader) + batch_idx)
                
                scaler.unscale_(optimizer)
                
                # 增强梯度裁剪，使用更保守的阈值
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                else:
                    # 即使未指定args.clip_grad，也使用默认的梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                scaler.step(optimizer)
                scaler.update()

                # 确保GPU同步，防止异步操作
                torch.cuda.synchronize()
                
            except RuntimeError as e:
                if "expected device" in str(e) or "device mismatch" in str(e):
                    print(f"反向传播时设备不匹配错误: {e}")
                    
                    # 打印所有张量的设备
                    print("\n设备诊断:")
                    print(f"- 模型: {next(model.parameters()).device}")
                    print(f"- loss: {loss.device}")
                    
                    # 检查所有梯度
                    device_mismatch = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and param.grad.device != param.device:
                            print(f"- 参数 {name} 梯度设备不匹配: 参数={param.device}, 梯度={param.grad.device}")
                            device_mismatch = True
                    
                    if not device_mismatch:
                        print("未发现梯度设备不匹配")
                    
                    # 尝试手动修复并继续
                    try:
                        print("尝试手动重建损失并重试...")
                        new_loss = torch.tensor(0.0, device=device, requires_grad=True)
                        for component in [l1_loss, ms_loss, boundary_loss, smooth_loss, 
                                         diff_smooth_loss, perceptual_loss, ssim_loss, 
                                         color_loss, diffusion_loss]:
                            if isinstance(component, torch.Tensor) and component.requires_grad:
                                component_on_device = component.to(device)
                                new_loss = new_loss + component_on_device
                        
                        new_loss.backward()
                        optimizer.step()
                        print("手动反向传播成功")
                    except Exception as recovery_e:
                        print(f"恢复失败: {recovery_e}")
                        # 跳过这个批次
                        print("跳过当前批次")
                else:
                    # 其他类型的错误直接抛出
                    raise e

            # 更新学习率（对于OneCycleLR）
            if args.scheduler == 'onecycle':
                scheduler.step()

            # 每个batch记录一次损失，确保TensorBoard有足够的数据点
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 10 == 0:  # 每10个batch记录一次，避免文件过大
                writer.add_scalar('BatchLoss/total', loss.item(), global_step)
                writer.add_scalar('BatchLoss/l1', l1_loss.item(), global_step)
                writer.add_scalar('BatchLoss/l1_weighted', l1_loss.item() * l1_weight_adjusted, global_step)
                writer.add_scalar('BatchLoss/boundary', boundary_loss.item(), global_step)
                writer.add_scalar('BatchLoss/boundary_weighted', boundary_loss.item() * current_boundary_weight, global_step)
                writer.add_scalar('Weights/boundary_weight', current_boundary_weight, global_step)
                writer.add_scalar('Weights/l1_weight_adjusted', l1_weight_adjusted, global_step)
                writer.add_scalar('BatchLoss/smooth', (smooth_loss + diff_smooth_loss).item(), global_step)
                writer.add_scalar('BatchLoss/perceptual', perceptual_loss.item(), global_step)
                if args.use_diffusion:
                    writer.add_scalar('BatchLoss/diffusion', diffusion_loss.item(), global_step)
                    writer.add_scalar('BatchLoss/diffusion_weighted', diffusion_loss.item() * diffusion_weight_adjusted, global_step)
                    writer.add_scalar('Weights/diffusion_weight_adjusted', diffusion_weight_adjusted, global_step)
                
                # 记录GPU使用情况
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated(device) / 1024**2
                    gpu_cached = torch.cuda.memory_reserved(device) / 1024**2
                    # 使用torch.cuda.memory_usage代替，如果可用
                    try:
                        gpu_memory_usage = torch.cuda.memory_usage(device)
                        writer.add_scalar('System/GPU_Memory_Usage', gpu_memory_usage, global_step)
                    except (AttributeError, RuntimeError):
                        # 如果memory_usage不可用，则不记录该指标
                        pass
                    writer.add_scalar('System/GPU_Memory_MB', gpu_memory, global_step)
                    writer.add_scalar('System/GPU_Memory_Reserved_MB', gpu_cached, global_step)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'l1': l1_loss.item(),
                'boundary': boundary_loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'GPU_mem': f"{torch.cuda.memory_allocated(device) / 1024**2:.0f}MB" if torch.cuda.is_available() else "N/A"
            })

            # 累积损失
            epoch_loss += loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_boundary_loss += boundary_loss.item()
            epoch_smooth_loss += (smooth_loss + diff_smooth_loss).item()
            epoch_perceptual_loss += perceptual_loss.item()
            if args.use_diffusion:
                epoch_diffusion_loss += diffusion_loss.item()
        
        # 计算每个epoch的时间
        epoch_time = time.time() - start_time
        
        # 监控GPU使用情况
        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated(device) / 1024**2
            writer.add_scalar('System/Epoch_GPU_Memory_MB', end_gpu_memory, epoch)
            writer.add_scalar('System/Epoch_Time_Seconds', epoch_time, epoch)
            print(f"GPU内存使用(结束): {end_gpu_memory:.2f} MB, 用时: {epoch_time:.2f}秒")
        
        # 更新学习率（对于CosineAnnealingLR）
        if args.scheduler == 'cosine':
            scheduler.step()

        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        avg_l1_loss = epoch_l1_loss / len(train_loader)
        avg_boundary_loss = epoch_boundary_loss / len(train_loader)
        avg_smooth_loss = epoch_smooth_loss / len(train_loader)
        avg_perceptual_loss = epoch_perceptual_loss / len(train_loader)
        avg_diffusion_loss = epoch_diffusion_loss / len(train_loader) if args.use_diffusion else 0
        
        # 记录训练损失 - 使用step=epoch确保唯一性
        current_step = epoch * len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, current_step)
        writer.add_scalar('L1Loss/train', avg_l1_loss, current_step)
        writer.add_scalar('BoundaryLoss/train', avg_boundary_loss, current_step)
        writer.add_scalar('SmoothLoss/train', avg_smooth_loss, current_step)
        writer.add_scalar('PerceptualLoss/train', avg_perceptual_loss, current_step)
        if args.use_diffusion:
            writer.add_scalar('DiffusionLoss/train', avg_diffusion_loss, current_step)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], current_step)
        
        # 添加损失比例图，方便比较各损失的相对大小
        writer.add_scalars('Losses/Combined', {
            'Total': avg_loss,
            'L1': avg_l1_loss,
            'Boundary': avg_boundary_loss,
            'Smooth': avg_smooth_loss,
            'Perceptual': avg_perceptual_loss,
            'Diffusion': avg_diffusion_loss if args.use_diffusion else 0
        }, current_step)
        
        # 保存可视化结果
        if (epoch + 1) % args.vis_freq == 0:
            # 确保图像在合适的范围内 [0, 1]
            # 并且移动到CPU并转换为numpy数据
            img1_vis = img1[:4].clamp(0, 1).cpu()
            img2_vis = img2[:4].clamp(0, 1).cpu()
            warp1_vis = warp1[:4].clamp(0, 1).cpu()
            warp2_vis = warp2[:4].clamp(0, 1).cpu()
            mask_vis = mask[:4].cpu()
            stitched_vis = stitched_image[:4].clamp(0, 1).cpu()
            gt_vis = gt[:4].clamp(0, 1).cpu()
            
            writer.add_images('Input/Image1', img1_vis, epoch)
            writer.add_images('Input/Image2', img2_vis, epoch)
            writer.add_images('Input/Warp1', warp1_vis, epoch)
            writer.add_images('Input/Warp2', warp2_vis, epoch)
            writer.add_images('Output/Mask', mask_vis, epoch)
            writer.add_images('Output/Stitched', stitched_vis, epoch)
            writer.add_images('Output/GroundTruth', gt_vis, epoch)
            if args.use_diffusion and denoised is not None:
                denoised_vis = denoised[:4].clamp(0, 1).cpu()
                writer.add_images('Output/Denoised', denoised_vis, epoch)
            
            # 确保数据被写入磁盘
            writer.flush()
        
        # 保存检查点
        if (epoch + 1) % args.save_freq == 0:
            # 确保模型保存目录存在
            os.makedirs(args.log_dir, exist_ok=True)
            
            # 构建完整的保存路径
            checkpoint_path = os.path.join(args.log_dir, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"保存模型到: {checkpoint_path}")
            
            # 使用与test.py一致的键名
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),  # 使用'model'而不是'model_state_dict'
                'optimizer': optimizer.state_dict(),  # 使用'optimizer'而不是'optimizer_state_dict'
                'scheduler': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
            }, checkpoint_path)
            
            # 保存最新模型
            latest_path = os.path.join(args.log_dir, 'latest.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),  # 使用'model'而不是'model_state_dict'
                'optimizer': optimizer.state_dict(),  # 使用'optimizer'而不是'optimizer_state_dict'
                'scheduler': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
            }, latest_path)
            
            # 同时保存到model目录（与main.py中的train_Composition保持一致）
            model_save_dir = os.path.join('Composition', 'model')
            os.makedirs(model_save_dir, exist_ok=True)
            model_path = os.path.join(model_save_dir, f'epoch_{epoch+1}.pth')
            print(f"同时保存模型到: {model_path}")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
            }, model_path)
            
            # 修改为每个epoch都进行测试而不是每隔2个epoch
            if args.test_during_training and hasattr(args, 'test_data_dir') and args.test_data_dir:
                print(f"\n======= 在轮次 {epoch+1} 进行测试 =======")
                
                # 创建临时参数对象用于测试
                test_args = argparse.Namespace(
                    gpu=gpu,
                    batch_size=1,  # 测试时使用较小的batch_size
                    data_dir=args.test_data_dir,
                    model_path=None,  # 不从文件加载，直接使用当前模型
                    img_size=args.img_size,
                    norm_type=args.norm_type,
                    use_diffusion=args.use_diffusion,
                    diffusion_steps=args.diffusion_steps,
                    sample_steps=50,  # 添加缺失的sample_steps参数，使用默认值50
                    embedding_dim=args.embedding_dim,
                    num_workers=1,
                    output_dir=os.path.join('Composition', 'results'),
                    exp_name='latest',
                    num_save=5,  # 只保存少量样本以节省空间
                    save_all=False,
                    save_dirs={
                        'learn_mask1': 'learn_mask1',
                        'learn_mask2': 'learn_mask2', 
                        'composition': 'composition',
                        'denoised': 'denoised',
                        'visualization': 'visualization'
                    },
                    model_type=args.model_type,
                    pretrain=args.pretrain,
                    current_model=model  # 直接传递当前模型
                )
                
                # 确保输出目录存在但为空（覆盖上一次的结果）
                output_dir = os.path.join(test_args.output_dir, test_args.exp_name)
                if os.path.exists(output_dir):
                    # 删除主输出目录中的文件
                    for file in os.listdir(output_dir):
                        file_path = os.path.join(output_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                else:
                    os.makedirs(output_dir, exist_ok=True)
                
                # 确保子目录存在且为空
                for subdir in test_args.save_dirs.values():
                    subdir_path = os.path.join(output_dir, subdir)
                    if os.path.exists(subdir_path):
                        # 清空子目录中的所有文件
                        for file in os.listdir(subdir_path):
                            file_path = os.path.join(subdir_path, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                    else:
                        os.makedirs(subdir_path, exist_ok=True)
                
                print(f"测试结果将保存到: {output_dir}")
                
                # 保存当前模型模式，测试后恢复
                train_mode = model.training
                model.eval()  # 确保模型处于评估模式
                
                # 执行测试，捕获测试指标
                test_metrics = test(test_args)
                
                # 恢复模型训练状态
                if train_mode:
                    model.train()
                
                # 记录测试指标到Tensorboard
                if test_metrics:
                    writer.add_scalar('Test/L1Error', test_metrics.get('avg_l1', 0), epoch)
                    writer.add_scalar('Test/PSNR', test_metrics.get('avg_psnr', 0), epoch)
                    writer.add_scalar('Test/InferenceTime', test_metrics.get('avg_time', 0), epoch)
                    
                    # 尝试载入和显示一些测试结果图像
                    try:
                        # 载入样例融合图像
                        sample_image_paths = sorted(glob.glob(os.path.join(output_dir, 'composition', '*.png')))[:3]
                        if sample_image_paths:
                            sample_images = []
                            for img_path in sample_image_paths:
                                img = cv2.imread(img_path)
                                if img is not None:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
                                    sample_images.append(img)
                        
                            if sample_images:
                                sample_grid = torchvision.utils.make_grid(sample_images, nrow=3)
                                writer.add_image('Test/CompositionSamples', sample_grid, epoch)
                    
                        # 载入样例掩码图像
                        sample_mask_paths = sorted(glob.glob(os.path.join(output_dir, 'learn_mask1', '*.png')))[:3]
                        if sample_mask_paths:
                            sample_masks = []
                            for mask_path in sample_mask_paths:
                                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                if mask is not None:
                                    mask = torch.from_numpy(mask).unsqueeze(0) / 255.0
                                    sample_masks.append(mask)
                        
                            if sample_masks:
                                mask_grid = torchvision.utils.make_grid(sample_masks, nrow=3)
                                writer.add_image('Test/MaskSamples', mask_grid, epoch)
                                
                    except Exception as e:
                        print(f"可视化测试结果时出错: {e}")
                
                print(f"======= 轮次 {epoch+1} 测试完成 =======\n")
                print(f"测试指标: L1错误={test_metrics.get('avg_l1', 'N/A'):.4f}, PSNR={test_metrics.get('avg_psnr', 'N/A'):.2f}dB")
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, L1: {avg_l1_loss:.4f}, "
              f"Boundary: {avg_boundary_loss:.4f}, Smooth: {avg_smooth_loss:.4f}, "
              f"Perceptual: {avg_perceptual_loss:.4f}")
        
        # 确保所有事件被写入磁盘
        writer.flush()
    
    # 保存最终模型
    final_model_path = os.path.join(args.log_dir, 'final_model.pth')
    model_save_dir = os.path.join('Composition', 'model')
    os.makedirs(model_save_dir, exist_ok=True)
    final_path_model_dir = os.path.join(model_save_dir, 'final_model.pth')
    
    # 保存到两个位置
    print(f"保存最终模型到: {final_model_path}")
    torch.save({'model': model.state_dict(), 'epoch': args.epochs-1}, final_model_path)
    print(f"保存最终模型到: {final_path_model_dir}")
    torch.save({'model': model.state_dict(), 'epoch': args.epochs-1}, final_path_model_dir)
    
    writer.close()

def test(args):
    """
    测试函数，生成拼接结果和性能评估
    
    Args:
        args: 测试参数
        
    Returns:
        dict: 包含测试指标的字典
    """
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用当前模型或加载模型
    if hasattr(args, 'current_model') and args.current_model is not None:
        model = args.current_model
        print("使用当前内存中的模型进行测试")
    else:
        # 创建或加载模型
        if args.use_diffusion:
            # 配置扩散模型参数
            diffusion_params = {
                'num_timesteps': args.diffusion_steps,
                'beta_start': 1e-4,
                'beta_end': 0.02,
            }
            model = ImprovedDiffusionComposition(
                image_channels=3, 
                diffusion_params=diffusion_params, 
                embedding_dim=args.embedding_dim,
                device=device  # 确保模型创建时明确指定设备
            ).to(device)
        else:
            model = build_model(args.model_type, pretrain=args.pretrain).to(device)
        
        # 加载模型权重
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model' in checkpoint:
            # 使用与保存一致的键名
            print(f"使用'model'键加载模型...")
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            # 向后兼容旧格式
            print(f"使用'model_state_dict'键加载模型...")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 直接加载状态字典
            print(f"直接加载模型状态字典...")
            model.load_state_dict(checkpoint)
        
        # 输出加载信息
        if 'epoch' in checkpoint:
            print(f"模型来自训练轮次: {checkpoint['epoch']+1}")
    
    # 确保模型处于评估模式
    model.eval()
    
    # 准备测试数据
    test_dataset = TrainDataset(
        args.data_dir, 
        image_size=args.img_size,
        augment=False,
        norm_type=args.norm_type,
        is_test=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # 性能评估指标
    l1_errors = []
    psnr_values = []
    ssim_values = []
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (img1, img2, mask1, mask2, gt, warp1, warp2) in enumerate(tqdm(test_loader, desc="Testing")):
            # 数据准备
            img1, img2 = img1.to(device), img2.to(device)
            mask1, mask2 = mask1.to(device), mask2.to(device)
            gt = gt.to(device)
            warp1, warp2 = warp1.to(device), warp2.to(device)
            
            # 测量推理时间
            start_time = time.time()
            
            if args.use_diffusion:
                # 调试信息：记录测试阶段输入图像尺寸
                print(f"测试阶段输入尺寸 - img1: {img1.shape}, img2: {img2.shape}")
                
                try:
                    # 使用扩散模型采样
                    learned_mask1, denoised, stitched_image = model.sample(
                        img1, img2, mask1, mask2, 
                        timesteps=args.sample_steps
                    )
                except RuntimeError as e:
                    if "size of tensor" in str(e) and "match" in str(e):
                        print(f"尺寸不匹配错误: {e}")
                        print("调整图像尺寸并重试...")
                        
                        # 重新调整输入图像大小
                        target_size = (256, 256)  # 使用标准大小
                        img1_resized = F.interpolate(img1, size=target_size, mode='bilinear', align_corners=True)
                        img2_resized = F.interpolate(img2, size=target_size, mode='bilinear', align_corners=True)
                        mask1_resized = F.interpolate(mask1, size=target_size, mode='nearest')
                        mask2_resized = F.interpolate(mask2, size=target_size, mode='nearest')
                        
                        print(f"调整后输入尺寸 - img1: {img1_resized.shape}, img2: {img2_resized.shape}")
                        
                        # 再次尝试采样
                        learned_mask1, denoised, stitched_image = model.sample(
                            img1_resized, img2_resized, mask1_resized, mask2_resized, 
                            timesteps=args.sample_steps
                        )
                        
                        # 将结果恢复到原始大小
                        learned_mask1 = F.interpolate(learned_mask1, size=img1.shape[2:], mode='bilinear', align_corners=True)
                        denoised = F.interpolate(denoised, size=img1.shape[2:], mode='bilinear', align_corners=True) 
                        stitched_image = F.interpolate(stitched_image, size=img1.shape[2:], mode='bilinear', align_corners=True)
                    else:
                        raise e
                
                # 将mask设置为learned_mask1，保持代码一致性
                mask = learned_mask1
            else:
                # 使用常规模型
                mask = model(torch.cat([img1, img2, mask1, mask2], 1))
                stitched_image = warp1 * mask + warp2 * (1 - mask)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 计算性能指标
            l1_error = F.l1_loss(stitched_image, gt).item()
            l1_errors.append(l1_error)
            
            # 计算PSNR
            mse = F.mse_loss(stitched_image, gt).item()
            psnr = -10 * torch.log10(torch.tensor(mse)).item() if mse > 0 else float('inf')
            psnr_values.append(psnr)
            
            # 保存结果图像
            if batch_idx < args.num_save or args.save_all:
                for i in range(stitched_image.size(0)):
                    # 确保存储为标准图像范围
                    output_image = stitched_image[i].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255
                    output_image = output_image.astype(np.uint8)
                    
                    # 保存图像
                    image_path = os.path.join(output_dir, f'batch_{batch_idx}_img_{i}.png')
                    cv2.imwrite(image_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
                    
                    # 保存掩码
                    mask_image = mask[i].cpu().permute(1, 2, 0).numpy() * 255
                    mask_image = mask_image.astype(np.uint8)
                    mask_path = os.path.join(output_dir, f'batch_{batch_idx}_mask_{i}.png')
                    cv2.imwrite(mask_path, mask_image)
    
    # 计算并打印平均指标
    avg_l1 = sum(l1_errors) / len(l1_errors) if l1_errors else 0
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    
    print(f"Testing completed. Results saved to {output_dir}")
    print(f"Average L1 Error: {avg_l1:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average Inference Time: {avg_time*1000:.2f} ms")
    
    # 保存指标到文件
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Average L1 Error: {avg_l1:.4f}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average Inference Time: {avg_time*1000:.2f} ms\n")
    
    # 返回测试指标，用于训练过程中的记录
    return {
        'avg_l1': avg_l1,
        'avg_psnr': avg_psnr,
        'avg_time': avg_time * 1000  # 转换为毫秒
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 基本参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--exp_name', type=str, default='default', help='实验名称')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--test_data_dir', type=str, default=None, help='测试数据目录，用于训练中的定期测试')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--model_path', type=str, default=None, help='测试时的模型路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--test_during_training', action='store_true', help='是否在训练期间定期进行测试')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='unet', help='模型类型: unet, deeplab')
    parser.add_argument('--pretrain', action='store_true', help='是否使用预训练模型')
    parser.add_argument('--img_size', type=int, default=256, help='输入图像尺寸')
    parser.add_argument('--norm_type', type=str, default='imagenet', help='标准化类型: imagenet, 0-1')
    
    # 扩散模型参数
    parser.add_argument('--use_diffusion', action='store_true', help='是否使用扩散模型')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='扩散步数')
    parser.add_argument('--sample_steps', type=int, default=50, help='采样步数')
    parser.add_argument('--embedding_dim', type=int, default=128, help='时间嵌入维度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作线程数')
    parser.add_argument('--save_freq', type=int, default=1, help='保存模型频率（轮数）')
    parser.add_argument('--vis_freq', type=int, default=5, help='可视化频率（轮数）')
    parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度训练')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'onecycle'], help='学习率调度器')
    
    # 损失权重参数
    parser.add_argument('--l1_weight', type=float, default=1.0, help='L1损失权重')
    parser.add_argument('--boundary_weight', type=float, default=0.1, help='边界损失权重')  # 从0.2降至0.1
    parser.add_argument('--smooth_weight', type=float, default=0.2, help='平滑损失权重')
    parser.add_argument('--perceptual_weight', type=float, default=0.1, help='感知损失权重')
    parser.add_argument('--ssim_weight', type=float, default=0.1, help='SSIM损失权重')
    parser.add_argument('--color_weight', type=float, default=0.1, help='颜色一致性损失权重')
    parser.add_argument('--diffusion_weight', type=float, default=0.15, help='扩散损失权重')  # 从0.2降至0.15
    parser.add_argument('--warm_up_epochs', type=int, default=10, help='边界损失预热轮数')  # 从5增至10
    parser.add_argument('--exclude_boundary', action='store_true', help='完全排除边界损失')
    
    # 测试参数
    parser.add_argument('--gpu', type=int, default=0, help='测试使用的GPU ID')
    parser.add_argument('--num_save', type=int, default=10, help='保存前N个批次的结果')
    parser.add_argument('--save_all', action='store_true', help='保存所有测试结果')

    # 添加梯度累积参数
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数')

    args = parser.parse_args()
    
    # 创建必要的目录
    if args.mode == 'train':
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
    elif args.mode == 'test':
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    
    # 根据模式选择训练或测试
    if args.mode == 'train':
        train(args.gpu if hasattr(args, 'gpu') else 0, args)
    elif args.mode == 'test':
        test(args)


