import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Warp.Codes.network import build_model, Network
from Warp.Codes.dataset import TrainDataset, UDISDataset
import glob
from Warp.Codes.loss import cal_lp_loss, inter_grid_loss, intra_grid_loss
from Warp.Codes.improved_network import ImprovedWarpNetwork
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random


# path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

def train(args):
    """
    单GPU训练函数，为保持向后兼容
    """
    return train_distributed(0, 1, args)

def train_distributed(rank, world_size, args, device=None):
    """
    分布式训练函数，支持单GPU和多GPU训练
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        args: 训练参数
        device: 训练设备
    """
    # 设置随机种子以确保可复现性
    seed = 42 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 判断是否为分布式训练
    is_distributed = world_size > 1 and hasattr(args, 'distributed') and args.distributed
    
    # 设置训练设备
    if device is None:
        if is_distributed:
            device = torch.device(f'cuda:{rank}')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置主进程标志
    is_main_process = (rank == 0)
    
    # 只有主进程打印详细信息
    if is_main_process:
        print(f"使用设备: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    
    # 设置参数
    batch_size = args.batch_size
    num_epochs = args.max_epoch
    learning_rate = args.learning_rate if hasattr(args, 'learning_rate') else 1e-4
    
    # 设置路径
    model_save_path = args.model_save_path if hasattr(args, 'model_save_path') else os.path.join(last_path, 'model')
    summary_path = args.summary_path if hasattr(args, 'summary_path') else os.path.join(last_path, 'summary')
    
    # 创建目录（只有主进程）
    if is_main_process:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
    
    # 设置tensorboard（只有主进程）
    writer = None
    if is_main_process:
        writer = SummaryWriter(log_dir=summary_path)
    
    # 创建数据集
    train_dataset = UDISDataset(root_dir=args.train_path)
    
    # 设置数据加载器
    if is_distributed:
        # 分布式采样器
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
        pin_memory=True,
        drop_last=True
    )
    
    # 创建模型
    model = ImprovedWarpNetwork().to(device)
    
    # 使用SyncBatchNorm（如果启用）
    if is_distributed and hasattr(args, 'sync_bn') and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_main_process:
            print("使用同步BatchNorm")
    
    # 将模型包装为DDP模型（如果启用分布式训练）
    if is_distributed:
        model = DDP(
            model, 
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True
        )
        if is_main_process:
            print("使用DistributedDataParallel")
    
    # 定义优化器
    if is_distributed:
        # 如果使用DDP，需要使用model.module获取实际模型
        optimizer = optim.Adam([
            {'params': model.module.feature_extractor.parameters(), 'lr': learning_rate},
            {'params': model.module.feature_matcher.parameters(), 'lr': learning_rate},
            {'params': model.module.valid_point_discriminator.parameters(), 'lr': learning_rate * 2},
            {'params': model.module.regressNet1_part1.parameters(), 'lr': learning_rate * 10},
            {'params': model.module.regressNet1_part2.parameters(), 'lr': learning_rate * 10},
            {'params': model.module.regressNet2_part1.parameters(), 'lr': learning_rate * 10},
            {'params': model.module.regressNet2_part2.parameters(), 'lr': learning_rate * 10}
        ])
    else:
        optimizer = optim.Adam([
            {'params': model.feature_extractor.parameters(), 'lr': learning_rate},
            {'params': model.feature_matcher.parameters(), 'lr': learning_rate},
            {'params': model.valid_point_discriminator.parameters(), 'lr': learning_rate * 2},
            {'params': model.regressNet1_part1.parameters(), 'lr': learning_rate * 10},
            {'params': model.regressNet1_part2.parameters(), 'lr': learning_rate * 10},
            {'params': model.regressNet2_part1.parameters(), 'lr': learning_rate * 10},
            {'params': model.regressNet2_part2.parameters(), 'lr': learning_rate * 10}
        ])
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=learning_rate * 0.01
    )
    
    # 从预训练权重恢复 - 检查现有权重
    start_epoch = 0
    pretrained_path = args.pretrained_path if hasattr(args, 'pretrained_path') else None
    
    # 从预训练权重恢复训练（如果提供了预训练路径）
    if pretrained_path:
        try:
            if os.path.isdir(pretrained_path):
                # 查找目录中最新的checkpoint
                ckpt_list = glob.glob(os.path.join(pretrained_path, "*.pth"))
                if ckpt_list:
                    # 按修改时间排序
                    ckpt_list.sort(key=os.path.getmtime, reverse=True)
                    checkpoint_path = ckpt_list[0]
                    if is_main_process:
                        print(f"找到最新的权重文件: {checkpoint_path}")
                else:
                    if is_main_process:
                        print(f"目录 {pretrained_path} 中没有找到权重文件，从头开始训练")
                    checkpoint_path = None
            else:
                # 直接使用指定的文件
                checkpoint_path = pretrained_path
                
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict):
                    if is_distributed:
                        # 处理分布式模型加载
                        if 'model' in checkpoint:
                            # 去掉"module."前缀（如果有）
                            state_dict = checkpoint['model']
                            new_state_dict = {}
                            for k, v in state_dict.items():
                                if k.startswith('module.'):
                                    k = k[7:]  # 去掉'module.'
                                new_state_dict[k] = v
                            
                            # 加载到模型中
                            if hasattr(model, 'module'):
                                model.module.load_state_dict(new_state_dict)
                            else:
                                model.load_state_dict(new_state_dict)
                            if is_main_process:
                                print("使用'model'键加载权重成功")
                        elif 'model_state_dict' in checkpoint:
                            # 去掉"module."前缀（如果有）
                            state_dict = checkpoint['model_state_dict']
                            new_state_dict = {}
                            for k, v in state_dict.items():
                                if k.startswith('module.'):
                                    k = k[7:]  # 去掉'module.'
                                new_state_dict[k] = v
                            
                            # 加载到模型中
                            if hasattr(model, 'module'):
                                model.module.load_state_dict(new_state_dict)
                            else:
                                model.load_state_dict(new_state_dict)
                            if is_main_process:
                                print("使用'model_state_dict'键加载权重成功")
                    else:
                        # 普通模型加载
                        if 'model' in checkpoint:
                            model.load_state_dict(checkpoint['model'])
                            if is_main_process:
                                print("使用'model'键加载权重成功")
                        elif 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                            if is_main_process:
                                print("使用'model_state_dict'键加载权重成功")
                        else:
                            # 尝试直接作为state_dict加载
                            try:
                                model.load_state_dict(checkpoint)
                                if is_main_process:
                                    print("直接加载权重成功")
                            except Exception as e:
                                if is_main_process:
                                    print(f"权重加载失败: {e}, 从头开始训练")
                    
                    # 如果存在optimizer和epoch信息，也加载它们
                    if 'optimizer' in checkpoint and checkpoint['optimizer']:
                        try:
                            optimizer.load_state_dict(checkpoint['optimizer'])
                            if is_main_process:
                                print("加载优化器状态成功")
                        except Exception as e:
                            if is_main_process:
                                print(f"优化器状态加载失败: {e}")
                    
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1
                        if is_main_process:
                            print(f"从轮次 {start_epoch} 开始续训")
                else:
                    try:
                        if is_distributed:
                            if hasattr(model, 'module'):
                                model.module.load_state_dict(checkpoint)
                            else:
                                model.load_state_dict(checkpoint)
                        else:
                            model.load_state_dict(checkpoint)
                        if is_main_process:
                            print("直接加载权重成功")
                    except Exception as e:
                        if is_main_process:
                            print(f"权重加载失败: {e}, 从头开始训练")
                        
                if is_main_process:
                    print(f"成功加载预训练权重: {checkpoint_path}")
            else:
                if is_main_process:
                    print(f"预训练权重路径不存在: {checkpoint_path}, 从头开始训练")
        except Exception as e:
            if is_main_process:
                print(f"加载预训练权重时出错: {e}")
                print("从头开始训练")
    else:
        if is_main_process:
            print("未提供预训练权重路径，从头开始训练")
        
    # 输出训练起始轮次信息
    if is_main_process:
        print(f"从轮次 {start_epoch} 开始训练，共 {num_epochs} 轮")
    
    # 设置损失权重
    homography_weight = args.homography_weight if hasattr(args, 'homography_weight') else 1.0
    mesh_weight = args.mesh_weight if hasattr(args, 'mesh_weight') else 1.0
    feature_weight = args.feature_weight if hasattr(args, 'feature_weight') else 0.1
    valid_point_weight = args.valid_point_weight if hasattr(args, 'valid_point_weight') else 0.5
    continuity_weight = args.continuity_weight if hasattr(args, 'continuity_weight') else 0.2
    
    # 定义损失函数
    criterion = {
        'homography': nn.MSELoss(),
        'mesh': nn.MSELoss(),
        'feature': nn.CosineSimilarity(),
        'valid_point': nn.BCEWithLogitsLoss()
    }
    
    # 自动混合精度训练设置
    use_amp = hasattr(args, 'use_amp') and args.use_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # 梯度累积步数
    grad_accum_steps = args.grad_accum_steps if hasattr(args, 'grad_accum_steps') else 1
    
    # 训练循环 - 从start_epoch开始
    for epoch in range(start_epoch, num_epochs):
        # 设置采样器的epoch（分布式训练）
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        total_continuity_loss = 0
        
        # 创建进度条（仅主进程）
        if is_main_process:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        else:
            progress_bar = train_loader
        
        # 重置梯度
        optimizer.zero_grad()
        
        # 训练一个epoch
        for batch_idx, (img1, img2, target) in enumerate(progress_bar):
            img1 = img1.to(device)
            img2 = img2.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            
            # 使用自动混合精度
            with torch.cuda.amp.autocast(enabled=use_amp):
                # 前向传播
                if is_distributed:
                    offset_1, offset_2, valid_scores, continuity_loss = model.module(img1, img2)
                else:
                    offset_1, offset_2, valid_scores, continuity_loss = model(img1, img2)
                
                # 将offset_1从[batch_size, 8]重塑为[batch_size, 4, 2]以匹配target['homography']的形状
                offset_1_reshaped = offset_1.reshape(-1, 4, 2)
                
                # 将offset_2重塑为[batch_size, (grid_h+1)*(grid_w+1), 2]以匹配target['mesh']
                # 根据grid_res.py，GRID_H=12, GRID_W=12，因此(GRID_H+1)*(GRID_W+1)=169
                offset_2_reshaped = offset_2.reshape(-1, 169, 2)
                
                # 计算损失
                loss_homography = criterion['homography'](offset_1_reshaped, target['homography'])
                loss_mesh = criterion['mesh'](offset_2_reshaped, target['mesh'])
                
                # 修改特征匹配损失计算
                # CosineSimilarity计算余弦相似度，但我们需要的是损失值(越小越好)
                # 因此用1减去余弦相似度的平均值作为损失
                if is_distributed:
                    cosine_sim = criterion['feature'](
                        model.module.feature_matcher.last_feature1.flatten(2), 
                        model.module.feature_matcher.last_feature2.flatten(2)
                    )
                else:
                    cosine_sim = criterion['feature'](
                        model.feature_matcher.last_feature1.flatten(2), 
                        model.feature_matcher.last_feature2.flatten(2)
                    )
                loss_feature = (1 - cosine_sim.mean())
                
                # 计算有效点判别损失
                batch_size, _, img_h, img_w = img1.size()
                with torch.no_grad():
                    H_motion = offset_1_reshaped  # 直接使用已经重塑的张量
                    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
                    if torch.cuda.is_available():
                        src_p = src_p.cuda()
                    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
                    dst_p = src_p + H_motion
                    
                    # 计算单应性矩阵H
                    H = torch_DLT.tensor_DLT(src_p/8, dst_p/8)
                    
                    # 使用H直接变换源点，而不是使用transformer函数
                    # 首先将源点转为齐次坐标
                    ones = torch.ones(batch_size, 4, 1).to(src_p.device)
                    src_p_hom = torch.cat([src_p, ones], dim=2)  # [batch_size, 4, 3]
                    
                    # 使用H变换点
                    dst_p_hom = torch.bmm(H, src_p_hom.transpose(1, 2)).transpose(1, 2)  # [batch_size, 4, 3]
                    
                    # 转回非齐次坐标
                    dst_p_transformed = dst_p_hom[:, :, :2] / dst_p_hom[:, :, 2:3]
                    
                    # 计算误差
                    error = torch.norm(dst_p - dst_p_transformed, dim=2)
                    valid_labels_raw = (error < 5.0).float()
                    
                    # 计算每个样本的平均有效点比例，形状为[batch_size, 1]
                    valid_labels = valid_labels_raw.mean(dim=1, keepdim=True)
                
                loss_valid = criterion['valid_point'](valid_scores, valid_labels)
                
                # 总损失
                total_batch_loss = (homography_weight * loss_homography + 
                                  mesh_weight * loss_mesh + 
                                  feature_weight * loss_feature + 
                                  valid_point_weight * loss_valid + 
                                  continuity_weight * continuity_loss)
                
                # 使用梯度累积，将损失除以累积步数
                if grad_accum_steps > 1:
                    total_batch_loss = total_batch_loss / grad_accum_steps
            
            # 使用scaler进行反向传播和梯度计算
            scaler.scale(total_batch_loss).backward()
            
            # 累积梯度达到指定步数后更新模型
            if (batch_idx + 1) % grad_accum_steps == 0:
                # 使用梯度裁剪（如果配置了）
                if hasattr(args, 'clip_grad') and args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # 统计损失
            total_loss += total_batch_loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1)
            total_continuity_loss += continuity_loss.item()
            
            # 写入tensorboard（仅主进程）
            if is_main_process:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/Total', total_batch_loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1), step)
                writer.add_scalar('Loss/Homography', loss_homography.item(), step)
                writer.add_scalar('Loss/Mesh', loss_mesh.item(), step)
                writer.add_scalar('Loss/Feature', loss_feature.item(), step)
                writer.add_scalar('Loss/ValidPoint', loss_valid.item(), step)
                writer.add_scalar('Loss/Continuity', continuity_loss.item(), step)
                writer.add_scalar('Metrics/ValidPoints', valid_scores.mean().item(), step)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], step)
                
                # 更新进度条
                progress_bar.set_postfix({
                    'Loss': f"{total_batch_loss.item():.4f}",
                    'Cont_Loss': f"{continuity_loss.item():.4f}",
                    'Valid_Points': f"{valid_scores.mean().item():.4f}"
                })
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_continuity_loss = total_continuity_loss / len(train_loader)
        
        # 打印训练信息（仅主进程）
        if is_main_process:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f}, '
                  f'Continuity Loss: {avg_continuity_loss:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # 写入epoch级别的tensorboard
            writer.add_scalar('Epoch/Loss', avg_loss, epoch)
            writer.add_scalar('Epoch/ContinuityLoss', avg_continuity_loss, epoch)
            writer.add_scalar('Epoch/LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存模型（仅主进程）
        if is_main_process and (epoch + 1) % 10 == 0:
            # 获取要保存的模型状态
            if is_distributed:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
                
            checkpoint = {
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'args': vars(args) if hasattr(args, '__dict__') else None
            }
            torch.save(checkpoint, os.path.join(model_save_path, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # 保存最新的checkpoint
            torch.save(checkpoint, os.path.join(model_save_path, 'checkpoint_latest.pth'))
        
        # 同步所有进程，确保每个epoch的训练完全结束后再开始下一个epoch
        if is_distributed:
            dist.barrier()
    
    # 保存最终模型（仅主进程）
    if is_main_process:
        # 获取要保存的模型状态
        if is_distributed:
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
            
        checkpoint = {
            'model': model_state,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': num_epochs - 1,
            'args': vars(args) if hasattr(args, '__dict__') else None
        }
        torch.save(checkpoint, os.path.join(model_save_path, 'checkpoint_final.pth'))
        
        # 关闭tensorboard writer
        if writer:
            writer.close()
    
    # 清理分布式环境
    if is_distributed:
        dist.barrier()
    
    return model


if __name__=="__main__":
    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_path', type=str, default='/opt/data/private/nl/Data/UDIS-D/training/')
    parser.add_argument('--model_save_path', type=str, default=os.path.join(last_path, 'model'))
    parser.add_argument('--summary_path', type=str, default=os.path.join(last_path, 'summary'))
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--homography_weight', type=float, default=1.0)
    parser.add_argument('--mesh_weight', type=float, default=1.0)
    parser.add_argument('--feature_weight', type=float, default=0.1)
    parser.add_argument('--valid_point_weight', type=float, default=0.5)
    parser.add_argument('--continuity_weight', type=float, default=0.2)
    
    # 添加新参数
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作线程数')
    parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度训练')
    parser.add_argument('--clip_grad', type=float, default=0.0, help='梯度裁剪阈值，0表示不裁剪')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--distributed', action='store_true', help='使用分布式训练')
    parser.add_argument('--sync_bn', action='store_true', help='使用同步BatchNorm')
    parser.add_argument('--local_rank', type=int, default=-1, help='分布式训练的local rank，由启动器自动设置')

    # parse the arguments
    args = parser.parse_args()
    print(args)

    # 根据是否启用分布式训练调用不同的训练函数
    if args.distributed and args.local_rank != -1:
        # 初始化分布式环境
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        # 设置当前设备
        torch.cuda.set_device(args.local_rank)
        
        # 训练
        train_distributed(args.local_rank, dist.get_world_size(), args)
        
        # 清理
        if dist.is_initialized():
            dist.destroy_process_group()
    else:
        # 单GPU训练
        train(args)


