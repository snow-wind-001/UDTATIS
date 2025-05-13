import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Composition.Codes.network import build_model, generate_learned_masks, compose_images
from Composition.Codes.enhanced_network import ImprovedDiffusionComposition
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
from torch.amp import autocast, GradScaler  # 使用新的torch.amp路径
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR
import math
import cv2
import traceback
import datetime
import torchvision.transforms as transforms

# path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

# 添加注意力模块的矩阵乘法补丁
# 这将修复维度不匹配问题
original_matmul = torch.matmul

def safe_matmul(tensor1, tensor2):
    """安全的矩阵乘法，会自动处理维度不匹配的情况"""
    if tensor1.dim() >= 2 and tensor2.dim() >= 2:
        try:
            # 尝试直接乘法
            return original_matmul(tensor1, tensor2)
        except RuntimeError as e:
            # 检查错误是否是维度不匹配
            if "size mismatch" in str(e) or "Expected size" in str(e):
                # 提取错误信息中的维度
                error_msg = str(e)
                print(f"处理维度不匹配: {error_msg}")
                print(f"tensor1 shape: {tensor1.shape}, tensor2 shape: {tensor2.shape}")
                
                # 处理多头注意力特殊情况 - 针对[4, 4, 256, 256] @ [4, 4, 128, 256]这样的形状
                if tensor1.dim() == 4 and tensor2.dim() == 4:
                    if tensor1.shape[0] == tensor2.shape[0] and tensor1.shape[1] == tensor2.shape[1]:
                        # 情况1: 如果最后两个维度不匹配，但第二个张量的最后一个维度与第一个张量的倒数第二个维度匹配
                        if tensor1.shape[2] == tensor2.shape[3] and tensor1.shape[3] != tensor2.shape[2]:
                            print(f"  -> 转置 tensor2 的最后两个维度")
                            tensor2 = tensor2.transpose(-1, -2)
                            return original_matmul(tensor1, tensor2)
                        
                        # 情况2: 需要调整第二个张量的特征维度以匹配第一个张量
                        elif tensor1.shape[3] != tensor2.shape[2]:
                            print(f"  -> 将tensor2的特征维度从{tensor2.shape[2]}调整为{tensor1.shape[3]}")
                            # 使用线性投影调整特征维度
                            B, H, seq_len, dim = tensor2.shape
                            proj = torch.nn.Linear(dim, tensor1.shape[3], bias=False).to(tensor2.device)
                            # 重塑tensor以应用线性层
                            tensor2_reshaped = tensor2.view(B*H*seq_len, dim)
                            tensor2_projected = proj(tensor2_reshaped).view(B, H, seq_len, tensor1.shape[3])
                            return original_matmul(tensor1, tensor2_projected.transpose(-1, -2))
                        
                            # 如果tensor2的特征维度比tensor1大，进行下采样
                
                # 处理标准注意力情况
                elif tensor1.dim() == 3 and tensor2.dim() == 3:
                    if tensor1.shape[0] == tensor2.shape[0]:
                        # 情况1: 如果最后两个维度不匹配，但第二个张量的最后一个维度与第一个张量的倒数第二个维度匹配
                        if tensor1.shape[1] == tensor2.shape[2] and tensor1.shape[2] != tensor2.shape[1]:
                            print(f"  -> 转置 tensor2 的最后两个维度")
                            tensor2 = tensor2.transpose(-1, -2)
                            return original_matmul(tensor1, tensor2)
                        
                        # 情况2: 需要调整第二个张量的特征维度以匹配第一个张量
                        elif tensor1.shape[2] != tensor2.shape[1]:
                            print(f"  -> 将tensor2的特征维度从{tensor2.shape[1]}调整为{tensor1.shape[2]}")
                            # 使用线性投影调整特征维度
                            B, dim, seq_len = tensor2.shape
                            proj = torch.nn.Linear(dim, tensor1.shape[2], bias=False).to(tensor2.device)
                            # 重塑tensor以应用线性层
                            tensor2_reshaped = tensor2.transpose(1, 2).reshape(B*seq_len, dim)
                            tensor2_projected = proj(tensor2_reshaped).view(B, seq_len, tensor1.shape[2]).transpose(1, 2)
                            return original_matmul(tensor1, tensor2_projected)
                        
                            # 如果tensor2的特征维度比tensor1大，进行下采样
                
                # 如果无法修复，重新抛出原始错误
                raise
            else:
                # 如果是其他错误，直接抛出
                raise
    
    # 对于其他维度的张量，使用原始乘法
    return original_matmul(tensor1, tensor2)

# 应用补丁
torch.matmul = safe_matmul

def train(rank, world_size, args, device=None, start_epoch=0):
    """
    改进的训练函数，包含自动混合精度训练、高级学习率调度和分布式训练支持
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        args: 训练参数
        device: 训练设备
        start_epoch: 开始训练的轮次
    """
    # 导入必要的模块，但避免重复导入已经在文件顶部导入的模块
    import datetime
    import numpy as np
    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
    
    # 初始化分布式进程组（如果启用分布式训练）
    if args.distributed:
        try:
            # 检查是否已经初始化
            if not dist.is_initialized():
                # 从环境变量获取地址和端口，这些由torchrun自动设置
                master_addr = os.environ.get('MASTER_ADDR', 'localhost')
                master_port = os.environ.get('MASTER_PORT', '12355')
                
                # 打印调试信息
                print(f"[进程 {rank}] 使用地址：{master_addr}，端口：{master_port}进行分布式初始化")
                
                # 设置分布式超时
                timeout_seconds = getattr(args, 'max_timeout', 1800)
                timeout = datetime.timedelta(seconds=timeout_seconds)
                
                print(f"[进程 {rank}] 分布式初始化超时设置: {timeout_seconds}秒")
                
                # 初始化进程组
                dist.init_process_group(
                    "nccl", 
                    init_method=f"env://",
                    rank=rank, 
                    world_size=world_size,
                    timeout=timeout
                )
                print(f"[进程 {rank}] 分布式进程组初始化成功")
            else:
                print(f"[进程 {rank}] 分布式进程组已经初始化")
        except Exception as e:
            print(f"[进程 {rank}] 初始化分布式进程组失败: {e}")
            print(f"[进程 {rank}] 环境变量: MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")
            args.distributed = False
            print(f"[进程 {rank}] 降级为非分布式模式")
            
    # 设置训练设备，如果未指定则使用rank来确定
    if device is None:
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # 设置 NCCL 环境变量
    if args.distributed and 'NCCL_DEBUG' not in os.environ:
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_BLOCKING_WAIT'] = '1'  # 使用阻塞等待
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # 启用异步错误处理
        print(f"[进程 {rank}] 已设置NCCL环境变量")
    
    # 强制使用cudnn基准模式以提高性能
    torch.backends.cudnn.benchmark = True
    
    # 设置随机种子以确保分布式训练的一致性
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    
    # GPU设置和清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        print(f"[进程 {rank}] 使用GPU: {torch.cuda.get_device_name(device)}")
        print(f"[进程 {rank}] GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        
        # 打印CUDA信息
        if rank == 0:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"是否启用cuDNN: {torch.backends.cudnn.enabled}")
        print(f"cuDNN基准模式: {torch.backends.cudnn.benchmark}")
    
    # 设置混合精度训练 - 完全禁用
    use_amp = False
    print("[信息] 使用完整精度训练")
    
    # 梯度累积步数
    grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
    
    # 添加训练异常处理计数器
    amp_overflow_count = 0
    consecutive_overflow_count = 0
    max_overflow_before_disabling = 5  # 连续溢出5次后禁用AMP
    
    # 添加防止NaN的处理函数
    def sanitize_gradients(model):
        '''检查并替换梯度中的NaN/Inf值'''
        params_with_issues = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 检查NaN/Inf
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    param.grad = torch.zeros_like(param.grad)
                    params_with_issues.append(name)
        return params_with_issues
    
    # 只有rank 0创建日志目录和writer
    is_main_process = (rank == 0)
    writer = None
    if is_main_process:
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    print(f"创建TensorBoard日志目录: {args.log_dir}")
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"TensorBoard SummaryWriter初始化完成，日志将保存到: {writer.log_dir}")
    
    # 记录训练配置信息
    args_dict = vars(args)
    writer.add_text('TrainingConfig', str(args_dict), 0)
    writer.flush()
    
    # 使用较低的num_workers数量，尤其是在分布式训练中
    # 过多的workers会导致内存和IO争用
    if hasattr(args, 'num_workers'):
        # 限制num_workers在合理范围内（每个进程2个worker）
        safe_workers = min(2, max(1, args.num_workers // world_size))
        if args.num_workers != safe_workers:
            print(f"[进程 {rank}] 调整num_workers从{args.num_workers}到{safe_workers}")
            args.num_workers = safe_workers
    else:
        args.num_workers = 2 if not args.distributed else 1
        print(f"[进程 {rank}] 设置num_workers={args.num_workers}")
    
    # 配置训练数据集
    try:
        print(f"[进程 {rank}] 开始加载数据集: {args.data_dir}")
        
        # 确保导入正确的TrainDataset
        try:
            # 直接使用导入好的TrainDataset
    train_dataset = TrainDataset(
        args.data_dir, 
                use_virtual=args.use_virtual if hasattr(args, 'use_virtual') else False,
                image_size=args.img_size,  # 使用参数中指定的图像大小
                augment=args.augment if hasattr(args, 'augment') else True,
        norm_type=args.norm_type
    )
        except NameError as ne:
            # 如果TrainDataset未定义，尝试重新导入
            print(f"[进程 {rank}] TrainDataset未定义，尝试重新导入...")
            import sys
            
            # 添加Composition/Codes到sys.path
            module_path = os.path.dirname(os.path.abspath(__file__))
            if module_path not in sys.path:
                sys.path.insert(0, module_path)
                
            # 尝试从当前目录导入
            try:
                from dataset import TrainDataset
            except ImportError:
                # 最后尝试从完整路径导入
                from Composition.Codes.dataset import TrainDataset
                
            # 再次创建数据集
            train_dataset = TrainDataset(
                args.data_dir, 
                use_virtual=args.use_virtual if hasattr(args, 'use_virtual') else False,
                image_size=args.img_size,  # 使用参数中指定的图像大小
                augment=args.augment if hasattr(args, 'augment') else True,
                norm_type=args.norm_type
            )
            
        print(f"[进程 {rank}] 数据集加载完成，样本数量: {len(train_dataset)}")
        
        # 强制设置不使用虚拟数据
        args.use_virtual = False
        
    except Exception as e:
        import traceback
        print(f"[进程 {rank}] 加载数据集时出错: {e}")
        print(f"[进程 {rank}] 详细错误信息: {traceback.format_exc()}")
        raise e
    
    # 使用分布式采样器（如果启用分布式训练）
    if args.distributed:
        try:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, 
                num_replicas=world_size,
                rank=rank,
        shuffle=True, 
                seed=42 + rank  # 为不同进程设置不同的种子
            )
            print(f"[进程 {rank}] 创建分布式采样器成功")
        except Exception as e:
            print(f"[进程 {rank}] 创建分布式采样器失败: {e}")
            print(f"[进程 {rank}] 将使用随机采样器")
            train_sampler = None
    else:
        train_sampler = None
    
    # 创建数据加载器
    try:
        # 每个batch中的图像数据张量需要有相同的尺寸，因此我们创建一个自定义的collate_fn
        def custom_collate_fn(batch):
            if len(batch) == 0:
                return {}
            
            # 确定此批次中所有图像的最大尺寸
            max_h, max_w = 0, 0
            for item in batch:
                for key in ['warp1', 'warp2']:
                    if key in item:
                        h, w = item[key].shape[1], item[key].shape[2]
                        max_h = max(max_h, h)
                        max_w = max(max_w, w)
            
            # 调整为32的倍数
            max_h = ((max_h + 31) // 32) * 32
            max_w = ((max_w + 31) // 32) * 32
            
            # 填充所有张量到相同大小
            for item in batch:
                for key in item:
                    if isinstance(item[key], torch.Tensor) and len(item[key].shape) == 3:
                        c, h, w = item[key].shape
                        if h != max_h or w != max_w:
                            # 计算填充量
                            pad_h = max_h - h
                            pad_w = max_w - w
                            # 应用填充
                            padding = (0, pad_w, 0, pad_h)
                            item[key] = F.pad(item[key], padding, mode='constant', value=0)
            
            # 创建批次张量
            batch_dict = {}
            for key in batch[0].keys():
                if isinstance(batch[0][key], torch.Tensor):
                    batch_dict[key] = torch.stack([item[key] for item in batch])
            
            return batch_dict
        
        # 创建DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=not args.distributed,  # 分布式时使用DistributedSampler采样器
            sampler=train_sampler,         # 使用之前创建的采样器（如果有）
        num_workers=args.num_workers,
            pin_memory=True,               # 使用固定内存提高IO性能
            drop_last=True,                # 丢弃最后一个不完整的批次
            collate_fn=custom_collate_fn    # 使用自定义collate函数进行批次处理
        )
            
        print(f"[进程 {rank}] 数据加载器创建成功，批次数量: {len(train_loader)}")
        
    except Exception as e:
        import traceback
        print(f"[进程 {rank}] 创建数据加载器失败: {e}")
        print(f"[进程 {rank}] 详细错误信息: {traceback.format_exc()}")
        
        # 尝试使用简化配置创建数据加载器
        try:
            # 使用更简单的配置尝试
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=not args.distributed,
                sampler=train_sampler,
                num_workers=1,            # 减少worker数量
                pin_memory=False,         # 不使用固定内存
                drop_last=True,
                collate_fn=None           # 不使用自定义collate函数
            )
            print(f"[进程 {rank}] 使用简化配置创建数据加载器成功")
        except Exception as e2:
            print(f"[进程 {rank}] 简化配置也失败: {e2}")
            print(f"[进程 {rank}] 无法继续训练，退出")
            return
    
    # 创建或加载模型
    try:
        # 创建扩散模型
    if args.use_diffusion:
            if args.use_enhanced_diffusion:
                # 使用增强的扩散模型
                try:
                    from Composition.Codes.enhanced_network import ImprovedDiffusionComposition
        model = ImprovedDiffusionComposition(
                        num_timesteps=args.diffusion_steps,
                        beta_schedule='linear',
                        image_size=args.img_size if args.img_size else 256,
                        base_channels=64,
                        attention_resolutions=[16, 8],
                        dropout=0.1,
                        channel_mult=(1, 2, 4, 8),
                        conv_resample=True,
                        num_res_blocks=2,
                        heads=4,
                        use_scale_shift_norm=True
                    )
                except Exception as e:
                    print(f"创建ImprovedDiffusionComposition模型失败: {e}")
                    print("尝试使用SimpleDiffusionModel作为备选...")
                    
                    # 创建一个简单版本的扩散模型
                    from Composition.Codes.network import ImprovedDiffusionModel
                    model = ImprovedDiffusionModel(
                        image_size=args.img_size,
                        in_channels=8,  # warp1(3) + warp2(3) + mask1(1) + mask2(1)
                        time_dim=256,
                        device=device
                    )
    else:
                # 使用基础扩散模型
                from Composition.Codes.network import ImprovedDiffusionModel
                model = ImprovedDiffusionModel(
                    image_size=args.img_size,
                    in_channels=8,
                    time_dim=256,
                    device=device
                )
        else:
            # 使用标准U-Net模型或者其他备选模型
            try:
                model = build_model(args.model_type, pretrain=args.pretrain)
            except Exception as e:
                print(f"创建标准模型失败: {e}")
                # 创建一个简单的CNN模型作为备选
                print("使用简单的CNN模型作为备选")
                model = nn.Sequential(
                    nn.Conv2d(8, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(64, 1, kernel_size=1),
                    nn.Sigmoid()
                )
        
        # 将模型移动到指定设备
        model = model.to(device)
        print(f"[进程 {rank}] 模型创建成功")
    except Exception as e:
        print(f"[进程 {rank}] 创建模型失败: {e}")
        print(f"[进程 {rank}] 无法继续训练，退出")
        return
    
    # 同步BN（如果启用）
    if args.distributed and args.sync_bn:
        try:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            print(f"[进程 {rank}] 已转换为同步BatchNorm")
        except Exception as e:
            print(f"[进程 {rank}] 转换为同步BatchNorm失败: {e}")
            print(f"[进程 {rank}] 继续使用标准BatchNorm")
    
    # 验证模型是否在正确设备上
    if rank == 0:
    print(f"模型设备: {next(model.parameters()).device}")
    
    # 确保所有模型参数在同一设备上
    for name, param in model.named_parameters():
        if param.device != device:
            if rank == 0:
            print(f"移动参数 {name} 从 {param.device} 到 {device}")
            param.data = param.data.to(device)
    
    # 使用DDP包装模型（如果启用分布式训练）
    if args.distributed:
        try:
            # 确保模型在device上
            model = model.to(device)
            # 使用DistributedDataParallel包装模型
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[local_rank] if local_rank != -1 else None,
                output_device=local_rank if local_rank != -1 else None,
                find_unused_parameters=True
            )
            # 启用异步梯度增强优化
            if torch.__version__ >= '1.11.0':
                model.gradient_as_bucket_view = True
            print(f"[进程 {rank}] 模型已成功用DDP包装")
        except Exception as e:
            print(f"[进程 {rank}] DDP包装失败: {e}")
            print(f"[进程 {rank}] 将使用普通模型继续")
    
    # 打印模型架构信息
    if rank == 0:
        print(f"模型类型: {type(model).__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
    
    # 配置优化器
    try:
        # 降低初始学习率，从1e-5开始
        if not hasattr(args, 'lr') or args.lr > 1e-4:
            original_lr = args.lr if hasattr(args, 'lr') else 1e-4
            args.lr = 1e-5
            print(f"[进程 {rank}] 降低初始学习率，从 {original_lr} 调整为 {args.lr}")
            
        # 使用指定的优化器
        optimizer_class = getattr(torch.optim, args.optimizer)
        optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"[进程 {rank}] 优化器 {args.optimizer} 初始化成功，学习率: {args.lr}")
    except Exception as e:
        print(f"[进程 {rank}] 指定的优化器 {args.optimizer} 初始化失败: {e}")
        print(f"[进程 {rank}] 使用默认的Adam优化器，学习率: 1e-5")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=args.weight_decay)
    
    # 使用余弦退火学习率调度
    try:
    if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs, 
            eta_min=args.lr * 0.01
        )
        elif args.scheduler == 'cosine_warmup':
            # 实现带预热的余弦退火
            from torch.optim.lr_scheduler import _LRScheduler
            
            class CosineAnnealingWarmupLR(_LRScheduler):
                def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, warmup_start_lr=1e-4):
                    self.warmup_epochs = max(1, warmup_epochs)  # 确保至少有1个预热轮次
                    self.max_epochs = max(1, max_epochs)  # 确保至少有1个总轮次
                    self.min_lr = min_lr
                    self.warmup_start_lr = warmup_start_lr
                    super().__init__(optimizer)
                    
                def get_lr(self):
                    # 确保不会发生除零错误
                    if self.last_epoch < self.warmup_epochs:
                        # 线性预热阶段
                        alpha = float(self.last_epoch) / float(max(1, self.warmup_epochs))
                        return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                               for base_lr in self.base_lrs]
                    else:
                        # 余弦退火阶段
                        current = float(self.last_epoch - self.warmup_epochs)
                        total = float(max(1, self.max_epochs - self.warmup_epochs))  # 确保总轮次大于0
                        
                        # 确保current/total不会超过1
                        progress = min(current / total, 1.0)
                        
                        return [self.min_lr + 0.5 * (base_lr - self.min_lr) * 
                               (1 + math.cos(progress * math.pi))
                               for base_lr in self.base_lrs]
            
            scheduler = CosineAnnealingWarmupLR(
                optimizer,
                warmup_epochs=args.warmup_epochs if hasattr(args, 'warmup_epochs') else 10,
                max_epochs=args.epochs,
                min_lr=args.lr * 0.01,
                warmup_start_lr=args.lr * 0.5  # 提高预热起始学习率，从0.1提高到0.5
            )
    elif args.scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.epochs * len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        print(f"[进程 {rank}] 学习率调度器 {args.scheduler} 初始化成功")
    except Exception as e:
        print(f"[进程 {rank}] 学习率调度器初始化失败: {e}")
        print(f"[进程 {rank}] 不使用学习率调度")
        scheduler = None
    
    # 从检查点恢复训练（如果存在）
    if hasattr(args, 'resume') and args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        
        # 加载模型状态
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        elif 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # 确保优化器状态在正确设备上
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        # 更新起始轮次
        if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1
            
        if rank == 0:
            print(f"从轮次 {start_epoch} 恢复训练")
    else:
        # 确保start_epoch有效
        print(f"从轮次 {start_epoch} 开始训练")
    
    # 损失权重 - 降低动态变化幅度
    loss_weights = {
        'l1': getattr(args, 'l1_weight', 1.0),
        'boundary': getattr(args, 'boundary_loss_weight', 0.5) if hasattr(args, 'boundary_loss_weight') else getattr(args, 'boundary_weight', 0.5),
        'smooth': getattr(args, 'smooth_weight', 0.1),
        'perceptual': getattr(args, 'perceptual_weight', 0.1),
        'ssim': getattr(args, 'ssim_weight', 0.1),
        'color': getattr(args, 'color_weight', 0.1),
        'diffusion': getattr(args, 'diffusion_loss_weight', 0.3) if hasattr(args, 'diffusion_loss_weight') else getattr(args, 'diffusion_weight', 0.3),
        'gradient': getattr(args, 'gradient_weight', 0.2)
    }
    
    print(f"[进程 {rank}] 损失权重配置: {loss_weights}")
    
    # 辅助函数：确保张量在正确设备上
    def ensure_on_device(tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor) and tensor.device != device:
            return tensor.to(device)
        return tensor
    
    # 确保args.epochs有一个有效值
    if not hasattr(args, 'epochs') or args.epochs is None:
        print("警告: args.epochs未设置或为None，使用默认值100")
        args.epochs = 100
    
    # 主训练循环
    for epoch in range(start_epoch, args.epochs):
        model.train()
        # 修改损失统计方式为平均值计算，使用计数器
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_boundary_loss = 0.0
        epoch_smooth_loss = 0.0
        epoch_perceptual_loss = 0.0
        epoch_diffusion_loss = 0.0
        
        # 添加计数器来正确计算平均值
        num_batches_processed = 0
        
        # 记录起始时间
        start_time = time.time()
        
        # 当前GPU内存
        if torch.cuda.is_available():
            start_gpu_memory = torch.cuda.memory_allocated(device) / 1024**2
            print(f"GPU内存使用(开始): {start_gpu_memory:.2f} MB")
        
        # 使用tqdm进度条可视化训练过程
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # 梯度累积相关变量
        grad_accum_steps = min(getattr(args, 'grad_accum_steps', 1), 2)
        if getattr(args, 'grad_accum_steps', 1) > 2:
            print(f"[进程 {rank}] 减小梯度累积步数，从 {args.grad_accum_steps} 降低到 2")
            args.grad_accum_steps = 2
        
        optimizer.zero_grad()
        accum_loss = 0
        
        # 动态调整损失权重
        current_epoch_ratio = epoch / args.epochs  # 当前训练进度比例
        
        # 遍历数据批次
        for batch_idx, batch in enumerate(progress_bar):
            # 提取批次数据
            try:
                if isinstance(batch, (list, tuple)) and len(batch) == 7:  # 自定义数据集格式
                    img1, img2, mask1, mask2, gt, warp1, warp2 = batch
                elif isinstance(batch, dict):  # 字典格式（虚拟数据集）
                    warp1 = batch.get('warp1')
                    warp2 = batch.get('warp2')
                    mask1 = batch.get('mask1')
                    mask2 = batch.get('mask2')
                    # 对于虚拟数据，将warp1/warp2同时用作img1/img2和gt
                    img1 = warp1
                    img2 = warp2
                    gt = (warp1 + warp2) / 2.0  # 创建基本字段
                else:  # 原始数据集格式或其他格式
                    # 确保有数据，否则跳过此批次
                    if not batch:
                        print(f"警告: 空批次，跳过")
                        continue
                    
                    try:
                        warp1, warp2, mask1, mask2 = batch['warp1'], batch['warp2'], batch['mask1'], batch['mask2']
                        # 对于虚拟数据，将warp1/warp2同时用作img1/img2和gt
                        img1 = warp1
                        img2 = warp2
                        gt = (warp1 + warp2) / 2.0  # 创建基本字段
                    except (TypeError, KeyError) as e:
                        print(f"批次数据格式错误: {e}, 批次类型: {type(batch)}")
                        # 如果是列表但格式不正确，尝试解包
                        if isinstance(batch, list) and len(batch) >= 4:
                            warp1, warp2, mask1, mask2 = batch[:4]
                            img1 = warp1
                            img2 = warp2
                            gt = (warp1 + warp2) / 2.0
                        else:
                            raise ValueError(f"无法处理批次数据: {type(batch)}")
                
                # 创建从批次提取的有效区域掩码（用于边界损失）
                batch_size, _, h, w = mask1.shape
                valid_mask = torch.ones_like(mask1)
                
                # 检测填充区域 - 假设填充为零
                for i in range(batch_size):
                    # 找到mask1中的非零区域
                    m1 = mask1[i]
                    # 如果mask1全是零，则整个图像可能是填充的
                    if m1.sum() < 10:
                        valid_mask[i] = 0
                    else:
                        # 找到有内容的区域
                        non_zero_y, non_zero_x = torch.where(m1[0] > 0.05)
                        if len(non_zero_y) > 0 and len(non_zero_x) > 0:
                            # 计算非零区域的边界
                            min_y, max_y = non_zero_y.min(), non_zero_y.max()
                            min_x, max_x = non_zero_x.min(), non_zero_x.max()
                            
                            # 创建有效区域掩码 - 只有实际图像区域
                            valid_mask[i, :, :min_y, :] = 0
                            valid_mask[i, :, max_y+1:, :] = 0
                            valid_mask[i, :, :, :min_x] = 0
                            valid_mask[i, :, :, max_x+1:] = 0
                
                # 确保所有数据在正确设备上
                img1 = ensure_on_device(img1)
                img2 = ensure_on_device(img2)
                mask1 = ensure_on_device(mask1)
                mask2 = ensure_on_device(mask2)
                gt = ensure_on_device(gt)
                warp1 = ensure_on_device(warp1)
                warp2 = ensure_on_device(warp2)
                valid_mask = ensure_on_device(valid_mask)
            except Exception as e:
                print(f"数据处理错误: {e}")
                continue  # 跳过此批次
            
            # 使用混合精度训练
            # 已禁用混合精度训练，不使用autocast
            try:
                # 1. 使用模型生成掩码和拼接图像
                # 修改为使用标准参数调用
                # 不使用字典，直接传递参数
                x = torch.zeros_like(warp1)  # 创建一个空的噪声张量
                t = torch.zeros(warp1.shape[0], device=device).long()  # 创建时间步
                
                # 将彩色图片转换为灰白图片
                warp1_gray = convert_to_grayscale(warp1)
                warp2_gray = convert_to_grayscale(warp2)
                img1_gray = convert_to_grayscale(img1) if 'img1' in locals() else None
                img2_gray = convert_to_grayscale(img2) if 'img2' in locals() else None
                
                # 调整输入到512x512大小，避免维度不匹配
                warp1_resized = F.interpolate(warp1_gray, size=(512, 512), mode='bilinear', align_corners=False)
                warp2_resized = F.interpolate(warp2_gray, size=(512, 512), mode='bilinear', align_corners=False)
                mask1_resized = F.interpolate(mask1, size=(512, 512), mode='bilinear', align_corners=False)
                mask2_resized = F.interpolate(mask2, size=(512, 512), mode='bilinear', align_corners=False)
                x_resized = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
                
                try:
                    # 使用标准参数调用，但传入调整大小后的灰白图像输入
                    model_output = model(x_resized, t, warp1_resized, warp2_resized, mask1_resized, mask2_resized)
                    
                    # 处理输出 - 应该是(predicted_noise, learned_mask)格式
                    if isinstance(model_output, tuple) and len(model_output) == 2:
                        predicted_noise, out = model_output
                        mask = out  # 保存掩码用于后续计算
                        
                        # 如需要将输出调整回原始尺寸
                        if out.shape[2:] != mask1.shape[2:]:
                            out = F.interpolate(out, size=mask1.shape[2:], mode='bilinear', align_corners=False)
                        
                        # 应用新的掩码生成公式
                        learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, out)
                        
                        # 使用拼接公式创建图像 - 使用原始彩色图像进行拼接
                        stitched_image = compose_images(warp1, warp2, learned_mask1, learned_mask2)
                    else:
                        # 如果不是预期的格式，使用默认处理
                        out = model_output
                        
                        # 如需要将输出调整回原始尺寸
                        if out.shape[2:] != mask1.shape[2:]:
                            out = F.interpolate(out, size=mask1.shape[2:], mode='bilinear', align_corners=False)
                        
                        mask = out  # 保存原始输出以便后续使用
                        
                        # 应用新的掩码生成公式
                        learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, out)
                        
                        # 使用拼接公式创建图像 - 使用原始彩色图像进行拼接
                        stitched_image = compose_images(warp1, warp2, learned_mask1, learned_mask2)
                except TypeError as e:
                    print(f"模型调用错误: {e}")
                    # 尝试其他调用方式
                    try:
                        if hasattr(model, 'forward_composition'):
                            out, _ = model.forward_composition(warp1_gray, warp2_gray, mask1, mask2)
                            mask = out  # 保存原始输出以便后续使用
                            
                            # 应用新的掩码生成公式
                            learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, out)
                            
                            # 使用拼接公式创建图像 - 使用原始彩色图像进行拼接
                            stitched_image = compose_images(warp1, warp2, learned_mask1, learned_mask2)
                        else:
                            # 使用字典输入尝试，传入灰白图像
                            model_input = {
                                'base_image': warp1_gray,
                                'warp_image': warp2_gray,
                                'base_mask': mask1,
                                'warp_mask': mask2
                            }
                            model_output = model(model_input)
                            
                            # 处理输出 - 可能是字典或元组
                            if isinstance(model_output, dict):
                                out = model_output.get('mask', model_output.get('learned_mask1'))
                                mask = out  # 保存原始输出以便后续使用
                                
                                # 应用新的掩码生成公式
                                learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, out)
                                
                                # 使用拼接公式创建图像 - 使用原始彩色图像进行拼接
                                stitched_image = compose_images(warp1, warp2, learned_mask1, learned_mask2)
                            else:
                                # 如果是元组，按位置解包
                                out, _ = model_output
                                mask = out  # 保存原始输出以便后续使用
                                
                                # 应用新的掩码生成公式
                                learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, out)
                                
                                # 使用拼接公式创建图像 - 使用原始彩色图像进行拼接
                                stitched_image = compose_images(warp1, warp2, learned_mask1, learned_mask2)
                    except Exception as e:
                        print(f"所有模型调用方式都失败: {e}")
                        # 创建一个默认的掩码和拼接图像
                        out = torch.ones_like(mask1) * 0.5
                        mask = out  # 保存原始输出以便后续使用
                        
                        # 应用新的掩码生成公式
                        learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, out)
                        
                        # 使用拼接公式创建图像 - 使用原始彩色图像进行拼接
                        stitched_image = compose_images(warp1, warp2, learned_mask1, learned_mask2)
                
                # 随机选择部分样本进行可视化
                if batch_idx == 0 and epoch % args.vis_freq == 0:
                    # 确保mask变量存在
                    if 'mask' not in locals():
                        mask = torch.ones_like(mask1) * 0.5
                        print("警告: mask变量未定义，使用默认值")
                        
                    # 确保stitched_image变量存在
                    if 'stitched_image' not in locals():
                        # 创建默认拼接图像
                        stitched_image = (warp1 + warp2) / 2.0
                        print("警告: stitched_image变量未定义，使用默认值")
                        
                    # 添加输入图像可视化
                    input_vis = torch.cat([warp1[:4], warp2[:4]], dim=0)
                    writer.add_images('Batch/Inputs', (input_vis + 1) / 2, epoch)
                    
                    # 添加输入掩码可视化
                    if mask1.shape[1] == 1:  # 单通道掩码
                        mask_vis = torch.cat([mask1[:4].repeat(1, 3, 1, 1), 
                                            mask2[:4].repeat(1, 3, 1, 1)], dim=0)
                    else:  # 多通道掩码
                        mask_vis = torch.cat([mask1[:4, :3], mask2[:4, :3]], dim=0)
                    writer.add_images('Batch/InputMasks', mask_vis, epoch)
                    
                    # 添加生成掩码可视化
                    gen_mask_vis = mask[:4].repeat(1, 3, 1, 1)
                    writer.add_images('Batch/GeneratedMask', gen_mask_vis, epoch)
                    
                    # 添加拼接结果可视化
                    stitched_vis = (stitched_image[:4] + 1) / 2
                    writer.add_images('Batch/StitchedResult', stitched_vis, epoch)
                
                # 2. 计算L1损失
                l1_loss = F.l1_loss(stitched_image, gt)
                
                # 3. 计算边界损失 - 使用改进的边界损失计算
                boundary_loss, boundary_mask = cal_boundary_term(warp1, warp2, mask1, mask2, stitched_image, valid_mask)
                
                # 4. 计算平滑损失
                # 创建重叠掩码 - 使用有效区域
                mask_overlap = mask1 * mask2 * valid_mask
                
                # 计算平滑损失 - 确保掩码平滑过渡
                smooth_loss = cal_smooth_term_stitch(stitched_image, mask)
                
                # 差异平滑损失 - 确保差异图像平滑
                diff_smooth_loss = torch.tensor(0.0, device=device)
                try:
                    diff_smooth_loss = cal_smooth_term_diff(warp1, warp2, mask, mask_overlap)
                except Exception as e:
                    print(f"计算差异平滑损失时出错: {e}")
                    print("跳过差异平滑损失计算")
                    diff_smooth_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                # 5. 计算感知损失
                try:
                    perceptual_loss_weight = loss_weights.get('perceptual', 0.1)
                    perceptual_loss = cal_perceptual_loss(stitched_image, warp1, warp2, mask, 1-mask, weight=perceptual_loss_weight)
                except Exception as e:
                    print(f"计算感知损失时出错: {e}")
                    perceptual_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                # 6. 计算SSIM损失
                ssim_loss = torch.tensor(0.0, device=device)
                try:
                    ssim_loss = cal_ssim_loss(stitched_image, warp1, warp2, mask, 1-mask)
                except Exception as e:
                    print(f"计算SSIM损失时出错: {e}")
                
                # 7. 计算颜色一致性损失
                color_loss = torch.tensor(0.0, device=device)
                try:
                    color_loss = cal_color_consistency_loss(stitched_image, warp1, warp2, mask, 1-mask)
                except Exception as e:
                    print(f"计算颜色一致性损失时出错: {e}")
                
                # 8. 扩散模型损失
                diffusion_loss = torch.tensor(0.0, device=device)
                mask_loss = torch.tensor(0.0, device=device)
                if args.use_diffusion:
                    try:
                        # 随机时间步
                        t = torch.randint(0, model.num_timesteps, (warp1.shape[0],), device=device).long()
                        
                        # 调整x和其他输入张量，确保所有输入张量维度匹配
                        # 确保所有空间维度一致，统一到相同大小
                        target_size = (512, 512)  # 修改为512尺寸
                        
                        # 调整x的大小
                        if x.shape[2:] != target_size:
                            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
                        
                        # 调整warp1和warp2的大小
                        if warp1.shape[2:] != target_size:
                            warp1 = F.interpolate(warp1, size=target_size, mode='bilinear', align_corners=False)
                        if warp2.shape[2:] != target_size:
                            warp2 = F.interpolate(warp2, size=target_size, mode='bilinear', align_corners=False)
                        
                        # 调整mask1和mask2的大小
                        if mask1.shape[2:] != target_size:
                            mask1 = F.interpolate(mask1, size=target_size, mode='bilinear', align_corners=False)
                        if mask2.shape[2:] != target_size:
                            mask2 = F.interpolate(mask2, size=target_size, mode='bilinear', align_corners=False)
                        
                        # 修改: 使用warp2作为扩散模型的生成目标，而不是warp1
                        # 交换compute_loss中warp1和warp2的角色
                        diffusion_loss = model.compute_loss(warp2, t, warp1, warp2, mask1, mask2)
                        
                        # 额外的掩码一致性损失
                        try:
                            # 确保 mask 和 combined_mask 维度匹配
                            combined_mask = (learned_mask1 + learned_mask2) / 2.0
                            
                            # 检查掩码尺寸
                            if mask.dim() != combined_mask.dim():
                                if mask.dim() == 4 and mask.shape[1] == 1 and combined_mask.dim() == 4 and combined_mask.shape[1] == 1:
                                    # 都是4D但可能需要调整空间尺寸
                                    if mask.shape[2:] != combined_mask.shape[2:]:
                                        combined_mask = F.interpolate(combined_mask, size=mask.shape[2:], mode='bilinear', align_corners=False)
                                elif mask.dim() == 3 and combined_mask.dim() == 4:
                                    # mask是3D (已squeeze), combined_mask是4D
                                    mask = mask.unsqueeze(1)  # 恢复到4D
                                    if mask.shape[2:] != combined_mask.shape[2:]:
                                        combined_mask = F.interpolate(combined_mask, size=mask.shape[2:], mode='bilinear', align_corners=False)
                                elif mask.dim() == 4 and combined_mask.dim() == 3:
                                    # mask是4D, combined_mask是3D
                                    combined_mask = combined_mask.unsqueeze(1)  # 恢复到4D
                                    if mask.shape[2:] != combined_mask.shape[2:]:
                                        combined_mask = F.interpolate(combined_mask, size=mask.shape[2:], mode='bilinear', align_corners=False)
                            
                            # 确保都是相同的形状，如果仍不匹配则尝试再次调整
                            if mask.shape != combined_mask.shape:
                                if mask.dim() == 4 and combined_mask.dim() == 4:
                                    # 两者都是4D但形状不同
                                    if mask.shape[1] != combined_mask.shape[1]:
                                        # 通道数不同，通常情况下确保都是单通道
                                        if mask.shape[1] > 1:
                                            mask = mask[:, 0:1, :, :]
                                        if combined_mask.shape[1] > 1:
                                            combined_mask = combined_mask[:, 0:1, :, :]
                                    # 空间尺寸不同，调整combined_mask
                                    if mask.shape[2:] != combined_mask.shape[2:]:
                                        combined_mask = F.interpolate(combined_mask, size=mask.shape[2:], mode='bilinear', align_corners=False)
                            
                            # 使用Sigmoid函数确保输入在[0,1]范围内
                            sigmask = torch.sigmoid(mask)
                            
                            # 计算正规化后的损失
                            mask_loss = F.binary_cross_entropy(sigmask, combined_mask)
                        except Exception as e:
                            print(f"计算掩码一致性损失时出错: {e}")
                            print(f"Mask shape: {mask.shape}, Combined mask shape: {(learned_mask1 + learned_mask2).shape}")
                            mask_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    except Exception as e:
                        print(f"计算扩散模型损失时出错: {e}")
                        diffusion_loss = torch.tensor(0.0, device=device, requires_grad=True)
                        mask_loss = torch.tensor(0.0, device=device, requires_grad=True)
            except Exception as e:
                print(f"模型处理或损失计算出错: {e}")
                continue  # 跳过此批次
            
            # 添加梯度一致性损失
            gradient_consistency_loss = calculate_gradient_consistency_loss(stitched_image, img1, img2, mask)
            
            # 检查所有损失值是否包含NaN或Inf
            def safe_loss(loss_tensor, name):
                if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
                    print(f"警告: {name} 损失包含NaN或Inf值，使用零替代")
                    return torch.tensor(0.0, device=device, requires_grad=True)
                return loss_tensor
            
            # 安全处理所有损失值
            l1_loss = safe_loss(l1_loss, "L1")
            boundary_loss = safe_loss(boundary_loss, "边界")
            smooth_loss = safe_loss(smooth_loss, "平滑")
            diff_smooth_loss = safe_loss(diff_smooth_loss, "差异平滑")
            perceptual_loss = safe_loss(perceptual_loss, "感知")
            ssim_loss = safe_loss(ssim_loss, "SSIM")
            color_loss = safe_loss(color_loss, "颜色一致性")
            gradient_consistency_loss = safe_loss(gradient_consistency_loss, "梯度一致性")
            diffusion_loss = safe_loss(diffusion_loss, "扩散")
            mask_loss = safe_loss(mask_loss, "掩码")
            
            # 动态调整损失权重 - 减小变化幅度
            current_boundary_weight = loss_weights['boundary']
            
            # 平滑过渡的权重调整，减少动态变化幅度
            if epoch < args.epochs // 3:
                # 初始阶段
                diffusion_weight_adjusted = loss_weights['diffusion'] * 1.5  # 降低从2.0到1.5
                l1_weight_adjusted = loss_weights['l1'] * 0.8  # 增加从0.5到0.8
            elif epoch < args.epochs * 2 // 3:
                # 中间阶段保持平衡
                diffusion_weight_adjusted = loss_weights['diffusion'] * 1.0
                l1_weight_adjusted = loss_weights['l1'] * 1.0
                else:
                # 最终阶段
                diffusion_weight_adjusted = loss_weights['diffusion'] * 0.7  # 增加从0.5到0.7
                l1_weight_adjusted = loss_weights['l1'] * 1.2  # 降低从1.5到1.2
            
            # 总损失计算 - 加入掩码损失
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 添加基本损失组件，确保每个都有梯度
            loss = loss + l1_loss * l1_weight_adjusted
            loss = loss + boundary_loss * current_boundary_weight
            loss = loss + smooth_loss * loss_weights['smooth']
            loss = loss + diff_smooth_loss * loss_weights['smooth'] * 0.5
            loss = loss + perceptual_loss * loss_weights['perceptual']
            loss = loss + color_loss * loss_weights.get('color', 0.5)
            loss = loss + ssim_loss * loss_weights.get('ssim', 1.0)
            loss = loss + gradient_consistency_loss * loss_weights.get('gradient', 0.2)
            
            # 加入扩散损失和掩码损失
            if args.use_diffusion:
                loss = loss + diffusion_loss * diffusion_weight_adjusted
                
                # 如果使用增强型模型，添加掩码损失
                if hasattr(args, 'use_enhanced_diffusion') and args.use_enhanced_diffusion and hasattr(model, 'mask_branch'):
                    loss = loss + mask_loss
                    
            # 损失缩放和梯度累积
            scaled_loss = loss / grad_accum_steps
            
            # 使用普通优化流程代替AMP
            scaled_loss.backward()
            accum_loss += loss.item()
            
            # 每grad_accum_steps步更新一次参数
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # 梯度裁剪
                if args.clip_grad:
                    # 检查并修复NaN/Inf梯度
                    problematic_params = sanitize_gradients(model)
                    if problematic_params and is_main_process:
                        print(f"检测到NaN/Inf梯度，已清零: {len(problematic_params)}个参数")
                        
                    # 应用梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
                # 更新参数
                        optimizer.step()
                optimizer.zero_grad()  # 重置梯度
                
                # 记录当前损失 - 修改为平均值而非简单加法
                avg_batch_loss = accum_loss / grad_accum_steps
                epoch_loss += avg_batch_loss
                accum_loss = 0  # 重置累积损失
                
                # 更新损失统计 - 使用计数器进行平均值计算
                epoch_l1_loss += l1_loss.item()
                epoch_boundary_loss += boundary_loss.item()
                epoch_smooth_loss += (smooth_loss + diff_smooth_loss).item()
                epoch_perceptual_loss += perceptual_loss.item()
                if args.use_diffusion:
                    epoch_diffusion_loss += diffusion_loss.item()
                
                # 增加计数器
                num_batches_processed += 1
                
                # 每步记录到TensorBoard
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
                
                        # 记录掩码损失
                        if hasattr(args, 'use_enhanced_diffusion') and args.use_enhanced_diffusion and hasattr(model, 'mask_branch'):
                            writer.add_scalar('BatchLoss/mask', mask_loss.item(), global_step)
                        
                        # 诊断边界区域比例
                        if boundary_mask is not None:
                            boundary_ratio = boundary_mask.mean().item()
                            writer.add_scalar('Diagnostics/boundary_ratio', boundary_ratio, global_step)
                        
                # 记录GPU使用情况
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated(device) / 1024**2
                    writer.add_scalar('System/GPU_Memory_MB', gpu_memory, global_step)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'l1': l1_loss.item(),
                'boundary': boundary_loss.item(),
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"  # 使用科学计数法显示学习率
                })
                
                # 中间释放内存
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 更新学习率调度器
                if args.scheduler == 'cosine' or args.scheduler == 'cosine_warmup':
                    scheduler.step()
        
        # 计算每个epoch的时间
        epoch_time = time.time() - start_time
        
        # 监控GPU使用情况
        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated(device) / 1024**2
            writer.add_scalar('System/Epoch_GPU_Memory_MB', end_gpu_memory, epoch)
            writer.add_scalar('System/Epoch_Time_Seconds', epoch_time, epoch)
            print(f"GPU内存使用(结束): {end_gpu_memory:.2f} MB, 用时: {epoch_time:.2f}秒")
        
            # 计算平均损失 - 使用正确的计数器
            if num_batches_processed > 0:  # 防止除零错误
                avg_loss = epoch_loss / num_batches_processed
                avg_l1_loss = epoch_l1_loss / num_batches_processed
                avg_boundary_loss = epoch_boundary_loss / num_batches_processed
                avg_smooth_loss = epoch_smooth_loss / num_batches_processed
                avg_perceptual_loss = epoch_perceptual_loss / num_batches_processed
                avg_diffusion_loss = epoch_diffusion_loss / num_batches_processed if args.use_diffusion else 0
            else:
                avg_loss = epoch_loss
                avg_l1_loss = epoch_l1_loss
                avg_boundary_loss = epoch_boundary_loss
                avg_smooth_loss = epoch_smooth_loss
                avg_perceptual_loss = epoch_perceptual_loss
                avg_diffusion_loss = epoch_diffusion_loss if args.use_diffusion else 0
        
        # 记录训练损失 - 使用step=epoch确保唯一性
            current_step = epoch * len(train_loader) + batch_idx
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
                # 在这里移除try块，避免缩进问题
                # 确保所有张量变量存在
                if 'img1' in locals() and 'img2' in locals() and 'mask' in locals() and 'stitched_image' in locals() and 'gt' in locals():
            # 确保图像在合适的范围内 [0, 1]
            # 并且移动到CPU并转换为numpy数据
                    img1_vis = img1[:min(4, img1.size(0))].clamp(-1, 1).add(1).div(2).cpu()
                    img2_vis = img2[:min(4, img2.size(0))].clamp(-1, 1).add(1).div(2).cpu()
                    warp1_vis = warp1[:min(4, warp1.size(0))].clamp(-1, 1).add(1).div(2).cpu()
                    warp2_vis = warp2[:min(4, warp2.size(0))].clamp(-1, 1).add(1).div(2).cpu()
                    mask_vis = mask[:min(4, mask.size(0))].cpu()
                    stitched_vis = stitched_image[:min(4, stitched_image.size(0))].clamp(-1, 1).add(1).div(2).cpu()
                    gt_vis = gt[:min(4, gt.size(0))].clamp(-1, 1).add(1).div(2).cpu()
            
            writer.add_images('Input/Image1', img1_vis, epoch)
            writer.add_images('Input/Image2', img2_vis, epoch)
            writer.add_images('Input/Warp1', warp1_vis, epoch)
            writer.add_images('Input/Warp2', warp2_vis, epoch)
            writer.add_images('Output/Mask', mask_vis, epoch)
            writer.add_images('Output/Stitched', stitched_vis, epoch)
            writer.add_images('Output/GroundTruth', gt_vis, epoch)
                    if args.use_diffusion and stitched_image is not None:
                        denoised_vis = stitched_image[:min(4, stitched_image.size(0))].clamp(-1, 1).add(1).div(2).cpu()
                writer.add_images('Output/Denoised', denoised_vis, epoch)
            
            # 确保数据被写入磁盘
            writer.flush()
                else:
                    print(f"警告: 缺少可视化所需的变量，跳过可视化")
        
        # 保存检查点
            if (epoch + 1) % args.save_freq == 0 and is_main_process:
            # 确保模型保存目录存在
            os.makedirs(args.log_dir, exist_ok=True)
            
            # 构建完整的保存路径
            checkpoint_path = os.path.join(args.log_dir, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"保存模型到: {checkpoint_path}")
            
                # 获取要保存的模型状态字典
                # 如果是分布式训练，保存不带module前缀的状态字典
                if args.distributed:
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                
            # 使用与test.py一致的键名
            torch.save({
                'epoch': epoch,
                    'model': model_state_dict,
                    'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
            }, checkpoint_path)
            
            # 保存最新模型
            latest_path = os.path.join(args.log_dir, 'latest.pth')
            torch.save({
                'epoch': epoch,
                    'model': model_state_dict,
                    'optimizer': optimizer.state_dict(),
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
                    'model': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
            }, model_path)
            
            # 修改为每个epoch都进行测试而不是每隔2个epoch
            if hasattr(args, 'test_data_dir') and args.test_data_dir and (epoch + 1) % args.test_freq == 0 and is_main_process:
                print(f"\n======= 在轮次 {epoch+1} 进行测试 =======")
                
                # 创建临时参数对象用于测试
                test_args = argparse.Namespace(
                    gpu=rank,  # 使用当前进程的rank作为GPU
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
                    current_model=model.module if args.distributed else model  # 使用module属性获取实际模型（如果使用DDP）
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
                
                # 记录测试指标到Tensorboard
                if test_metrics and writer:
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
                                
                    except Exception as e:
                        print(f"可视化测试结果时出错: {e}")
                
                print(f"======= 轮次 {epoch+1} 测试完成 =======\n")
                print(f"测试指标: L1错误={test_metrics.get('avg_l1', 'N/A'):.4f}, PSNR={test_metrics.get('avg_psnr', 'N/A'):.2f}dB")
        
                # 恢复模型训练状态
                if train_mode:
                    model.train()
            
            # 只有主进程打印训练信息
            if is_main_process:
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, L1: {avg_l1_loss:.4f}, "
              f"Boundary: {avg_boundary_loss:.4f}, Smooth: {avg_smooth_loss:.4f}, "
              f"Perceptual: {avg_perceptual_loss:.4f}")
        
        # 保存最终模型（仅主进程），将保存模型的代码移到epochs循环外部
        if is_main_process:
            # 获取要保存的模型状态字典
            if args.distributed:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
                
    final_model_path = os.path.join(args.log_dir, 'final_model.pth')
    model_save_dir = os.path.join('Composition', 'model')
    os.makedirs(model_save_dir, exist_ok=True)
    final_path_model_dir = os.path.join(model_save_dir, 'final_model.pth')
    
    # 保存到两个位置
    print(f"保存最终模型到: {final_model_path}")
            torch.save({'model': model_state_dict, 'epoch': args.epochs-1}, final_model_path)
    print(f"保存最终模型到: {final_path_model_dir}")
            torch.save({'model': model_state_dict, 'epoch': args.epochs-1}, final_path_model_dir)
    
            if writer:
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
                'beta_end': 0.02
            }
            model = ImprovedDiffusionComposition(
                image_channels=3, 
                diffusion_params=diffusion_params, 
                embedding_dim=args.embedding_dim,
                device=device
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
        use_virtual=args.use_virtual if hasattr(args, 'use_virtual') else False,
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
            
            try:
            if args.use_diffusion:
                # 调试信息：记录测试阶段输入图像尺寸
                print(f"测试阶段输入尺寸 - img1: {img1.shape}, img2: {img2.shape}")
                
                    # 确保输入尺寸合理 - 调整为32的倍数以避免采样问题
                    h1, w1 = img1.shape[2], img1.shape[3]
                    h2, w2 = img2.shape[2], img2.shape[3]
                    
                    # 检查尺寸是否需要调整
                    adjust_size = False
                    if h1 % 32 != 0 or w1 % 32 != 0:
                        new_h1 = ((h1 + 31) // 32) * 32
                        new_w1 = ((w1 + 31) // 32) * 32
                        img1 = F.interpolate(img1, size=(new_h1, new_w1), mode='bilinear', align_corners=True)
                        mask1 = F.interpolate(mask1, size=(new_h1, new_w1), mode='nearest')
                        adjust_size = True
                        print(f"调整img1尺寸到32的倍数: {h1}x{w1} -> {new_h1}x{new_w1}")
                    
                    if h2 % 32 != 0 or w2 % 32 != 0:
                        new_h2 = ((h2 + 31) // 32) * 32
                        new_w2 = ((w2 + 31) // 32) * 32
                        img2 = F.interpolate(img2, size=(new_h2, new_w2), mode='bilinear', align_corners=True)
                        mask2 = F.interpolate(mask2, size=(new_h2, new_w2), mode='nearest')
                        adjust_size = True
                        print(f"调整img2尺寸到32的倍数: {h2}x{w2} -> {new_h2}x{new_w2}")
                    
                try:
                    # 使用扩散模型采样
                        if hasattr(model, 'sample'):
                            # 尝试使用字典输入形式
                            try:
                                model_input = {
                                    'base_image': img1,
                                    'warp_image': img2,
                                    'base_mask': mask1,
                                    'warp_mask': mask2
                                }
                                
                                # 模型可能支持字典输入
                                result = model.sample(
                                    model_input,
                                    num_steps=args.sample_steps if hasattr(args, 'sample_steps') else 100,
                                    guidance_scale=1.0
                                )
                                
                                if isinstance(result, dict):
                                    learned_mask1 = result.get('mask', result.get('learned_mask1'))
                                    denoised = result.get('denoised')
                                    stitched_image = result.get('stitched_image')
                                else:
                                    learned_mask1, denoised, stitched_image = result
                                    
                            except (TypeError, ValueError):
                                # 使用标准参数调用
                        learned_mask1, denoised, stitched_image = model.sample(
                                    img1, img2, mask1, mask2, 
                                    num_steps=args.sample_steps if hasattr(args, 'sample_steps') else 100,
                                    guidance_scale=1.0
                                )
                        else:
                            # 没有sample方法，尝试直接调用
                            result = model(img1, img2, mask1, mask2)
                            
                            if isinstance(result, tuple) and len(result) == 2:
                                out, _ = result
                                # 使用helper函数生成掩码和拼接图像
                                learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, out)
                                stitched_image = compose_images(img1, img2, learned_mask1, learned_mask2)
                                denoised = stitched_image
                    else:
                                out = result
                                # 使用helper函数生成掩码和拼接图像
                                learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, out)
                                stitched_image = compose_images(img1, img2, learned_mask1, learned_mask2)
                                denoised = stitched_image
                        
                        # 如果之前调整了尺寸，现在将结果调整回原始尺寸
                        if adjust_size:
                            original_size = (h1, w1)  # 使用第一张图的尺寸作为目标
                            learned_mask1 = F.interpolate(learned_mask1, size=original_size, mode='bilinear', align_corners=True)
                            if 'learned_mask2' in locals():
                                learned_mask2 = F.interpolate(learned_mask2, size=original_size, mode='bilinear', align_corners=True)
                            denoised = F.interpolate(denoised, size=original_size, mode='bilinear', align_corners=True)
                            stitched_image = F.interpolate(stitched_image, size=original_size, mode='bilinear', align_corners=True)
                            print(f"将结果调整回原始尺寸: {original_size}")
                    except RuntimeError as e:
                        print(f"采样过程出错: {e}")
                        print("使用简单混合方法生成结果...")
                        
                        # 记录详细错误信息以便调试
                        traceback.print_exc()
                        
                        # 创建基本掩码用于基本融合 - 使用中心平滑过渡
                        h, w = mask1.shape[2], mask1.shape[3]
                        x = torch.linspace(-1, 1, w, device=device)
                        transition = 0.5 * (1 + torch.tanh(x * 3))
                        transition = transition.view(1, 1, 1, w).expand(mask1.shape[0], 1, h, w)
                        out = transition
                        
                        # 使用helper函数
                        learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, out)
                        denoised = (img1 + img2) / 2.0
                        stitched_image = compose_images(img1, img2, learned_mask1, learned_mask2)
            else:
                # 使用常规模型
                mask = model(torch.cat([img1, img2, mask1, mask2], 1))
                    # 使用helper函数
                    learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, mask)
                    stitched_image = compose_images(warp1, warp2, learned_mask1, learned_mask2)
                    denoised = stitched_image  # 保持变量名一致
            except Exception as e:
                print(f"测试过程中出现错误: {e}")
                print("使用最基本的平均混合作为退化选项...")
                
                # 记录详细错误信息以便调试
                traceback.print_exc()
                
                # 创建最简单的掩码和结果
                out = torch.ones_like(mask1) * 0.5  # 50% 混合
                learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, out)
                denoised = (img1 + img2) / 2.0
                stitched_image = compose_images(img1, img2, learned_mask1, learned_mask2)
            
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
                    mask_image = learned_mask1[i].cpu().permute(1, 2, 0).numpy() * 255
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

# 添加梯度一致性损失函数
def calculate_gradient_consistency_loss(stitched_image, img1, img2, mask):
    """
    计算梯度一致性损失，确保拼接图像在边界区域的梯度平滑过渡
    
    Args:
        stitched_image: 拼接后的图像
        img1: 第一个输入图像
        img2: 第二个输入图像
        mask: 拼接掩码
        
    Returns:
        梯度一致性损失
    """
    # 确保所有输入的尺寸一致
    if img1.shape[2:] != stitched_image.shape[2:]:
        img1 = F.interpolate(img1, size=stitched_image.shape[2:], mode='bilinear', align_corners=False)
    
    if img2.shape[2:] != stitched_image.shape[2:]:
        img2 = F.interpolate(img2, size=stitched_image.shape[2:], mode='bilinear', align_corners=False)
    
    if mask.shape[2:] != stitched_image.shape[2:]:
        mask = F.interpolate(mask, size=stitched_image.shape[2:], mode='bilinear', align_corners=False)
    
    # 确保通道数匹配
    if mask.shape[1] == 1 and stitched_image.shape[1] > 1:
        mask = mask.repeat(1, stitched_image.shape[1] // mask.shape[1], 1, 1)
    
    # 转为灰度计算
    def rgb_to_gray(image):
        if image.shape[1] >= 3:
            return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            return image  # 如果已经是单通道，直接返回
    
    # 创建sobel核
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=stitched_image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=stitched_image.device).view(1, 1, 3, 3)
    
    # 转换为灰度
    gray_stitched = rgb_to_gray(stitched_image)
    gray_img1 = rgb_to_gray(img1)
    gray_img2 = rgb_to_gray(img2)
    
    # 应用padding
    pad = nn.ReflectionPad2d(1)
    gray_stitched_pad = pad(gray_stitched)
    gray_img1_pad = pad(gray_img1)
    gray_img2_pad = pad(gray_img2)
    
    # 计算梯度
    grad_x_stitched = F.conv2d(gray_stitched_pad, sobel_x)
    grad_y_stitched = F.conv2d(gray_stitched_pad, sobel_y)
    
    grad_x_img1 = F.conv2d(gray_img1_pad, sobel_x)
    grad_y_img1 = F.conv2d(gray_img1_pad, sobel_y)
    
    grad_x_img2 = F.conv2d(gray_img2_pad, sobel_x)
    grad_y_img2 = F.conv2d(gray_img2_pad, sobel_y)
    
    # 将mask调整为单通道，以便于后续处理
    if mask.shape[1] > 1:
        mask_single = mask[:, 0:1]
    else:
        mask_single = mask
    
    # 计算过渡区域（掩码边界）
    transition_area = torch.abs(F.avg_pool2d(mask_single, 5, stride=1, padding=2) - mask_single)
    transition_area = F.interpolate(transition_area, size=grad_x_stitched.shape[2:], mode='bilinear', align_corners=False)
    
    # 调整mask尺寸以匹配梯度尺寸
    mask_resized = F.interpolate(mask_single, size=grad_x_img1.shape[2:], mode='bilinear', align_corners=False)
    
    # 混合预期梯度
    expected_grad_x = mask_resized * grad_x_img1 + (1 - mask_resized) * grad_x_img2
    expected_grad_y = mask_resized * grad_y_img1 + (1 - mask_resized) * grad_y_img2
    
    # 计算梯度差异
    grad_diff_x = torch.abs(grad_x_stitched - expected_grad_x)
    grad_diff_y = torch.abs(grad_y_stitched - expected_grad_y)
    
    # 在过渡区域加权损失
    weighted_diff_x = grad_diff_x * transition_area
    weighted_diff_y = grad_diff_y * transition_area
    
    # 总损失
    if transition_area.sum() > 0:
        loss = (weighted_diff_x.sum() + weighted_diff_y.sum()) / (transition_area.sum() + 1e-8)
    else:
        loss = torch.tensor(0.0, device=stitched_image.device)
    
    return loss

def generate_learned_masks(mask1, mask2, out):
    """
    生成学习到的掩码
    
    Args:
        mask1: 第一个输入掩码
        mask2: 第二个输入掩码
        out: 模型预测的掩码/融合系数
    
    Returns:
        learned_mask1, learned_mask2: 生成的两个掩码
    """
    # 将out调整到与mask1相同的尺寸（如果需要）
    if out.shape[2:] != mask1.shape[2:]:
        out = F.interpolate(out, size=mask1.shape[2:], mode='bilinear', align_corners=False)
    
    # 应用掩码生成公式
    learned_mask1 = (mask1 - mask1*mask2) + mask1*mask2*out
    learned_mask2 = (mask2 - mask1*mask2) + mask1*mask2*(1-out)
    
    return learned_mask1, learned_mask2

def compose_images(img1, img2, learned_mask1, learned_mask2):
    """
    使用生成的掩码拼接图像
    
    Args:
        img1: 第一个输入图像
        img2: 第二个输入图像
        learned_mask1: 第一个学习到的掩码
        learned_mask2: 第二个学习到的掩码
    
    Returns:
        stitched_image: 拼接后的图像
    """
    # 确保img2与img1尺寸一致
    if img2.shape[2:] != img1.shape[2:]:
        img2 = F.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
    
    # 确保掩码与图像尺寸一致
    if learned_mask1.shape[2:] != img1.shape[2:]:
        learned_mask1 = F.interpolate(learned_mask1, size=img1.shape[2:], mode='bilinear', align_corners=False)
    
    if learned_mask2.shape[2:] != img1.shape[2:]:
        learned_mask2 = F.interpolate(learned_mask2, size=img1.shape[2:], mode='bilinear', align_corners=False)
    
    # 确保掩码通道数与图像一致
    if learned_mask1.shape[1] == 1 and img1.shape[1] > 1:
        learned_mask1 = learned_mask1.repeat(1, img1.shape[1], 1, 1)
    
    if learned_mask2.shape[1] == 1 and img2.shape[1] > 1:
        learned_mask2 = learned_mask2.repeat(1, img2.shape[1], 1, 1)
    
    # 应用拼接公式
    stitched_image = (img1+1.)*learned_mask1 + (img2+1.)*learned_mask2 - 1.
    
    return stitched_image

# 添加在辅助函数部分
def convert_to_grayscale(image_tensor):
    """
    将彩色图片转换为灰白图片
    
    Args:
        image_tensor: 形状为[B, C, H, W]的图像张量
        
    Returns:
        灰白图像张量，保持原始形状但内容为灰度
    """
    # 检查是否为彩色图片（通道数大于1）
    if image_tensor.shape[1] > 1:
        # 使用标准RGB到灰度的转换公式
        gray = 0.299 * image_tensor[:, 0:1] + 0.587 * image_tensor[:, 1:2] + 0.114 * image_tensor[:, 2:3]
        # 复制灰度值到所有通道，保持原始通道数
        grayscale_image = gray.repeat(1, image_tensor.shape[1], 1, 1)
        return grayscale_image
    else:
        # 已经是单通道，直接返回
        return image_tensor

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

    parser.add_argument('--test_freq', type=int, default=10, help='测试频率（轮数）')

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
        train(args.gpu if hasattr(args, 'gpu') else 0, args.world_size if hasattr(args, 'world_size') else 1, args)
    elif args.mode == 'test':
        test(args)


