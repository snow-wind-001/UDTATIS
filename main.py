import os
import json
import argparse
import subprocess
from pathlib import Path
import sys
import torch
import warnings
from datetime import datetime
import torch.nn as nn
import importlib.util
import torch.nn.functional as F
# Handle different scikit-image versions
try:
    from skimage.metrics import structural_similarity as compare_ssim
except ImportError:
    try:
        from skimage.measure import structural_similarity as compare_ssim
    except ImportError:
        from skimage.measure import compare_ssim
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import signal
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import random
from PIL import Image

# 添加辅助函数
def ensure_dense_tensor(tensor):
    """
    确保张量是密集张量，如果是稀疏张量则转换为密集张量
    
    Args:
        tensor: 要处理的张量
        
    Returns:
        密集张量
    """
    if tensor is None:
        return None
        
    if isinstance(tensor, torch.Tensor):
        if hasattr(tensor, 'is_sparse') and tensor.is_sparse:
            return tensor.to_dense()
    
    return tensor
    
def safe_mask_to_tensor(mask_array, device=None):
    """
    安全地将掩码数组转换为张量，处理各种可能的维度和稀疏问题
    
    Args:
        mask_array: 要转换的掩码数组(numpy数组或torch张量)
        device: 目标设备(可选)
        
    Returns:
        转换后的掩码张量
    """
    try:
        # 如果已经是张量
        if isinstance(mask_array, torch.Tensor):
            # 检查是否为稀疏张量并处理
            if hasattr(mask_array, 'is_sparse') and mask_array.is_sparse:
                mask_array = mask_array.to_dense()
            
            # 移动到指定设备
            if device is not None and str(mask_array.device) != str(device):
                mask_array = mask_array.to(device)
            
            return mask_array
        
        # NumPy数组处理
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
            
        # 移动到指定设备
        if device is not None:
            mask_tensor = mask_tensor.to(device)
            
        return mask_tensor
    except Exception as e:
        print(f"掩码转换错误: {e}")
        # 返回一个安全的默认值
        default_tensor = torch.ones(1, 1, 1)
        if device is not None:
            default_tensor = default_tensor.to(device)
        return default_tensor

# 添加路径 - 修改添加顺序，确保Warp路径优先级高于Composition
# 这样在导入时，Python会先从Warp/Codes中查找模块，避免名称冲突
sys.path.append('Warp/Codes')
sys.path.append('Composition/Codes')

# 注意：最好使用完整路径进行导入，例如：
# from Warp.Codes.network import Network
# from Composition.Codes.network import ImprovedDiffusionComposition

# Add import for enhanced diffusion model
from Composition.Codes.enhanced_network import EnhancedDiffusionComposition, ImprovedDiffusionComposition

def load_pretrained_weights(model, pretrained_path, device=None):
    """
    通用函数，用于加载预训练权重到模型。
    支持多种权重格式，优先使用最新的checkpoint。
    
    Args:
        model: 要加载权重的模型
        pretrained_path: 权重文件路径或目录
        device: 加载到的设备
        
    Returns:
        start_epoch: 开始训练的轮次
        model: 加载权重后的模型
    """
    start_epoch = 0
    
    if not pretrained_path or not os.path.exists(pretrained_path):
        print(f"没有找到预训练权重路径 {pretrained_path}，从头开始训练")
        return start_epoch, model
    
    if os.path.isdir(pretrained_path):
        # 如果是目录，查找最新的checkpoint
        ckpt_list = glob.glob(os.path.join(pretrained_path, "*.pth"))
        if not ckpt_list:
            print(f"目录 {pretrained_path} 中没有找到权重文件，从头开始训练")
            return start_epoch, model
        
        # 按修改时间排序
        ckpt_list.sort(key=os.path.getmtime, reverse=True)
        checkpoint_path = ckpt_list[0]
        print(f"找到最新的权重文件: {checkpoint_path}")
    else:
        # 直接使用指定的文件
        checkpoint_path = pretrained_path
    
    try:
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 处理不同格式的权重文件
        if isinstance(checkpoint, dict):
            # 支持多种键名格式
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                print(f"使用'model'键加载权重成功")
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"使用'model_state_dict'键加载权重成功")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"使用'state_dict'键加载权重成功")
            else:
                # 尝试直接作为state_dict加载
                model.load_state_dict(checkpoint)
                print(f"直接加载权重成功")
            
            # 获取开始轮次
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"从轮次 {start_epoch} 开始续训")
        else:
            # 如果不是字典，尝试直接加载
            model.load_state_dict(checkpoint)
            print(f"直接加载权重成功")
        
        print(f"成功加载预训练权重: {checkpoint_path}")
    except Exception as e:
        print(f"加载权重时出错: {e}")
        print(f"从头开始训练")
        start_epoch = 0
    
    return start_epoch, model

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def create_directories(config):
    # Create directories for Warp
    Path(config['warp']['train']['model_save_path']).mkdir(parents=True, exist_ok=True)
    Path(config['warp']['train']['summary_path']).mkdir(parents=True, exist_ok=True)
    Path(config['warp']['test']['result_path']).mkdir(parents=True, exist_ok=True)
    
    # Create directories for Composition
    Path(config['composition']['train']['model_save_path']).mkdir(parents=True, exist_ok=True)
    Path(config['composition']['train']['summary_path']).mkdir(parents=True, exist_ok=True)
    Path(config['composition']['test']['result_path']).mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for Composition test results
    for dir_name in config['composition']['test']['save_dirs'].values():
        Path(os.path.join(config['composition']['test']['result_path'], dir_name)).mkdir(parents=True, exist_ok=True)

def train_warp(args, config):
    """训练Warp模块"""
    print("==== Training Warp Module ====")
    
    # 判断是否使用分布式训练
    is_distributed = hasattr(args, 'distributed') and args.distributed
    
    # 导入必要模块
    from Warp.Codes.train import train, train_distributed
    
    # 获取配置参数
    train_params = config['warp']['train']
    
    # 设置GPU设备 - 兼容分布式训练
    if is_distributed:
        # 对于分布式训练，local_rank由启动器提供
        if hasattr(args, 'local_rank') and args.local_rank != -1:
            local_rank = args.local_rank
            # 不设置环境变量，让分布式模块自己处理
            use_gpu = True
            world_size = args.world_size if hasattr(args, 'world_size') else torch.cuda.device_count()
            print(f"使用分布式训练: local_rank={local_rank}, world_size={world_size}")
        else:
            print("警告: distributed参数已设置但local_rank未定义, 回退到单GPU训练")
            is_distributed = False
            os.environ['CUDA_VISIBLE_DEVICES'] = train_params['gpu']
            use_gpu = train_params['gpu'] != '-1' and torch.cuda.is_available()
    else:
        # 单GPU训练
    os.environ['CUDA_VISIBLE_DEVICES'] = train_params['gpu']
    use_gpu = train_params['gpu'] != '-1' and torch.cuda.is_available()
    
    # 使用虚拟数据集 (如果配置中指定)
    use_virtual = args.virtual if hasattr(args, 'virtual') else False
    
    # 创建参数对象
    class Args:
        pass
    
    train_args = Args()
    train_args.batch_size = train_params['batch_size']
    train_args.max_epoch = train_params['max_epoch']
    train_args.learning_rate = train_params['learning_rate']
    train_args.train_path = train_params['train_path']
    train_args.model_save_path = train_params['model_save_path']
    train_args.summary_path = train_params['summary_path']
    
    # 设置预训练权重路径
    # 优先使用命令行参数中的pretrained路径
    # 如果没有提供，则检查模型保存目录中是否有已有权重
    if hasattr(args, 'pretrained') and args.pretrained:
        train_args.pretrained_path = args.pretrained
    elif os.path.exists(train_args.model_save_path) and os.listdir(train_args.model_save_path):
        # 如果模型保存目录存在且非空，使用该目录作为预训练权重目录
        train_args.pretrained_path = train_args.model_save_path
        print(f"未指定预训练权重路径，将使用模型保存目录作为预训练权重: {train_args.pretrained_path}")
    else:
        train_args.pretrained_path = None
        print("未找到预训练权重，将从头开始训练")
    
    train_args.homography_weight = train_params['loss_weights']['homography']
    train_args.mesh_weight = train_params['loss_weights']['mesh']
    train_args.feature_weight = train_params['loss_weights']['feature']
    train_args.valid_point_weight = train_params['loss_weights']['valid_point']
    train_args.continuity_weight = train_params['loss_weights']['continuity']
    
    # 添加分布式训练所需参数
    train_args.distributed = is_distributed
    if is_distributed:
        train_args.local_rank = local_rank
        train_args.world_size = world_size
        train_args.sync_bn = args.sync_bn if hasattr(args, 'sync_bn') else False
    
    # 添加性能优化参数
    train_args.num_workers = args.num_workers if hasattr(args, 'num_workers') else 4
    train_args.use_amp = args.use_amp if hasattr(args, 'use_amp') else True
    train_args.clip_grad = args.clip_grad if hasattr(args, 'clip_grad') else 0.5
    train_args.grad_accum_steps = args.grad_accum_steps if hasattr(args, 'grad_accum_steps') else 1
    
    # 确保保存目录存在
    os.makedirs(train_args.model_save_path, exist_ok=True)
    os.makedirs(train_args.summary_path, exist_ok=True)
    
    # 打印主要训练参数（只在主进程上）
    if not is_distributed or (is_distributed and local_rank == 0):
    print(f"Starting Warp training with parameters:")
    for attr in dir(train_args):
        if not attr.startswith('__') and not callable(getattr(train_args, attr)):
            print(f"  {attr}: {getattr(train_args, attr)}")
    
    # 记录训练开始时间
    start_time = datetime.now()
    if not is_distributed or (is_distributed and local_rank == 0):
    print(f"Training started at: {start_time}")
    
    # 执行训练 - 根据是否分布式选择不同函数
    if is_distributed:
        train_distributed(local_rank, world_size, train_args)
    else:
    train(train_args)
    
    # 记录训练结束时间
    end_time = datetime.now()
    if not is_distributed or (is_distributed and local_rank == 0):
    print(f"Training completed at: {end_time}")
    print(f"Total training time: {end_time - start_time}")
    
    print("==== Warp Training Completed ====")

def test_warp(args, config):
    """测试Warp模块"""
    print("==== Testing Warp Module ====")
    
    # 导入必要模块
    from Warp.Codes.test import test
    
    # 获取配置参数
    test_params = config['warp']['test']
    
    # 设置GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = test_params['gpu']
    use_gpu = test_params['gpu'] != '-1' and torch.cuda.is_available()
    
    # 使用虚拟数据集 (如果配置中指定)
    use_virtual = args.virtual if hasattr(args, 'virtual') else False
    
    # 准备测试参数
    test_args = {
        'batch_size': test_params['batch_size'],
        'test_path': test_params['test_path'],
        'result_path': test_params['result_path'],
        'model_path': args.model_path if hasattr(args, 'model_path') else None,
        'use_virtual': use_virtual,
        'use_gpu': use_gpu
    }
    
    # 确保保存目录存在
    os.makedirs(test_args['result_path'], exist_ok=True)
    
    # 启动测试
    print(f"Starting Warp testing with parameters:")
    for k, v in test_args.items():
        print(f"  {k}: {v}")
    
    # 记录测试开始时间
    start_time = datetime.now()
    print(f"Testing started at: {start_time}")
    
    # 创建参数对象
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # 将字典转换为对象属性
    test_args_obj = Args(**test_args)
    
    # 执行测试
    test(test_args_obj)
    
    # 记录测试结束时间
    end_time = datetime.now()
    print(f"Testing completed at: {end_time}")
    print(f"Total testing time: {end_time - start_time}")
    
    print("==== Warp Testing Completed ====")

def train_Composition(config, debug_mode=False):
    """
    训练Composition模块
    """
    # 导入必要的库
    import os
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR
    import signal
    import sys
    import argparse
    
    # 解析训练参数
    # 从外部函数传入的参数
    # 初始化train_args变量，确保在使用前已定义
    train_args = debug_mode
     
    # 从配置文件读取默认参数
    train_params = config['composition']['train']
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 获取扩散模型参数
    diffusion_params = train_params.get('diffusion', {
        'num_timesteps': 1000,  # 修改为1000步，与保存的模型匹配
        'beta_start': 1e-4,
        'beta_end': 0.01  # 降低最大噪声水平
    })
    
    # 获取设备信息
    gpu = train_params.get('gpu', '0')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 构建训练参数
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    summary_path = os.path.join(train_params['summary_path'], f'run_{timestamp}')
    
    # 使用命令行参数的类来创建训练参数对象
    if not isinstance(train_args, argparse.Namespace):
    train_args = argparse.Namespace(
        mode='train',
        data_dir=train_params['train_path'],
        test_data_dir=config['composition']['test']['test_path'],  # 添加测试数据路径
        log_dir=summary_path,
        img_size=256,
        norm_type='imagenet',
        model_type='unet',
        pretrain=False,
        use_diffusion=True,
            diffusion_steps=diffusion_params.get('num_timesteps', 1000),  # 使用1000步扩散
            sample_steps=100,  # 增加采样步数以获得更平滑的结果
        embedding_dim=128,
        batch_size=train_params['batch_size'],
        epochs=train_params['max_epoch'],
            # 进一步降低初始学习率以提高稳定性
            lr=5e-5,  # 使用1e-4的学习率
        weight_decay=1e-5,
        num_workers=4,
        save_freq=1,  # 每轮都保存一次模型
        vis_freq=5,    # 每5轮可视化一次
            test_freq=10,  # 每10轮测试一次
            use_amp=False,  # 禁用自动混合精度训练以提高稳定性
            clip_grad=0.5, # 设置更严格的梯度裁剪阈值，提高训练稳定性
            scheduler='cosine_warmup',  # 使用带预热的余弦调度
            # 调整损失权重
        l1_weight=1.0,
            boundary_weight=train_params['loss_weights'].get('boundary', 0.5),  # 降低边界损失权重
            smooth_weight=train_params['loss_weights'].get('smooth', 0.8),
            perceptual_weight=train_params['loss_weights'].get('perceptual', 0.4),
        ssim_weight=0.1,
        color_weight=0.1,
            diffusion_weight=train_params['loss_weights'].get('diffusion', 0.1),  # 降低扩散损失权重
            warm_up_epochs=5,  # 减少预热轮数，从20降到5
        exclude_boundary=False,
        gpu=int(gpu) if gpu.isdigit() else 0,
            grad_accum_steps=4,  # 增加梯度累积步数以提高稳定性
            test_during_training=True,  # 默认在训练期间进行测试
        resume=None,  # 默认不从检查点恢复
            # 添加NaN检测参数
            detect_anomaly=True,  # 启用异常检测以捕获NaN和Inf
            freeze_layers=True,  # 启用选择性冻结层功能以提高初期训练稳定性
    )
    
    # 如果debug_mode是一个包含命令行参数的对象，则更新train_args
    elif debug_mode and isinstance(debug_mode, argparse.Namespace):
        # 只更新已存在的属性
        for attr in vars(debug_mode):
            if hasattr(train_args, attr):
                setattr(train_args, attr, getattr(debug_mode, attr))
    
    def ensure_outputs_on_device(outputs, target_device):
        """确保输出字典中的所有张量都在正确的设备上"""
        if not isinstance(outputs, dict):
            return outputs
        
        for key in outputs:
            if isinstance(outputs[key], torch.Tensor) and outputs[key].device != target_device:
                outputs[key] = outputs[key].to(target_device)
        return outputs
    
    # 启用PyTorch的异常检测以捕获NaN
    if hasattr(train_args, 'detect_anomaly') and train_args.detect_anomaly:
        print("启用PyTorch异常检测以捕获NaN和无穷值")
        torch.set_anomaly_enabled(True)
    
    # 设置训练数据集和数据加载器
    from Composition.Codes.dataset import TrainDataset
    train_dataset = TrainDataset(
        train_args.data_dir,
        image_size=train_args.img_size,
        augment=True,
        norm_type=train_args.norm_type,
        overlap_based_stitching=getattr(train_args, 'overlap_based_stitching', False)  # 添加这一行
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 创建TensorBoard摘要写入器
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=train_args.log_dir)
    
    # 训练循环
    print(f"开始训练，从epoch {train_args.epochs}到{train_args.epochs}")
    best_loss = float('inf')
    # 创建模型实例
    print("Creating model...")
    print(f"Using device: {device}")
    
    # 获取diffusion参数
    diffusion_steps = diffusion_params.get("num_timesteps", 1000)
    beta_start = diffusion_params.get("beta_start", 1e-4)
    beta_end = diffusion_params.get("beta_end", 0.01)
    
    # 使用ImprovedDiffusionComposition类
    from Composition.Codes.enhanced_network import ImprovedDiffusionComposition
    
    net = ImprovedDiffusionComposition(
        num_timesteps=diffusion_steps,
        beta_schedule="linear",
        image_size=256,
        base_channels=64,
        attention_resolutions=[16, 8],
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_res_blocks=2,
        heads=4,
        use_scale_shift_norm=True
    ).to(device)
    
    # 使用优化器
    optimizer = optim.Adam(
        net.parameters(), 
        lr=train_args.lr,
        weight_decay=train_args.weight_decay
    )
    
    # 学习率调度器
    # 预训练权重路径
    pretrained_path = None
    if debug_mode and hasattr(debug_mode, "pretrained"):
        pretrained_path = debug_mode.pretrained
    elif "pretrained_path" in train_params:
        pretrained_path = train_params["pretrained_path"]
    
    # 如果有预训练权重，加载权重
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        if "model_state_dict" in checkpoint:
            net.load_state_dict(checkpoint["model_state_dict"])
            print("成功加载模型权重")
        elif "model" in checkpoint:
            net.load_state_dict(checkpoint["model"])
            print("成功加载模型权重")
        
        # 如果有epoch信息，更新start_epoch
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"从epoch {start_epoch}开始训练")
    else:
        print("未找到预训练权重，从头开始训练")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # 梯度缩放器
    scaler = torch.amp.GradScaler(enabled=train_args.use_amp)
    
    # 设置起始轮次
    start_epoch = 0
    
    # 梯度裁剪阈值
    grad_clip_threshold = train_args.clip_grad
    print(f"设置梯度裁剪阈值: {grad_clip_threshold}")
    
    # 初始化失败计数
    epoch_failures = 0
    max_failures = 3  # 最大允许连续失败次数
    
    # 初始化连续批次失败计数
    consecutive_batch_failures = 0
    max_consecutive_batch_failures = 50  # 最大连续批次失败次数
    
    # 初始化运行损失和平均损失，确保它们在任何情况下都有定义
    running_loss = 0.0
    avg_loss = float('inf')  # 默认设置为无穷大，这样只有成功完成至少一个epoch才会更新
    
    # 使用梯度累积进行训练
    for epoch in range(start_epoch, train_args.epochs):
        try:
            net.train()
            running_loss = 0.0
            epoch_start_time = time.time()
            optimizer.zero_grad()  # 开始新一轮训练前清零梯度
            
            # 检查是否需要解冻网络层
            if hasattr(train_args, 'unfreeze_hook'):
                train_args.unfreeze_hook(epoch)
            
            for batch_idx, batch_data in enumerate(train_loader):
                try:
                    # 解包数据
                    base_image = batch_data['base_image'].to(device)
                    warp_image = batch_data['warp_image'].to(device)
                    base_mask = batch_data['base_mask'].to(device)
                    warp_mask = batch_data['warp_mask'].to(device)
                    
                    # 检查输入数据是否包含NaN
                    if torch.isnan(base_image).any() or torch.isnan(warp_image).any():
                        print(f"警告: 批次 {batch_idx} 输入数据包含NaN，跳过")
                        consecutive_batch_failures += 1
                        if consecutive_batch_failures > max_consecutive_batch_failures:
                            print(f"连续批次失败次数超过{max_consecutive_batch_failures}，训练中断")
                            return
                        continue
                    
                    # 初始噪声图像 - 混合基准图像和待拼接图像
                    x = (base_image + warp_image) / 2.0
                    
                    # 随机生成时间步
                    t = torch.randint(0, train_args.diffusion_steps, (base_image.size(0),), device=device).long()
                    
                    # 记录时间步分布（每个epoch记录一次）
                    if batch_idx == 0:
                        writer.add_histogram('train/time_steps', t, epoch)
                    
                    # 使用混合精度进行计算
                    with torch.amp.autocast(device_type=device.type, enabled=train_args.use_amp):
                        # 确保所有张量都是密集的，不是稀疏的，防止permute操作失败
                        x = ensure_dense_tensor(x)
                        t = ensure_dense_tensor(t)
                        base_image = ensure_dense_tensor(base_image)
                        warp_image = ensure_dense_tensor(warp_image)
                        base_mask = ensure_dense_tensor(base_mask)
                        warp_mask = ensure_dense_tensor(warp_mask)
                        
                        # 确保没有NaN或Inf值
                        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                        base_image = torch.nan_to_num(base_image, nan=0.0, posinf=1.0, neginf=-1.0)
                        warp_image = torch.nan_to_num(warp_image, nan=0.0, posinf=1.0, neginf=-1.0)
                        base_mask = torch.nan_to_num(base_mask, nan=0.5, posinf=1.0, neginf=0.0)
                        warp_mask = torch.nan_to_num(warp_mask, nan=0.5, posinf=1.0, neginf=0.0)
                        
                        # 确保所有空间维度一致
                        target_shape = x.shape[2:]
                        if base_image.shape[2:] != target_shape:
                            base_image = F.interpolate(base_image, size=target_shape, mode='bilinear', align_corners=False)
                        if warp_image.shape[2:] != target_shape:
                            warp_image = F.interpolate(warp_image, size=target_shape, mode='bilinear', align_corners=False)
                        if base_mask.shape[2:] != target_shape:
                            base_mask = F.interpolate(base_mask, size=target_shape, mode='bilinear', align_corners=False)
                        if warp_mask.shape[2:] != target_shape:
                            warp_mask = F.interpolate(warp_mask, size=target_shape, mode='bilinear', align_corners=False)
                        
                        # 计算损失
                        loss = net.compute_loss(x, t, base_image, warp_image, base_mask, warp_mask)
                        
                        # 检查损失是否为NaN或Inf
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"警告：在epoch {epoch+1}，batch {batch_idx+1}中检测到NaN或Inf损失。跳过此批次。")
                            if torch.isnan(loss):
                                print("损失为NaN")
                            else:
                                print("损失为Inf")
                            consecutive_batch_failures += 1
                            if consecutive_batch_failures > max_consecutive_batch_failures:
                                print(f"连续批次失败次数超过{max_consecutive_batch_failures}，训练中断")
                                # 保存当前模型状态，方便调试
                                torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': net.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                                    'scaler': scaler.state_dict(),
                                }, os.path.join(train_args.log_dir, 'error_model.pth'))
                                return
                            continue
                        # 重置连续失败计数器
                        consecutive_batch_failures = 0
                        
                        # 额外记录损失组件 - 增强版，获取更详细的损失分解
                        try:
                            # 使用自定义函数获取更详细的损失组件
                            with torch.no_grad():
                                # 获取详细的损失组件
                                loss_details = net.get_loss_components(x, t, base_image, warp_image, base_mask, warp_mask)
                                
                                # 记录每个批次的损失分解（每10个批次记录一次）
                                global_step = epoch * len(train_loader) + batch_idx
                                if batch_idx % 10 == 0:  # 每10个批次记录一次，避免TensorBoard文件过大
                                    # 记录总损失
                                    writer.add_scalar('BatchLoss/Total', loss.item(), global_step)
                                    
                                    # 记录各损失组件，增加对元组类型的检查
                                    for loss_name, loss_value in loss_details.items():
                                        # 检查损失值是否为元组，如果是则取第一个元素
                                        if isinstance(loss_value, tuple):
                                            loss_value = loss_value[0]  # 取元组的第一个元素作为损失值
                                        
                                        # 确保损失值是张量，并检查是否为NaN或无穷大
                                        if isinstance(loss_value, torch.Tensor) and not torch.isnan(loss_value) and not torch.isinf(loss_value):
                                            writer.add_scalar(f'LossComponents/{loss_name}', loss_value.item(), global_step)
                                        elif isinstance(loss_value, (int, float)) and not math.isnan(loss_value) and not math.isinf(loss_value):
                                            writer.add_scalar(f'LossComponents/{loss_name}', loss_value, global_step)
                                    
                                    # 额外记录学习率
                                    writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
                                    
                                    # 记录示例的输入图像和掩码（每100个批次记录一次）
                                    if batch_idx % 100 == 0:
                                        # 将张量转换为0-1范围用于可视化
                                        writer.add_images('Train/BaseImage', (base_image + 1) / 2, global_step)
                                        writer.add_images('Train/WarpImage', (warp_image + 1) / 2, global_step)
                                        writer.add_images('Train/BaseMask', base_mask, global_step)
                                        writer.add_images('Train/WarpMask', warp_mask, global_step)
        except Exception as e:
                            print(f"记录批次损失组件时出错: {e}")
                        
                        # 应用梯度累积
                        loss = loss / train_args.grad_accum_steps
                    
                    # 反向传播
                    scaler.scale(loss).backward()
                    
                    # 累积此批次的原始损失（乘以累积系数以获取正确的平均值）
                    # 注意：在计算原始损失时我们需要乘以train_args.grad_accum_steps来抵消之前的除法
                    true_loss = loss.item() * train_args.grad_accum_steps
                    running_loss += true_loss  # 只在这里累加一次损失
                    
                    # 如果是梯度累积的最后一步或者批次的最后一个样本
                    if (batch_idx + 1) % train_args.grad_accum_steps == 0 or batch_idx == len(train_loader) - 1:
                        # 在取消缩放之前检查是否有无效梯度
                        valid_gradients = True
                        
                        # 取消缩放梯度以进行梯度裁剪和检查
                        scaler.unscale_(optimizer)
                        
                        # 检查梯度有效性
                        for param in net.parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    valid_gradients = False
                                    print(f"警告: 在epoch {epoch+1}，batch {batch_idx+1}中检测到NaN或Inf梯度。跳过此梯度更新。")
                                    break
                        
                        if valid_gradients:
                            # 梯度裁剪
                            if train_args.clip_grad > 0:
                                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), train_args.clip_grad)
                                # 记录梯度范数
                                if batch_idx % 10 == 0:  # 每10个批次记录一次
                                    writer.add_scalar('Gradients/norm', grad_norm.item(), epoch * len(train_loader) + batch_idx)
                            
                            # 更新权重
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # 跳过此批次的更新
                            scaler.update()
                        
                        # 清零梯度
                        optimizer.zero_grad(set_to_none=True)  # set_to_none=True 可以更高效地清零梯度
                    
                    # 打印批次信息
                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch+1}/{train_args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                              f"Loss: {loss.item() * train_args.grad_accum_steps:.6f}, "
                              f"LR: {optimizer.param_groups[0]['lr']:.2e}")  # 使用科学计数法显示学习率
                
                except Exception as batch_e:
                    print(f"处理批次 {batch_idx} 时出现错误: {batch_e}")
                    import traceback
                    traceback.print_exc()
                    consecutive_batch_failures += 1
                    if consecutive_batch_failures > max_consecutive_batch_failures:
                        print(f"连续批次失败次数超过{max_consecutive_batch_failures}，训练中断")
                        return
                    continue
            
            # 计算平均损失 - 确保不会除以零
            if len(train_loader) > 0:
                avg_loss = running_loss / len(train_loader)
            else:
                avg_loss = running_loss  # 防止除以零
            epoch_time = time.time() - epoch_start_time
            
            # 更新学习率
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()
            
            # 打印轮次信息
            print(f"Epoch {epoch+1}/{train_args.epochs} completed in {epoch_time:.2f}s, "
                  f"Avg Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")  # 使用科学计数法显示学习率
            
            # 记录到TensorBoard - 更详细的损失分解
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            # 额外计算并记录平均噪声损失和掩码损失
            try:
                with torch.no_grad():
                    # 从验证集或训练集获取一小批数据用于计算损失组件
                    eval_data = next(iter(train_loader))
                    base_image_eval = eval_data['base_image'][:4].to(device)  # 仅使用4个样本
                    warp_image_eval = eval_data['warp_image'][:4].to(device)
                    base_mask_eval = eval_data['base_mask'][:4].to(device)
                    warp_mask_eval = eval_data['warp_mask'][:4].to(device)
                    
                    # 混合基准图像和待拼接图像
                    x_eval = (base_image_eval + warp_image_eval) / 2.0
                    
                    # 随机生成时间步
                    t_eval = torch.randint(0, train_args.diffusion_steps, (base_image_eval.size(0),), device=device).long()
                    
                    # 添加噪声
                    noisy_x_eval, target_noise_eval = net.forward_diffusion(x_eval, t_eval)
                    
                    # 使用前向函数预测噪声和掩码
                    predicted_noise_eval, learned_mask_eval = net.forward(noisy_x_eval, t_eval, base_image_eval, warp_image_eval, base_mask_eval, warp_mask_eval)
                    
                    # 计算各种损失
                    noise_loss_eval = F.mse_loss(predicted_noise_eval, target_noise_eval)
                    
                    # 确保维度匹配
                    combined_mask_eval = (base_mask_eval + warp_mask_eval) / 2.0
                    if learned_mask_eval.shape[2:] != combined_mask_eval.shape[2:]:
                        combined_mask_eval = F.interpolate(combined_mask_eval, size=learned_mask_eval.shape[2:], mode='bilinear', align_corners=False)
                    
                    mask_loss_eval = F.binary_cross_entropy_with_logits(learned_mask_eval, combined_mask_eval)
                    
                    # 记录详细损失
                    writer.add_scalar('Loss/noise', noise_loss_eval.item(), epoch)
                    writer.add_scalar('Loss/mask', mask_loss_eval.item(), epoch)
                    
                    # 计算SNR (Signal-to-Noise Ratio)
                    alphas_cumprod = net.alphas_cumprod.gather(-1, t_eval).reshape(-1, 1, 1, 1)
                    snr = alphas_cumprod / (1 - alphas_cumprod)
                    writer.add_scalar('Metrics/SNR', snr.mean().item(), epoch)
                    
                    # 记录权重和梯度统计
                    for name, param in net.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            writer.add_histogram(f'Weights/{name}', param.data, epoch)
                            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    except Exception as e:
                print(f"记录详细损失组件时出错: {e}")
        import traceback
        traceback.print_exc()
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"发现新的最佳模型，保存到 {train_args.log_dir}/best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                    'scaler': scaler.state_dict(),
                    'loss': best_loss,
                }, os.path.join(train_args.log_dir, 'best_model.pth'))
            
            # 定期保存检查点
            if (epoch + 1) % train_args.save_freq == 0:
                save_path = os.path.join(train_args.log_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                    'scaler': scaler.state_dict(),
                    'loss': avg_loss,
                }, save_path)
                print(f"已保存周期检查点: {save_path}")
                
                # 同时保存到model目录（与main.py中的train_Composition保持一致）
                model_save_path = os.path.join(train_params['model_save_path'], f'model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model': net.state_dict(),  # 保持与原始代码兼容的键名
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                    'scaler': scaler.state_dict(),
                    'loss': avg_loss,
                }, model_save_path)
                print(f"已保存到模型目录: {model_save_path}")
            
            # 测试阶段
            if (epoch + 1) % train_args.test_freq == 0 and train_args.test_during_training:
                print("\n进行测试...")
                net.eval()
                
                # 随机选择一个训练样本进行可视化
                with torch.no_grad():
                    # 获取第一个批次数据
                    vis_data = next(iter(train_loader))
                    base_image = vis_data['base_image'][0:1].to(device)
                    warp_image = vis_data['warp_image'][0:1].to(device)
                    base_mask = vis_data['base_mask'][0:1].to(device)
                    warp_mask = vis_data['warp_mask'][0:1].to(device)
                    
                    # 进行合成
                    mask, result = net.forward_composition(base_image, warp_image, base_mask, warp_mask)
                    
                    # 添加图像到TensorBoard - 增强版可视化
                    writer.add_images('Image/Base', (base_image + 1) / 2, epoch)
                    writer.add_images('Image/Warp', (warp_image + 1) / 2, epoch)
                    writer.add_images('Mask/Base', base_mask, epoch)
                    writer.add_images('Mask/Warp', warp_mask, epoch)
                    writer.add_images('Mask/Generated', mask, epoch)
                    writer.add_images('Result/Composite', (result + 1) / 2, epoch)
                    
                    # 生成并可视化扩散过程中间结果 - 每10个epoch记录一次完整扩散过程
                    if (epoch + 1) % 10 == 0:
                        try:
                            # 生成扩散过程的中间结果
                            steps_to_record = [0, 25, 50, 75, 99]  # 仅记录几个关键步骤
                            
                            # 从纯噪声开始
                            x_sample = torch.randn((1, 3, base_image.shape[2], base_image.shape[3]), device=device)
                            
                            # 记录每个关键时间步的结果
                            for step_idx, i in enumerate(reversed(range(0, net.num_timesteps, net.num_timesteps // 100))):
                                if step_idx in steps_to_record:
                                    t_sample = torch.full((1,), i, device=device, dtype=torch.long)
                                    with torch.no_grad():
                                        x_sample, clean_x, step_mask = net.p_sample(x_sample, t_sample, base_image, warp_image, base_mask, warp_mask)
                                        
                                        # 记录这个时间步的结果
                                        writer.add_images(f'Diffusion/Step_{step_idx}_Noise', (x_sample + 1) / 2, epoch)
                                        writer.add_images(f'Diffusion/Step_{step_idx}_Clean', (clean_x + 1) / 2, epoch)
                                        writer.add_images(f'Diffusion/Step_{step_idx}_Mask', step_mask, epoch)
                                        
                            # 记录最终合成结果
                            final_mask, clean_final, final_result = net.sample(base_image, warp_image, base_mask, warp_mask, num_steps=20)
                            writer.add_images('Diffusion/Final_Mask', final_mask, epoch)
                            writer.add_images('Diffusion/Final_Result', (final_result + 1) / 2, epoch)
                        except Exception as e:
                            print(f"记录扩散过程可视化时出错: {e}")
                    
                    # 记录注意力图可视化
                    try:
                        # 提取一个注意力块用于可视化
                        # 这里简化处理，仅记录全局注意力的一个头的注意力图
                        # 实际实现可能需要根据网络结构调整
                        if hasattr(net, 'global_attn'):
                            with torch.no_grad():
                                # 合并输入以供特征提取
                                combined = torch.cat([base_image, warp_image, base_mask, warp_mask], dim=1)
                                x_input = net.channel_adapter(combined)
                                
                                # 随机时间步
                                t_viz = torch.zeros(base_image.shape[0], device=device, dtype=torch.long)
                                t_emb = net.time_mlp(t_viz)
                                
                                # 特征提取到全局注意力之前
                                d1 = net.down1(x_input, t_emb)
                                d2 = net.down2(d1, t_emb)
                                d3 = net.down3(d2, t_emb)
                                d4 = net.down4(d3, t_emb)
                                
                                # 中间块处理
                                h = net.mid_block1(d4, t_emb)
                                h = net.mid_attn(h)
                                h = net.mid_block2(h, t_emb)
                                
                                # 提取全局注意力的注意力图 (简化版)
                                # 注意：此处简化处理，实际可能需要修改网络结构来提取注意力图
                                b, c, height, width = h.shape
                                
                                # 创建注意力热图 - 这里是简化示例，真实实现可能更复杂
                                # 使用特征图的平均值作为热图
                                attention_map = h.mean(dim=1, keepdim=True)
                                
                                # 归一化注意力图以便可视化
                                attention_map = F.interpolate(attention_map, size=(64, 64), mode='bilinear', align_corners=False)
                                min_val = attention_map.min()
                                max_val = attention_map.max()
                                if max_val > min_val:
                                    attention_map = (attention_map - min_val) / (max_val - min_val)
                                
                                # 记录注意力图
                                writer.add_images('Attention/GlobalMap', attention_map, epoch)
                    except Exception as e:
                        print(f"记录注意力可视化时出错: {e}")
                
                print("测试完成\n")

            # 重置失败计数（成功完成epoch）
            epoch_failures = 0
            
        except Exception as epoch_e:
            print(f"处理epoch {epoch} 时出现错误: {epoch_e}")
            import traceback
            traceback.print_exc()
            epoch_failures += 1
            if epoch_failures > max_failures:
                print(f"Epoch失败次数超过{max_failures}，训练中断")
                return
            continue
    
    # 保存最终模型
    final_save_path = os.path.join(train_args.log_dir, 'final_model.pth')
    
    torch.save({
        'epoch': train_args.epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'scaler': scaler.state_dict(),
        'loss': avg_loss,
    }, final_save_path)
    print(f"已保存最终模型: {final_save_path}")
    
    # 同时保存到model目录
    model_save_path = os.path.join(train_params['model_save_path'], 'model_final.pth')
    torch.save({
        'epoch': train_args.epochs,
        'model': net.state_dict(),  # 保持与原始代码兼容的键名
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'scaler': scaler.state_dict(),
        'loss': avg_loss,
    }, model_save_path)
    print(f"已保存到模型目录: {model_save_path}")
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 禁用异常检测
    if hasattr(train_args, 'detect_anomaly') and train_args.detect_anomaly:
        torch.set_anomaly_enabled(False)
    
    print("训练完成!")
    return net

def test_composition(args, config):
    """
    测试Composition模块 - 改进版，实际加载并使用模型进行测试
    
    Args:
        args: 命令行参数
        config: 配置字典
    """
    print("==== Testing Composition Module ====")
    
    # 设置GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = config['composition']['test']['gpu']
    use_gpu = config['composition']['test']['gpu'] != '-1' and torch.cuda.is_available()
    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    
    # 检查数据路径
    data_path = config['composition']['test']['test_path']
    if not os.path.exists(data_path):
        print(f"Warning: Composition test data path not found at {data_path}")
        print("Preparing composition test data...")
        config = prepare_composition_data(config, args, mode='test')
        data_path = config['composition']['test']['test_path']
    
    # 验证数据路径下的目录结构
    required_dirs = ['warp1', 'warp2', 'mask1', 'mask2']
    missing_dirs = [d for d in required_dirs if not os.path.exists(os.path.join(data_path, d))]
    if missing_dirs:
        print(f"Warning: Missing required directories in test data path: {missing_dirs}")
        print("Re-preparing composition test data...")
        config = prepare_composition_data(config, args, mode='test')
        data_path = config['composition']['test']['test_path']
    
    # 创建结果目录
    result_path = config['composition']['test']['result_path']
    os.makedirs(result_path, exist_ok=True)
    
    # 创建子目录
    save_dirs = {}
    for dir_name, sub_dir in config['composition']['test']['save_dirs'].items():
        full_path = os.path.join(result_path, sub_dir)
        os.makedirs(full_path, exist_ok=True)
        save_dirs[dir_name] = full_path
        
    # 创建合并图像目录
    merged_dir = os.path.join(result_path, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    
    # 确定模型路径
    model_path = None
    if hasattr(args, 'model_path') and args.model_path:
        model_path = args.model_path
    else:
        # 尝试寻找最新的模型文件
        model_dir = config['composition']['train']['model_save_path']
        if os.path.exists(model_dir):
            checkpoint_files = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
            if checkpoint_files:
                model_path = checkpoint_files[-1]  # 使用最新的checkpoint
                print(f"Using latest model checkpoint: {model_path}")
    
    # 确定交互模式和阈值设置
    interactive_mode = False
    threshold = 127
    if hasattr(args, 'interactive') and args.interactive:
        interactive_mode = True
    if hasattr(args, 'threshold') and args.threshold is not None:
        threshold = args.threshold
    
    image_size = 512  # 默认输入图像的处理分辨率
    if hasattr(args, 'image_size') and args.image_size:
        image_size = args.image_size
    
    # 打印测试参数
    print(f"Starting Composition testing with parameters:")
    print(f"  test_path: {data_path}")
    print(f"  device: {device}")
    print(f"  image_size: {image_size}")
    print(f"  threshold: {threshold}")
    print(f"  interactive_mode: {interactive_mode}")
    
    if not model_path or not os.path.exists(model_path):
        print("  No valid model found, using simplified test implementation")
        return simplified_test_composition(data_path, save_dirs)
    
        print(f"  model_path: {model_path}")
    
    # 创建模型实例
    try:
        # 导入必要模块
        from Composition.Codes.enhanced_network import ImprovedDiffusionComposition
        import torchvision.transforms as transforms
        from PIL import Image
        import numpy as np
        from tqdm import tqdm
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        raise

        # 获取diffusion参数
        diffusion_params = config['composition']['train'].get('diffusion', {
            'num_timesteps': 1000,  # 修改为1000，与保存的模型参数匹配
            'beta_start': 1e-4,
            'beta_end': 0.01
        })
        diffusion_steps = diffusion_params.get('num_timesteps', 1000)  # 默认值改为1000
        
        # 创建模型实例
        print("Creating model...")
        net = ImprovedDiffusionComposition(
            num_timesteps=1000,  # 明确设置为1000，与保存的模型参数匹配
            beta_schedule='linear',
            image_size=image_size,
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
        print(f"Loading weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 处理checkpoint格式
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            net.load_state_dict(checkpoint['model'])
    else:
            net.load_state_dict(checkpoint)
        
        print("Model loaded successfully")
        net.eval()  # 设置为评估模式
        
        # 定义预处理函数
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 收集测试图像
        warp1_path = os.path.join(data_path, 'warp1')
        warp2_path = os.path.join(data_path, 'warp2')
        mask1_path = os.path.join(data_path, 'mask1')
        mask2_path = os.path.join(data_path, 'mask2')
        
        warp1_files = sorted(glob.glob(os.path.join(warp1_path, '*.*')))
        warp2_files = sorted(glob.glob(os.path.join(warp2_path, '*.*')))
        mask1_files = sorted(glob.glob(os.path.join(mask1_path, '*.*')))
        mask2_files = sorted(glob.glob(os.path.join(mask2_path, '*.*')))
        
        # 处理测试限制
        max_test_images = min(len(warp1_files), len(warp2_files), len(mask1_files), len(mask2_files))
        limit = config['composition']['test'].get('limit_test_images', 50)
        if limit > 0 and limit < max_test_images:
            max_test_images = limit
            print(f"Limiting test to {max_test_images} images")
        else:
            print(f"Processing all {max_test_images} images")
            
        # 确保所有保存目录都存在
        # 添加mask目录到save_dirs如果不存在
        if 'mask' not in save_dirs:
            mask_dir = os.path.join(result_path, 'mask')
            os.makedirs(mask_dir, exist_ok=True)
            save_dirs['mask'] = mask_dir
            
        # 打印实际使用的目录路径
        print(f"结果保存路径: {result_path}")
        print(f"子目录:")
        for dir_name, path in save_dirs.items():
            print(f"  - {dir_name}: {path}")
        print(f"合并图像目录: {merged_dir}")
        
        if max_test_images == 0:
            print("No test images found")
            return config
        
        print(f"Processing {max_test_images} test images...")
        
        # 追踪处理时间和成功率
        total_time = 0
        success_count = 0
        results = []
        
        # 如果是交互模式，显示使用说明
        if interactive_mode:
            print("使用说明:")
            print("  - 按回车键 (Enter): 处理下一张照片")
            print("  - 按 '+' 键: 增加阈值")
            print("  - 按 '-' 键: 减小阈值")
            print("  - 按 's' 键: 保存当前图像")
            print("  - 按 ESC 键: 退出程序")
            
        # 当前阈值
        current_threshold = threshold
        
        # 定义张量到图像的转换函数
        def tensor_to_image(tensor):
            """
            将PyTorch张量转换为NumPy图像
            """
            if tensor.dim() == 4:  # [B,C,H,W]
                tensor = tensor[0]  # 只取第一个样本
                
            # 确保是CPU张量
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
        
        with torch.no_grad():  # 禁用梯度计算
            for idx in tqdm(range(max_test_images)):
                try:
                    # 读取图像和掩码
                    warp1_file = warp1_files[idx]
                    warp2_file = warp2_files[idx]
                    mask1_file = mask1_files[idx]
                    mask2_file = mask2_files[idx]
                    
                    # 获取基础文件名用于保存
                    file_name = os.path.basename(warp1_file)
                    name_without_ext = os.path.splitext(file_name)[0]
                    
                    print(f"\n处理图像对 {idx+1}/{max_test_images}: {name_without_ext}")
                    
                    # 读取并预处理图像
                    warp1_img = Image.open(warp1_file).convert('RGB')
                    warp2_img = Image.open(warp2_file).convert('RGB')
                    
                    # 确保掩码是单通道
                    mask1_img = Image.open(mask1_file).convert('L')
                    mask2_img = Image.open(mask2_file).convert('L')
                    
                    # 记录原始尺寸
                    original_size1 = warp1_img.size
                    original_size2 = warp2_img.size
                    print(f"  原始图像尺寸: warp1={original_size1}, warp2={original_size2}")
                    
                    # 转换为张量并添加批次维度
                    warp1_tensor = preprocess(warp1_img.resize((image_size, image_size), Image.LANCZOS)).unsqueeze(0).to(device)
                    warp2_tensor = preprocess(warp2_img.resize((image_size, image_size), Image.LANCZOS)).unsqueeze(0).to(device)
                    
                    # 处理掩码
                    mask1_tensor = mask_preprocess(mask1_img.resize((image_size, image_size), Image.LANCZOS)).unsqueeze(0).to(device)
                    mask2_tensor = mask_preprocess(mask2_img.resize((image_size, image_size), Image.LANCZOS)).unsqueeze(0).to(device)
                    
                    # 使用模型进行处理
                    start_time = time.time()
                    sample_steps = 100
                    if hasattr(args, 'sample_steps') and args.sample_steps:
                        sample_steps = args.sample_steps
                    print(f"使用 {sample_steps} 步扩散采样生成mask")
                    
                    # 进行合成
                    try:
                        # 使用safe_forward方法处理图像对
                        output_mask, output_result = net.safe_forward(
                            warp1=warp1_tensor,
                            warp2=warp2_tensor,
                            mask1=mask1_tensor,
                            mask2=mask2_tensor
                        )
                        
                        # 计算处理时间
                        process_time = time.time() - start_time
                        total_time += process_time
                        
                        # 转换模型输出为OpenCV图像
                        mask_img = tensor_to_image(output_mask)
                        mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
                        
                        # 转换原始图像为OpenCV格式
                        warp1_img_cv = cv2.cvtColor(np.array(warp1_img), cv2.COLOR_RGB2BGR)
                        warp2_img_cv = cv2.cvtColor(np.array(warp2_img), cv2.COLOR_RGB2BGR)
                        mask1_img_np = np.array(mask1_img)
                        
                        # 保存网络生成的mask图像
                        mask_save_path = os.path.join(save_dirs['mask'], f"{name_without_ext}_mask.png")
                        cv2.imwrite(mask_save_path, mask_img_gray)
                        
                        # 处理当前图像
                        if interactive_mode:
                            current_threshold = threshold
                            while True:
                                # 使用当前阈值处理图像
                                warp1_display, warp2_display, merged_img, mask_binary = process_mask_for_composition(
                                    mask_img_gray, warp1_img_cv, warp2_img_cv, mask1_img_np, current_threshold
                                )
                                
                                # 创建输入图像的并排展示
                                # 调整warp1和warp2，使它们具有相同的高度
                                h1, w1 = warp1_display.shape[:2]
                                h2, w2 = warp2_display.shape[:2]
                                
                                # 使用较大的高度
                                max_height = max(h1, h2)
                                # 等比例缩放
                                if h1 != max_height:
                                    scale = max_height / h1
                                    new_width = int(w1 * scale)
                                    warp1_resized_display = cv2.resize(warp1_display, (new_width, max_height), interpolation=cv2.INTER_LANCZOS4)
                                else:
                                    warp1_resized_display = warp1_display
                                    
                                if h2 != max_height:
                                    scale = max_height / h2
                                    new_width = int(w2 * scale)
                                    warp2_resized_display = cv2.resize(warp2_display, (new_width, max_height), interpolation=cv2.INTER_LANCZOS4)
                                else:
                                    warp2_resized_display = warp2_display
                                
                                # 创建水平拼接图像
                                inputs_img = np.hstack((warp1_resized_display, warp2_resized_display))
                                
                                # 显示两个独立窗口
                                cv2.namedWindow("Input Images", cv2.WINDOW_NORMAL)
                                cv2.imshow("Input Images", inputs_img)
                                cv2.resizeWindow("Input Images", min(1200, inputs_img.shape[1]), min(800, inputs_img.shape[0]))
                                
                                cv2.namedWindow("Merged Result", cv2.WINDOW_NORMAL)
                                cv2.imshow("Merged Result", merged_img)
                                cv2.resizeWindow("Merged Result", min(1200, merged_img.shape[1]), min(800, merged_img.shape[0]))
                                
                                # 在结果窗口上显示阈值信息
                                info_img = merged_img.copy()
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(info_img, f"Threshold: {current_threshold}", (20, 30), font, 1, (0, 255, 255), 2)
                                cv2.imshow("Merged Result", info_img)
                                
                                # 等待键盘输入
                                key = cv2.waitKey(0) & 0xFF
                                
                                if key == 27:  # ESC键
                                    cv2.destroyAllWindows()
                                    print("User terminated testing")
                                    return config  # 完全退出程序
                                elif key == 13 or key == 10:  # 回车键
                                    # 保存当前结果，然后继续到下一张图片
                                    output_path = os.path.join(merged_dir, f"{name_without_ext}_merged_thresh{current_threshold}.png")
                                    cv2.imwrite(output_path, merged_img)
                                    
                                    mask_output_path = os.path.join(save_dirs['mask'], f"{name_without_ext}_mask_binary.png")
                                    cv2.imwrite(mask_output_path, mask_binary)
                                    
                                    print(f"  已保存合并图像到: {output_path}")
                                    print(f"  已保存二值化mask到: {mask_output_path}")
                                    
                                    threshold = current_threshold  # 更新默认阈值
                                    cv2.destroyAllWindows()
                                    break  # 退出当前图片的循环，继续下一张
                                elif key == ord('+') or key == ord('='):  # '+'键
                                    # 增加阈值
                                    current_threshold = min(255, current_threshold + 10)
                                    print(f"  阈值增加到: {current_threshold}")
                                elif key == ord('-') or key == ord('_'):  # '-'键
                                    # 减小阈值
                                    current_threshold = max(0, current_threshold - 10)
                                    print(f"  阈值减小到: {current_threshold}")
                                elif key == ord('s'):  # 's'键
                                    # 保存当前结果但不继续到下一张
                                    output_path = os.path.join(merged_dir, f"{name_without_ext}_merged_thresh{current_threshold}.png")
                                    cv2.imwrite(output_path, merged_img)
                                    
                                    mask_output_path = os.path.join(save_dirs['mask'], f"{name_without_ext}_mask_binary.png")
                                    cv2.imwrite(mask_output_path, mask_binary)
                                    
                                    print(f"  已保存合并图像到: {output_path}")
                                    print(f"  已保存二值化mask到: {mask_output_path}")
                        else:
                            # 非交互模式 - 直接处理并保存
                            # 处理图像合成
                            try:
                                warp1_display, warp2_display, merged_img, mask_binary = process_mask_for_composition(
                                    mask_img_gray, warp1_img_cv, warp2_img_cv, mask1_img_np, threshold
                                )
                                
                                # 保存结果
                                merged_output_path = os.path.join(merged_dir, f"{name_without_ext}_merged.png")
                                print(f"正在保存合并图像到: {merged_output_path}")
                                success = cv2.imwrite(merged_output_path, merged_img)
                                print(f"保存状态: {'成功' if success else '失败'}")
                                
                                # 确保mask目录存在
                                mask_dir = save_dirs.get('mask', os.path.join(result_path, 'mask'))
                                os.makedirs(mask_dir, exist_ok=True)
                                
                                mask_binary_path = os.path.join(mask_dir, f"{name_without_ext}_mask_binary.png")
                                success = cv2.imwrite(mask_binary_path, mask_binary)
                                
                                print(f"  已保存合并结果: {merged_output_path}")
                                print(f"  已保存二值化mask: {mask_binary_path}")
                            except Exception as e:
                                print(f"合成过程出错: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # 显示图像（仅显示不等待交互）
                            # 创建输入图像的并排展示
                            h1, w1 = warp1_display.shape[:2]
                            h2, w2 = warp2_display.shape[:2]
                            
                            # 使用较大的高度
                            max_height = max(h1, h2)
                            # 等比例缩放
                            if h1 != max_height:
                                scale = max_height / h1
                                new_width = int(w1 * scale)
                                warp1_resized_display = cv2.resize(warp1_display, (new_width, max_height), interpolation=cv2.INTER_LANCZOS4)
                            else:
                                warp1_resized_display = warp1_display
                                
                            if h2 != max_height:
                                scale = max_height / h2
                                new_width = int(w2 * scale)
                                warp2_resized_display = cv2.resize(warp2_display, (new_width, max_height), interpolation=cv2.INTER_LANCZOS4)
                            else:
                                warp2_resized_display = warp2_display
                            
                            # 创建水平拼接图像
                            inputs_img = np.hstack((warp1_resized_display, warp2_resized_display))
                            
                            # 显示两个独立窗口
                            cv2.namedWindow("Input Images", cv2.WINDOW_NORMAL)
                            cv2.imshow("Input Images", inputs_img)
                            cv2.resizeWindow("Input Images", min(1200, inputs_img.shape[1]), min(800, inputs_img.shape[0]))
                            
                            cv2.namedWindow("Merged Result", cv2.WINDOW_NORMAL)
                            cv2.imshow("Merged Result", merged_img)
                            cv2.resizeWindow("Merged Result", min(1200, merged_img.shape[1]), min(800, merged_img.shape[0]))
                            
                            # 短暂显示后继续
                            cv2.waitKey(500)
                            
                        # 更新处理信息
                        success_count += 1
                        results.append({
                            'filename': name_without_ext,
                            'process_time': process_time,
                            'threshold': threshold,
                            'status': 'success'
                        })
                    except Exception as sampling_err:
                        print(f"Error during processing: {sampling_err}")
                        import traceback
                        traceback.print_exc()
                        results.append({
                            'filename': name_without_ext,
                            'status': 'error',
                            'error': str(sampling_err)
                        })
                    def tensor_to_image(tensor):
                        # 确保是CPU张量
                        tensor = tensor.cpu()
                        # 从[-1,1]转换到[0,1]
                        tensor = (tensor + 1) / 2
                        # 裁剪到有效范围
                        tensor = torch.clamp(tensor, 0, 1)
                        # 转换为NumPy并调整为[0,255]的uint8
                        np_img = (tensor.permute(0, 2, 3, 1).squeeze().numpy() * 255).astype(np.uint8)
                        return np_img
                    
                    # 检查是否为全景拼接结果(宽度大于输入)
                    is_panorama = result.shape[3] > warp1_tensor.shape[3]
                    
                    if is_panorama:
                        print(f"处理全景拼接结果，尺寸: {result.shape[2]}x{result.shape[3]}")
                        # 全景模式下，使用全景图像作为参考
                        # 创建一个全尺寸的参考图像用于评估
                        reference = torch.zeros_like(result)
                        # 放入第一张图片在左侧
                        w1 = warp1_tensor.shape[3]
                        reference[:, :, :, :w1] = warp1_tensor
                        # 放入第二张图片在右侧(如果有空间)
                        if reference.shape[3] >= w1 + warp2_tensor.shape[3]:
                            reference[:, :, :, w1:w1 + warp2_tensor.shape[3]] = warp2_tensor
                        else:
                            # 否则尝试在重叠区域右侧放置
                            overlap = max(0, warp1_tensor.shape[3] + warp2_tensor.shape[3] - reference.shape[3])
                            w2_start = w1 - overlap
                            reference[:, :, :, w2_start:] = warp2_tensor[:, :, :, :reference.shape[3] - w2_start]
                    else:
                        # 简单地平均两张输入图像作为参考
                        reference = (warp1_tensor + warp2_tensor) / 2
                    
                                            # 保存生成的mask和结果
                        mask_save_path = os.path.join(save_dirs['mask'], f"{name_without_ext}_mask.png")
                        cv2.imwrite(mask_save_path, tensor_to_image(mask))
                        
                        # 使用增强的mask处理工具
                        warp1_save_path = os.path.join(save_dirs['warp1'], f"{name_without_ext}.png")
                        warp2_save_path = os.path.join(save_dirs['warp2'], f"{name_without_ext}.png")
                        mask1_save_path = os.path.join(save_dirs['mask'], f"{name_without_ext}_mask1.png")
                        
                        # 保存输入图像供处理使用
                        cv2.imwrite(warp1_save_path, cv2.cvtColor(tensor_to_image(warp1_tensor), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(warp2_save_path, cv2.cvtColor(tensor_to_image(warp2_tensor), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(mask1_save_path, tensor_to_image(mask1_tensor))
                        
                        # 生成合并输出
                        merged_output_path = os.path.join(result_path, f"{name_without_ext}_merged.png")
                        
                        # 判断是否使用交互模式
                        interactive_mode = False
                        if hasattr(args, 'interactive') and args.interactive:
                            interactive_mode = True
                        
                        # 运行增强版处理
                        apply_composition_mask_processing(
                            mask_path=mask_save_path,
                            warp1_path=warp1_save_path,
                            warp2_path=warp2_save_path,
                            mask1_path=mask1_save_path,
                            output_path=merged_output_path,
                            threshold=threshold,
                            interactive=interactive_mode
                        )
                    
                except Exception as e:
                    print(f"Error processing image pair {idx+1}: {e}")
                    import traceback
                    traceback.print_exc()
            cv2.destroyAllWindows()
        
        # 打印统计信息
        avg_time = total_time / max(success_count, 1)
        print(f"\n处理完成!")
        print(f"成功处理: {success_count}/{max_test_images} 图像对")
        print(f"平均处理时间: {avg_time:.2f}秒")
        
        # 保存处理报告
        with open(os.path.join(result_path, "processing_report.json"), "w") as f:
            json.dump({
                'total_images': max_test_images,
                'success_count': success_count,
                'average_time': avg_time,
                'results': results
            }, f, indent=4)
    except Exception as e:
        print(f"Error setting up or running test: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to simplified testing implementation")
        return simplified_test_composition(data_path, save_dirs)

    # 检查结果是否保存成功
    print("\n==== 检查生成结果 ====")
    
    # 检查模型生成的mask文件
    mask_dir = os.path.join(result_path, 'mask')
    if os.path.exists(mask_dir):
        mask_files = glob.glob(os.path.join(mask_dir, '*_mask.png'))
        print(f"生成mask数量: {len(mask_files)}")
    else:
        print(f"未找到mask目录: {mask_dir}")
    
    # 检查合并结果文件
    if os.path.exists(merged_dir):
        merged_files = glob.glob(os.path.join(merged_dir, '*_merged.png'))
        print(f"生成合并图像数量: {len(merged_files)}")
        for file in merged_files[:3]:  # 只显示前3个文件
            print(f"  - {file}")
    else:
        print(f"未找到合并图像目录: {merged_dir}")
    
    print("==== Composition Testing Completed ====")
    return config

def process_mask_for_composition(mask_img, warp1_img, warp2_img, mask1_img, threshold=127):
    """
    处理图像并按照要求生成合成图像，保持原始分辨率
    
    参数:
        mask_img: 模型输出的mask图像 (灰度图)
        warp1_img: 第一张输入图像 (BGR格式)
        warp2_img: 第二张输入图像 (BGR格式)
        mask1_img: mask1图像 (灰度图)
        threshold: 二值化阈值
        
    返回:
        warp1_img: 原始warp1图像
        warp2_img: 原始warp2图像
        merged_img: 合成结果图像
        mask_binary: 二值化后的mask
    """
    # 获取原始尺寸
    h_mask, w_mask = mask_img.shape
    h1, w1 = warp1_img.shape[:2]
    h2, w2 = warp2_img.shape[:2]
    h_mask1, w_mask1 = mask1_img.shape
    
    # 截取mask的左侧512x512区域用于处理
    if h_mask >= 512 and w_mask >= 512:
        mask_crop = mask_img[:512, :512]
    else:
        # 如果原始mask小于512x512，则调整大小
        mask_crop = cv2.resize(mask_img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    
    # 二值化mask，使用给定的阈值
    _, mask_binary = cv2.threshold(mask_crop, threshold, 255, cv2.THRESH_BINARY)
    
    # 创建与warp2相同尺寸的mask
    mask_full = np.zeros((h2, w2), dtype=np.uint8)
    # 将512x512区域填充到左上角
    h_fill = min(512, h2)
    w_fill = min(512, w2)
    mask_full[:h_fill, :w_fill] = mask_binary[:h_fill, :w_fill]
    
    # 归一化mask到0-1范围
    mask_norm = mask_full.astype(float) / 255.0
    
    # 扩展维度以匹配颜色通道
    mask_norm_expanded = np.expand_dims(mask_norm, axis=2)
    
    # 归一化mask1到0-1范围
    mask1_norm = mask1_img.astype(float) / 255.0
    mask1_norm_expanded = np.expand_dims(mask1_norm, axis=2)
    
    # 第一部分: warp2 * mask - 保持原始分辨率
    warp2_masked = (warp2_img.astype(float) * mask_norm_expanded).astype(np.uint8)
    
    # 第二部分: warp1 * mask1 - 保持原始分辨率
    warp1_masked = (warp1_img.astype(float) * mask1_norm_expanded).astype(np.uint8)
    
    # 创建合并图像 - 使用非零区域叠加
    # 1. 首先找到warp2_masked中的非零区域
    non_zero_mask = cv2.cvtColor(warp2_masked, cv2.COLOR_BGR2GRAY) > 0
    
    # 2. 创建与warp1_masked相同尺寸的合并图像
    merged_img = warp1_masked.copy()
    
    # 3. 在非零区域应用warp2_masked
    # 确保处理区域不超出边界
    h_common = min(merged_img.shape[0], warp2_masked.shape[0])
    w_common = min(merged_img.shape[1], warp2_masked.shape[1])
    
    # 只在共同区域内进行合并
    for y in range(h_common):
        for x in range(w_common):
            if non_zero_mask[y, x]:
                merged_img[y, x] = warp2_masked[y, x]
    
    return warp1_img, warp2_img, merged_img, mask_binary

def apply_composition_mask_processing(mask_path, warp1_path, warp2_path, mask1_path, output_path, threshold=127, interactive=False):
    """
    直接处理图像合成，使用process_mask_for_composition函数
    
    参数:
        mask_path: 模型生成的掩码图像路径
        warp1_path: 第一张图像路径
        warp2_path: 第二张图像路径
        mask1_path: 第一张图像掩码路径
        output_path: 输出合并图像路径
        threshold: 二值化阈值
        interactive: 是否启用交互模式
    
    返回:
        bool: 处理是否成功
    """
    try:
        # 加载图像
        warp1_img = cv2.imread(warp1_path)
        warp2_img = cv2.imread(warp2_path)
        mask1_img = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 检查图像是否正确加载
        if warp1_img is None or warp2_img is None or mask1_img is None or mask_img is None:
            print(f"Error: 无法加载图像文件")
            print(f"warp1: {warp1_path} - 加载状态: {'成功' if warp1_img is not None else '失败'}")
            print(f"warp2: {warp2_path} - 加载状态: {'成功' if warp2_img is not None else '失败'}")
            print(f"mask1: {mask1_path} - 加载状态: {'成功' if mask1_img is not None else '失败'}")
            print(f"mask: {mask_path} - 加载状态: {'成功' if mask_img is not None else '失败'}")
            return False
        
        # 如果是交互模式，允许用户调整阈值
        current_threshold = threshold
        if interactive:
            try:
                print("==== 交互式调整阈值 ====")
                print("使用说明:")
                print("  - 按回车键 (Enter): 保存并继续")
                print("  - 按 '+' 键: 增加阈值")
                print("  - 按 '-' 键: 减小阈值")
                print("  - 按 ESC 键: 取消操作")
                
                while True:
                    # 使用当前阈值处理图像
                    _, _, merged_img, mask_binary = process_mask_for_composition(
                        mask_img, warp1_img, warp2_img, mask1_img, current_threshold
                    )
                    
                    # 创建输入图像的并排展示
                    h1, w1 = warp1_img.shape[:2]
                    h2, w2 = warp2_img.shape[:2]
                    
                    # 使用较大的高度
                    max_height = max(h1, h2)
                    # 等比例缩放
                    if h1 != max_height:
                        scale = max_height / h1
                        new_width = int(w1 * scale)
                        warp1_resized = cv2.resize(warp1_img, (new_width, max_height), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        warp1_resized = warp1_img
                        
                    if h2 != max_height:
                        scale = max_height / h2
                        new_width = int(w2 * scale)
                        warp2_resized = cv2.resize(warp2_img, (new_width, max_height), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        warp2_resized = warp2_img
                    
                    # 创建水平拼接图像
                    inputs_img = np.hstack((warp1_resized, warp2_resized))
                    
                    # 显示两个独立窗口
                    cv2.namedWindow("Input Images", cv2.WINDOW_NORMAL)
                    cv2.imshow("Input Images", inputs_img)
                    cv2.resizeWindow("Input Images", min(1200, inputs_img.shape[1]), min(800, inputs_img.shape[0]))
                    
                    cv2.namedWindow("Merged Result", cv2.WINDOW_NORMAL)
                    cv2.imshow("Merged Result", merged_img)
                    cv2.resizeWindow("Merged Result", min(1200, merged_img.shape[1]), min(800, merged_img.shape[0]))
                    
                    # 在结果窗口上显示阈值信息
                    info_img = merged_img.copy()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(info_img, f"Threshold: {current_threshold}", (20, 30), font, 1, (0, 255, 255), 2)
                    cv2.imshow("Merged Result", info_img)
                    
                    # 等待键盘输入
                    key = cv2.waitKey(0) & 0xFF
                    
                    if key == 27:  # ESC键
                        cv2.destroyAllWindows()
                        print("用户取消操作")
                        return False
                    elif key == 13 or key == 10:  # 回车键
                        # 保存当前结果并继续
                        threshold = current_threshold  # 更新实际使用的阈值
                        break
                    elif key == ord('+') or key == ord('='):  # '+'键
                        # 增加阈值
                        current_threshold = min(255, current_threshold + 10)
                        print(f"阈值增加到: {current_threshold}")
                    elif key == ord('-') or key == ord('_'):  # '-'键
                        # 减小阈值
                        current_threshold = max(0, current_threshold - 10)
                        print(f"阈值减小到: {current_threshold}")
                
                cv2.destroyAllWindows()
                
            except Exception as e:
                print(f"交互模式出错: {e}")
                print("返回非交互模式")
                # 在交互模式失败时才进行处理
                _, _, merged_img, mask_binary = process_mask_for_composition(
                    mask_img, warp1_img, warp2_img, mask1_img, threshold
                )
        else:
            # 如果非交互模式，直接处理图像
            if not interactive:
                _, _, merged_img, mask_binary = process_mask_for_composition(
                    mask_img, warp1_img, warp2_img, mask1_img, threshold
                )
        
        # 保存结果
        cv2.imwrite(output_path, merged_img)
        mask_binary_path = os.path.splitext(output_path)[0] + "_mask.png"
        cv2.imwrite(mask_binary_path, mask_binary)
        
        print(f"已保存合并图像到: {output_path}")
        print(f"已保存二值化mask到: {mask_binary_path}")
        return True
        
    except Exception as e:
        print(f"Error processing composition: {e}")
        import traceback
        traceback.print_exc()
        return False

def simplified_test_composition(data_path, save_dirs):
    """简化版的测试实现，当无法加载模型时使用"""
    print("Using simplified testing implementation")
    print("This will create sample output but not perform actual testing")
    
    # 找到一些输入图像
    warp1_path = os.path.join(data_path, 'warp1')
    warp1_files = sorted(glob.glob(os.path.join(warp1_path, '*.*')))[:5]  # 只处理5张图片
    
    if len(warp1_files) > 0:
        print(f"Generating sample results for {len(warp1_files)} images")
        
        for i, img_path in enumerate(warp1_files):
            try:
                # 读取图像
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                filename = os.path.basename(img_path)
                
                # 在各个输出目录中保存结果
                for dir_name, dir_path in save_dirs.items():
                    # 生成简单的示例输出
                    output_path = os.path.join(dir_path, filename)
                    cv2.imwrite(output_path, img)
                    
                print(f"Generated sample results for image {i+1}/{len(warp1_files)}")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    else:
        print("No images found to process")
    
    print("==== Composition Testing Completed (Simplified) ====")
    
    # 返回空字典而不是未定义的config变量
    return {}

def end_to_end_test(args, config):
    """端到端测试：先运行Warp测试生成中间结果，然后运行Composition测试"""
    print("==== Running End-to-End Test ====")
    
    # 首先运行Warp测试
    test_warp(args, config)
    
    # 使用Warp测试结果为Composition准备数据
    print("Preparing Composition test data from Warp results...")
    config = prepare_composition_data(config, args, mode='test', force_prepare=args.force_prepare)
    
    # 然后运行Composition测试
    test_composition(args, config)
    
    print("==== End-to-End Test Completed ====")

def test_train_test_functionality(config, model_path=None):
    """
    测试训练过程中的测试环节是否能正常工作
    
    Args:
        config: 配置字典
        model_path: 可选的模型路径，如果不提供则尝试找到最新的模型
    """
    import torch
    import torch.nn.functional as F
    import os
    from tqdm import tqdm
    from PIL import Image
    import numpy as np
    import torchvision.transforms as transforms
    import cv2
    
    print("==== 测试训练中的测试环节 ====")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取数据路径
    data_path = config['composition']['train']['train_path']
    if not os.path.exists(data_path):
        print(f"训练数据路径不存在: {data_path}")
        return False
    
    # 创建临时输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'train_test_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定模型路径
    if not model_path:
        model_dir = config['composition']['train']['model_save_path']
        if os.path.exists(model_dir):
            checkpoint_files = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
            if checkpoint_files:
                model_path = checkpoint_files[-1]  # 使用最新的checkpoint
                print(f"使用最新模型检查点: {model_path}")
            else:
                print("未找到模型检查点")
                return False
    
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        return False
    
    try:
        # 导入必要模块
        from Composition.Codes.enhanced_network import ImprovedDiffusionComposition
        
        # 获取diffusion参数
        diffusion_params = config['composition']['train'].get('diffusion', {
            'num_timesteps': 500, 
            'beta_start': 1e-4,
            'beta_end': 0.01
        })
        diffusion_steps = diffusion_params.get('num_timesteps', 500)
        
        # 创建模型实例
        print("创建模型...")
        net = ImprovedDiffusionComposition(
            num_timesteps=diffusion_steps,
            beta_schedule='linear',
            image_size=256,
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
        print(f"从 {model_path} 加载权重")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 处理checkpoint格式
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            net.load_state_dict(checkpoint['model'])
        else:
            net.load_state_dict(checkpoint)
        
        print("模型加载成功")
        net.eval()  # 设置为评估模式
        
        # 定义预处理函数
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 收集测试图像
        warp1_path = os.path.join(data_path, 'warp1')
        warp2_path = os.path.join(data_path, 'warp2')
        mask1_path = os.path.join(data_path, 'mask1')
        mask2_path = os.path.join(data_path, 'mask2')
        
        # 检查路径存在
        if not all(os.path.exists(p) for p in [warp1_path, warp2_path, mask1_path, mask2_path]):
            print("数据目录结构不完整，缺少必要的子目录")
            return False
        
        # 获取图像文件
        warp1_files = sorted(glob.glob(os.path.join(warp1_path, '*.*')))
        warp2_files = sorted(glob.glob(os.path.join(warp2_path, '*.*')))
        mask1_files = sorted(glob.glob(os.path.join(mask1_path, '*.*')))
        mask2_files = sorted(glob.glob(os.path.join(mask2_path, '*.*')))
        
        # 限制测试图像数量
        max_test_images = min(len(warp1_files), len(warp2_files), len(mask1_files), len(mask2_files), 5)
        
        if max_test_images == 0:
            print("未找到测试图像")
            return False
        
        print(f"处理 {max_test_images} 个测试图像...")
        
        with torch.no_grad():  # 禁用梯度计算
            for idx in range(max_test_images):
                try:
                    # 读取图像和掩码
                    warp1_file = warp1_files[idx]
                    warp2_file = warp2_files[idx]
                    mask1_file = mask1_files[idx]
                    mask2_file = mask2_files[idx]
                    
                    # 获取基础文件名用于保存
                    file_name = os.path.basename(warp1_file)
                    name_without_ext = os.path.splitext(file_name)[0]
                    
                    # 读取并预处理图像
                    warp1_img = Image.open(warp1_file).convert('RGB')
                    warp2_img = Image.open(warp2_file).convert('RGB')
                    
                    # 确保掩码是单通道
                    mask1_img = Image.open(mask1_file).convert('L')
                    mask2_img = Image.open(mask2_file).convert('L')
                    
                    # 转换为张量并添加批次维度
                    warp1_tensor = preprocess(warp1_img).unsqueeze(0).to(device)
                    warp2_tensor = preprocess(warp2_img).unsqueeze(0).to(device)
                    
                    # 处理掩码
                    mask1_tensor = transforms.ToTensor()(mask1_img).unsqueeze(0).to(device)
                    mask2_tensor = transforms.ToTensor()(mask2_img).unsqueeze(0).to(device)
                    
                    # 确保所有输入尺寸一致
                    h, w = warp1_tensor.shape[2], warp1_tensor.shape[3]
                    if warp2_tensor.shape[2:] != (h, w):
                        warp2_tensor = F.interpolate(warp2_tensor, size=(h, w), mode='bilinear', align_corners=False)
                    if mask1_tensor.shape[2:] != (h, w):
                        mask1_tensor = F.interpolate(mask1_tensor, size=(h, w), mode='bilinear', align_corners=False)
                    if mask2_tensor.shape[2:] != (h, w):
                        mask2_tensor = F.interpolate(mask2_tensor, size=(h, w), mode='bilinear', align_corners=False)
                    
                    # 测试训练中的测试功能：forward_composition
                    print(f"测试前向合成方法... (图像 {idx+1}/{max_test_images})")
                    mask, result = net.forward_composition(
                        warp1_tensor,  # 要拼接的图像
                        warp2_tensor,  # 基准图像
                        mask1_tensor,  # 要拼接图像掩码
                        mask2_tensor   # 基准图像掩码
                    )
                    
                    # 测试训练中的测试功能：sample
                    print(f"测试采样方法... (图像 {idx+1}/{max_test_images})")
                    try:
                        sample_mask, clean_output, sample_result = net.sample(
                            warp2_tensor,  # 基准图像
                            warp1_tensor,  # 要拼接的图像
                            mask2_tensor,  # 基准图像掩码
                            mask1_tensor,  # 要拼接图像掩码
                            num_steps=20   # 使用较少步数加快测试
                        )
                    except Exception as e:
                        print(f"采样方法出错: {e}")
                        print("跳过采样测试")
                        sample_mask = mask
                        sample_result = result
                    
                    # 将结果转换为图像并保存
                    def tensor_to_image(tensor):
                        tensor = tensor.cpu()
                        tensor = (tensor + 1) / 2
                        tensor = torch.clamp(tensor, 0, 1)
                        np_img = (tensor.permute(0, 2, 3, 1).squeeze().numpy() * 255).astype(np.uint8)
                        return np_img
                    
                    # 保存结果
                    result_img = tensor_to_image(result)
                    mask_img = (mask.cpu().squeeze().numpy() * 255).astype(np.uint8)
                    sample_result_img = tensor_to_image(sample_result)
                    sample_mask_img = (sample_mask.cpu().squeeze().numpy() * 255).astype(np.uint8)
                    
                    # 保存原始图像
                    warp1_img = tensor_to_image(warp1_tensor)
                    warp2_img = tensor_to_image(warp2_tensor)
                    cv2.imwrite(os.path.join(output_dir, f"{name_without_ext}_warp1.png"), cv2.cvtColor(warp1_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(output_dir, f"{name_without_ext}_warp2.png"), cv2.cvtColor(warp2_img, cv2.COLOR_RGB2BGR))
                    
                    # 保存forward_composition结果
                    cv2.imwrite(os.path.join(output_dir, f"{name_without_ext}_result.png"), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(output_dir, f"{name_without_ext}_mask.png"), mask_img)
                    
                    # 保存sample结果
                    cv2.imwrite(os.path.join(output_dir, f"{name_without_ext}_sample_result.png"), cv2.cvtColor(sample_result_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(output_dir, f"{name_without_ext}_sample_mask.png"), sample_mask_img)
                    
                except Exception as e:
                    print(f"处理图像 {idx} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"测试完成。结果已保存到 {output_dir}")
        print("==== 训练中的测试环节功能测试完成 ====")
        return True
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def prepare_composition_data(config, args, mode='train', force_prepare=False):
    """
    准备Composition阶段的训练/测试数据：应用Warp模型生成扭曲图像和掩码
    
    Args:
        config: 配置字典
        args: 命令行参数
        mode: 'train'或'test'，指定准备训练集还是测试集
        force_prepare: 是否强制重新生成数据，即使已存在
    """
    print(f"==== Preparing Composition {mode.upper()} Data ====")
    
    # 导入必要模块
    import shutil
    import tempfile
    import sys
    
    # 根据模式决定源路径和目标路径
    source_path = config['warp'][mode]['train_path' if mode == 'train' else 'test_path']
    
    # 在data/UDIS-D下创建新的目录结构来保存处理后的数据
    output_base = os.path.join('data', 'UDIS-D', 'composition_data')
    output_path = os.path.join(output_base, mode)
    
    # 创建扭曲图像和掩码的目录
    warp1_dir = os.path.join(output_path, 'warp1')
    warp2_dir = os.path.join(output_path, 'warp2')
    mask1_dir = os.path.join(output_path, 'mask1')
    mask2_dir = os.path.join(output_path, 'mask2')
    
    # 检查是否所有必需的目录都已存在
    required_dirs = [warp1_dir, warp2_dir, mask1_dir, mask2_dir]
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    # 如果不强制重新生成并且所有目录都存在，检查数据集是否完整
    if not force_prepare and not missing_dirs:
        # 检查每个目录中是否有文件
        files_count = [len(os.listdir(d)) for d in required_dirs]
        if all(count > 0 for count in files_count):
            print(f"发现完整的Composition {mode} 数据集:")
            for i, d in enumerate(required_dirs):
                print(f"  - {os.path.basename(d)}: {files_count[i]} 文件")
            print(f"跳过数据生成过程。如需重新生成，请使用--force_prepare参数")
            
            # 更新配置中的路径并返回
            if mode == 'train':
                config['composition']['train']['train_path'] = output_path
            else:  # mode == 'test'
                config['composition']['test']['test_path'] = output_path
                
            return config
    
    # 确保目录存在
    for directory in [output_base, output_path, warp1_dir, warp2_dir, mask1_dir, mask2_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")
    
    # 强制使用CPU进行处理，避免设备不一致的问题
    use_cpu = True
    
    # 明确定义设备并确保所有处理都使用同一设备
    device = torch.device('cpu') if use_cpu else torch.device('cuda' if torch.cuda.is_available() and config['warp'][mode]['gpu'] != '-1' else 'cpu')
    print(f"Using device: {device} for data preparation")
    
    # 设置设备
    device_id = config['warp'][mode]['gpu'] if not use_cpu else "-1"
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    
    # 导入Warp测试函数
    try:
        # 确定正确的网络文件并导入
        current_dir = os.path.dirname(os.path.abspath(__file__))
        warp_codes_dir = os.path.join(current_dir, 'Warp', 'Codes')
        
        # 优先检查improved_network.py
        if os.path.exists(os.path.join(warp_codes_dir, 'improved_network.py')):
            spec = importlib.util.spec_from_file_location(
                "improved_network", 
                os.path.join(warp_codes_dir, 'improved_network.py')
            )
            improved_network = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(improved_network)
            ImprovedWarpNetwork = improved_network.ImprovedWarpNetwork
        # 回退到network.py
        elif os.path.exists(os.path.join(warp_codes_dir, 'network.py')):
            spec = importlib.util.spec_from_file_location(
                "network", 
                os.path.join(warp_codes_dir, 'network.py')
            )
            network = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(network)
            ImprovedWarpNetwork = network.Network
        else:
            raise ImportError(f"在{warp_codes_dir}中找不到网络模块文件")
            
    except Exception as e:
        print(f"Error loading network module: {e}")
        print("Falling back to virtual data generation")
        return prepare_virtual_composition_data(config, args, mode)
        
    # 获取输入图像路径
    img1_path = os.path.join(source_path, 'img1')
    img2_path = os.path.join(source_path, 'img2')
    
    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        print(f"Warning: Source directories not found - {img1_path} or {img2_path}")
        print("Falling back to virtual data generation")
        return prepare_virtual_composition_data(config, args, mode)
    
    # 收集图像文件
    img1_files = sorted(glob.glob(os.path.join(img1_path, '*.jpg')))
    img2_files = sorted(glob.glob(os.path.join(img2_path, '*.jpg')))
    
    if len(img1_files) == 0 or len(img2_files) == 0:
        print(f"Warning: No images found in source directories")
        print(f"img1_path: {img1_path}, found {len(img1_files)} files")
        print(f"img2_path: {img2_path}, found {len(img2_files)} files")
        print("Falling back to virtual data generation")
        return prepare_virtual_composition_data(config, args, mode)
    
    print(f"Found {len(img1_files)} images in img1 and {len(img2_files)} images in img2")
    
    # 加载Warp模型
    try:
        # 初始化模型并确保在正确的设备上
        model = ImprovedWarpNetwork().to(device)
        
        # 获取模型路径
        model_path = None
        if hasattr(config['warp'][mode], 'model_path') and config['warp'][mode]['model_path']:
            model_path = config['warp'][mode]['model_path']
        elif mode == 'test' and hasattr(config['warp'], 'model_path') and config['warp']['model_path']:
            model_path = config['warp']['model_path']
        
        # 如果通过命令行参数提供了模型路径，优先使用它
        import sys
        for i in range(len(sys.argv)):
            if sys.argv[i] == '--model_path' and i + 1 < len(sys.argv):
                model_path = sys.argv[i + 1]
                break
        
        if not model_path:
            # 尝试从模型目录加载最新的模型
            model_dir = config['warp']['train']['model_save_path']
            checkpoint_files = sorted(glob.glob(os.path.join(model_dir, '*.pth')))
            if checkpoint_files:
                model_path = checkpoint_files[-1]  # 使用最新的checkpoint
            else:
                print("Warning: No model path specified and no checkpoints found")
                print("Using untrained model - results may be poor")
        
        # 加载模型权重
        if model_path and os.path.exists(model_path):
            print(f"Loading Warp model from {model_path}")
            try:
                # 使用指定设备加载模型权重
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                print("Using untrained model - results may be poor")
        else:
            print(f"Model path not found: {model_path}")
            print("Using untrained model - results may be poor")
        
        # 确保模型在正确的设备上
        model = model.to(device)
        
        # 检查模型参数是否都在正确的设备上
        for name, param in model.named_parameters():
            if param.device != device:
                print(f"Warning: Parameter {name} is on {param.device}, moving to {device}")
                param.data = param.data.to(device)
        
        model.eval()  # 设置为评估模式
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to virtual data generation")
        return prepare_virtual_composition_data(config, args, mode)
    
    # 设置网格分辨率
    grid_h = 12
    grid_w = 12
    
    # 批处理图像
    print(f"Processing images for {mode} dataset...")
    num_processed = 0
    
    # 获取所有可用的图像数量
    max_images = min(len(img1_files), len(img2_files))
    
    # 根据process_all_data参数决定是否限制数据量
    if not args.process_all_data:
        print("Using limited dataset size...")
    if mode == 'train':
        if max_images > 1000:  # 训练集最多处理1000张
            max_images = 1000
                print(f"Limiting training dataset to {max_images} images")
    else:  # 测试集
        if max_images > 100:   # 测试集最多处理100张
            max_images = 100
                print(f"Limiting test dataset to {max_images} images")
    else:
        print(f"Processing all available data: {max_images} images")
    
    # 记录当前内存使用情况
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"CUDA memory before processing: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    
    with torch.no_grad():  # 禁用梯度计算
        for i in tqdm(range(max_images), desc=f"Generating {mode} data"):
            img1_file = img1_files[i]
            img2_file = img2_files[i]
            
            img1_name = os.path.basename(img1_file)
            img2_name = os.path.basename(img2_file)
            
            try:
                # 使用改进的process_image_pair_simple函数处理图像
                result = process_image_pair_simple(
                    img1_file, img2_file, output_path, model
                )
                
                if not result:
                    print(f"Error processing image pair: {img1_file}, {img2_file}")
                    continue
                
                # 提取字典中的结果
                warped_img1 = result['warp1']
                warped_img2 = result['warp2']
                mask1 = result['mask1']
                mask2 = result['mask2']
                
                # 保存结果
                cv2.imwrite(os.path.join(warp1_dir, img1_name), warped_img1)
                cv2.imwrite(os.path.join(warp2_dir, img2_name), warped_img2)
                cv2.imwrite(os.path.join(mask1_dir, img1_name), mask1)
                cv2.imwrite(os.path.join(mask2_dir, img2_name), mask2)
                
                # 额外应用Poisson融合进行后处理
                try:
                    # 识别重叠区域
                    overlap_mask = (mask1 > 10) & (mask2 > 10)
                    if np.sum(overlap_mask) > 100:  # 只有在有足够的重叠区域时才应用
                        # 将掩码转换为单通道
                        if len(mask1.shape) > 2:
                            mask1_gray = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
                        else:
                            mask1_gray = mask1
                        
                        # 创建混合权重 - 使用平滑过渡
                        weight = mask1_gray.astype(np.float32) / 255.0
                        
                        # 应用高斯模糊使权重平滑过渡
                        weight = cv2.GaussianBlur(weight, (21, 21), 11)
                        
                        # 应用简单的泊松融合
                        # 对于每个通道分别应用
                        for c in range(3):
                            # 计算拉普拉斯金字塔
                            lap1 = cv2.Laplacian(warped_img1[:,:,c].astype(np.float32), cv2.CV_32F)
                            lap2 = cv2.Laplacian(warped_img2[:,:,c].astype(np.float32), cv2.CV_32F)
                            
                            # 根据权重混合拉普拉斯
                            lap_blended = weight * lap1 + (1 - weight) * lap2
                            
                            # 使用混合的拉普拉斯重建
                            # 为简化，直接使用傅里叶变换求解泊松方程
                            img_blended = np.fft.ifft2(np.fft.fft2(lap_blended) / 
                                                     (np.fft.fft2(np.array([[0,1,0],[1,-4,1],[0,1,0]], 
                                                                           dtype=np.float32)) + 1e-10))
                            img_blended = np.real(img_blended)
                            
                            # 归一化到原始范围
                            min_val = np.min(img_blended)
                            max_val = np.max(img_blended)
                            if max_val > min_val:
                                img_blended = 255 * (img_blended - min_val) / (max_val - min_val)
                            
                            # 只在重叠区域应用
                            warped_img1[:,:,c] = np.where(
                                overlap_mask, 
                                img_blended.astype(np.uint8), 
                                warped_img1[:,:,c]
                            )
                        
                        # 保存后处理的结果
                        poisson_dir = os.path.join(os.path.dirname(warp1_dir), 'poisson_blend')
                        os.makedirs(poisson_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(poisson_dir, img1_name), warped_img1)
                        print(f"应用Poisson融合后处理到 {img1_name}")
                except Exception as e:
                    print(f"Poisson融合过程中出错: {e}, 继续下一张")
                
                num_processed += 1
                
                # 定期清理缓存，防止内存溢出
                if i % 20 == 0 and torch.cuda.is_available() and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing image pair {i}: {e}")
                continue
    
    # 清理GPU内存
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"CUDA memory after processing: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    
    print(f"Composition {mode} data preparation complete. Successfully processed {num_processed}/{max_images} image pairs.")
    print(f"Data saved to: {output_path}")
    
    # 更新配置以指向新生成的数据
    if mode == 'train':
        config['composition']['train']['train_path'] = output_path
    else:  # mode == 'test'
        config['composition']['test']['test_path'] = output_path
    
    print(f"Updated config['{mode}'] path: {output_path}")
    print(f"==== Composition {mode.upper()} Data Preparation Completed ====")
    
    return config

def process_image_pair_simple(img1_path, img2_path, output_dir, warp_model=None):
    """
    使用简单的预处理方法处理一对图像，不进行尺寸调整
    
    Args:
        img1_path: 第一张图像的路径
        img2_path: 第二张图像的路径
        output_dir: 输出目录
        warp_model: 可选的变形模型
    
    Returns:
        处理结果字典，包含warped图像和掩码
    """
    try:
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 创建子目录
        warp1_dir = os.path.join(output_dir, 'warp1')
        warp2_dir = os.path.join(output_dir, 'warp2')
        mask1_dir = os.path.join(output_dir, 'mask1')
        mask2_dir = os.path.join(output_dir, 'mask2')
        
        for d in [warp1_dir, warp2_dir, mask1_dir, mask2_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
        
        # 获取文件名（不带路径和扩展名）
        img1_name = os.path.basename(img1_path)
        img2_name = os.path.basename(img2_path)
        
        # 读取图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"无法读取图像: {img1_path} 或 {img2_path}")
            return False
        
        # 使用原始分辨率，但检查是否太大
        max_dim = 2000  # 最大允许维度
        resize_factor1 = 1.0
        resize_factor2 = 1.0
        
        # 检查并适当缩小图像1
        if img1.shape[0] > max_dim or img1.shape[1] > max_dim:
            resize_factor1 = max_dim / max(img1.shape[0], img1.shape[1])
            new_height = int(img1.shape[0] * resize_factor1)
            new_width = int(img1.shape[1] * resize_factor1)
            img1 = cv2.resize(img1, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"调整图像1到合理大小: {new_width}x{new_height}")
        
        # 检查并适当缩小图像2
        if img2.shape[0] > max_dim or img2.shape[1] > max_dim:
            resize_factor2 = max_dim / max(img2.shape[0], img2.shape[1])
            new_height = int(img2.shape[0] * resize_factor2)
            new_width = int(img2.shape[1] * resize_factor2)
            img2 = cv2.resize(img2, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"调整图像2到合理大小: {new_width}x{new_height}")
        
        print(f"处理图像: img1 {img1.shape[1]}x{img1.shape[0]}, img2 {img2.shape[1]}x{img2.shape[0]}")
        
        # 预处理图像以提高特征匹配质量
        img1_enhanced = img1.copy()
        img2_enhanced = img2.copy()
        
        # 使用CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for i in range(3):  # 对每个通道应用CLAHE
            img1_enhanced[:,:,i] = clahe.apply(img1[:,:,i])
            img2_enhanced[:,:,i] = clahe.apply(img2[:,:,i])
        
        if warp_model is not None:
            # 使用模型进行变形
            try:
                # 将图像转换为模型所需格式
                img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
                img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
                
                # 添加批次维度
                img1_tensor = img1_tensor.unsqueeze(0)
                img2_tensor = img2_tensor.unsqueeze(0)
                
                # 确保在执行推理之前将张量移动到正确的设备
                device = next(warp_model.parameters()).device
                img1_tensor = img1_tensor.to(device)
                img2_tensor = img2_tensor.to(device)
                
                # 使用模型生成变形图像
                warped_img1_tensor, warped_img2_tensor = warp_model(img1_tensor, img2_tensor)
                
                # 转换回numpy格式
                warped_img1 = (warped_img1_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                warped_img2 = (warped_img2_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            except Exception as e:
                print(f"模型变形失败: {e}")
                warped_img1 = None
                warped_img2 = None
        else:
            warped_img1 = None
            warped_img2 = None
        
        # 如果模型变形失败，使用传统方法
        if warped_img1 is None or warped_img2 is None:
            try:
                # 使用增强后的图像进行特征匹配
                warped_img1, warped_img2, H = improved_feature_matching_and_warping(img1_enhanced, img2_enhanced)
                
                if warped_img1 is None or warped_img2 is None:
                    print("找到的好匹配点不足，尝试额外的处理...")
                    
                    # 尝试额外处理方法：多尺度特征检测
                    warped_img1, warped_img2, H = multi_scale_feature_matching(img1, img2)
                    
                    if warped_img1 is None or warped_img2 is None:
                        print("多尺度特征检测也失败，使用原始图像")
                        warped_img1 = img1
                        warped_img2 = img2
            except Exception as e:
                print(f"传统变形方法失败: {e}")
                warped_img1 = img1
                warped_img2 = img2
        
        # 生成掩码 - 使用改进的掩码生成
        mask1 = generate_improved_mask(img2, warped_img1)
        mask2 = generate_improved_mask(img1, warped_img2)
        
        # 检查掩码是否有效
        if mask1 is None or mask2 is None or np.sum(mask1) < 100 or np.sum(mask2) < 100:
            print("生成的掩码无效，使用改进的默认掩码")
            # 创建渐变掩码 - 使用径向渐变
            h, w = warped_img1.shape[:2]
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            
            # 计算到中心的距离，创建径向渐变
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt((h/2)**2 + (w/2)**2)
            
            # 创建平滑渐变掩码
            mask1 = np.ones((h, w), dtype=np.float32)
            mask1 = 1.0 - 0.7 * (dist / max_dist)  # 中心为1，边缘为0.3
            
            # 使用sigmoid平滑过渡
            mask1 = 1.0 / (1.0 + np.exp(-8.0 * (mask1 - 0.5)))
            
            # 高斯模糊使边缘更平滑
            kernel_size = min(31, max(11, int(min(h, w) / 30)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            mask1 = cv2.GaussianBlur(mask1, (kernel_size, kernel_size), kernel_size/4)
            
            # 转换为8位图像
            mask1 = (mask1 * 255).astype(np.uint8)
            
            # 对mask2使用相同处理
            h, w = warped_img2.shape[:2]
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt((h/2)**2 + (w/2)**2)
            
            mask2 = np.ones((h, w), dtype=np.float32)
            mask2 = 1.0 - 0.7 * (dist / max_dist)
            mask2 = 1.0 / (1.0 + np.exp(-8.0 * (mask2 - 0.5)))
            mask2 = cv2.GaussianBlur(mask2, (kernel_size, kernel_size), kernel_size/4)
            mask2 = (mask2 * 255).astype(np.uint8)
        
        # 将结果准备为字典返回
        result = {
            'warp1': warped_img1,
            'warp2': warped_img2,
            'mask1': mask1,
            'mask2': mask2
        }
        
        return result
        
    except Exception as e:
        print(f"处理图像对时发生错误: {e}")
        return False

def improved_feature_matching_and_warping(img1, img2):
    """
    使用改进的特征匹配和图像变形方法
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        
    Returns:
        warped_img1: 变形后的img1
        warped_img2: 变形后的img2
        H: 变换矩阵
    """
    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 使用SIFT检测特征点
    sift = cv2.SIFT_create(nfeatures=2000)  # 增加特征点数量
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if len(kp1) < 10 or len(kp2) < 10:
        print("SIFT检测到的特征点太少")
        return None, None, None
    
    # 使用FLANN进行特征匹配
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
    flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
    # 应用Lowe比率测试筛选好的匹配
            good_matches = []
            for m, n in matches:
        if m.distance < 0.7 * n.distance:  # 使用更严格的阈值
                    good_matches.append(m)
            
    # 如果好的匹配点太少，返回None
    if len(good_matches) < 10:
        print(f"找到的好匹配点不足: {len(good_matches)}")
        return None, None, None
    
    # 从好的匹配点提取出坐标
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
    # 使用RANSAC计算单应矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is None:
        print("无法计算单应矩阵")
        return None, None, None
    
    # 检查单应矩阵的质量
    inliers = mask.ravel() == 1
    if np.sum(inliers) < 8:
        print(f"内点数量太少: {np.sum(inliers)}")
        return None, None, None
    
    # 应用单应矩阵进行变形
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
    # 计算变换后的边界
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    # 变换img1的角点到img2的坐标系
    warp_corners1 = cv2.perspectiveTransform(corners1, H)
    
    # 计算边界框
    all_corners = np.vstack([corners2, warp_corners1])
    x_min = np.int32(np.min(all_corners[:, 0, 0]))
    y_min = np.int32(np.min(all_corners[:, 0, 1]))
    x_max = np.int32(np.max(all_corners[:, 0, 0]))
    y_max = np.int32(np.max(all_corners[:, 0, 1]))
    
    # 计算平移矩阵，确保所有内容都在可见范围内
    T = np.eye(3)
    if x_min < 0:
        T[0, 2] = -x_min
    if y_min < 0:
        T[1, 2] = -y_min
    
    # 应用平移矩阵
    H_with_T = T @ H
    
    # 计算输出图像的尺寸
    output_size = (x_max - x_min, y_max - y_min)
    
    # 应用变换
    warped_img1 = cv2.warpPerspective(img1, H_with_T, output_size)
    
    # 为img2创建一个新图像，位置适当偏移
    warped_img2 = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    warped_img2[int(T[1, 2]):int(T[1, 2])+h2, int(T[0, 2]):int(T[0, 2])+w2] = img2
    
    return warped_img1, warped_img2, H_with_T

def multi_scale_feature_matching(img1, img2):
    """
    使用多尺度特征匹配方法进行图像对齐
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        
    Returns:
        warped_img1: 变形后的img1
        warped_img2: 变形后的img2
        H: 变换矩阵
    """
    # 创建不同尺度的图像金字塔
    pyramid_levels = 3
    img1_pyramid = [img1]
    img2_pyramid = [img2]
    
    for i in range(1, pyramid_levels):
        img1_pyramid.append(cv2.pyrDown(img1_pyramid[-1]))
        img2_pyramid.append(cv2.pyrDown(img2_pyramid[-1]))
    
    # 从最低分辨率开始处理
    H_final = None
    
    # 从最低分辨率层开始
    for level in range(pyramid_levels-1, -1, -1):
        current_img1 = img1_pyramid[level]
        current_img2 = img2_pyramid[level]
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(current_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(current_img2, cv2.COLOR_BGR2GRAY)
        
        # 检测特征点
        detector = cv2.ORB_create(nfeatures=2000)
        
        # 如果我们有前一级的单应矩阵，用它来指导当前的特征检测
        if H_final is not None:
            # 缩放H_final以适应当前层级
            scale_factor = 2 ** level
            H_scaled = H_final.copy()
            H_scaled[0, 2] *= scale_factor
            H_scaled[1, 2] *= scale_factor
            
            # 应用变换到灰度图
            warped_gray1 = cv2.warpPerspective(gray1, H_scaled, (gray2.shape[1], gray2.shape[0]))
            
            # 在变换后的灰度图上检测特征
            kp1, des1 = detector.detectAndCompute(warped_gray1, None)
        else:
            kp1, des1 = detector.detectAndCompute(gray1, None)
        
        kp2, des2 = detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            continue
        
        # 特征匹配
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        
        # 按距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 只使用最佳的一部分匹配
        good_matches = matches[:int(len(matches) * 0.8)]
        
        if len(good_matches) < 10:
            continue
        
        # 提取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 计算单应矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            continue
        
        # 如果这是最高分辨率的层次或我们还没有得到单应矩阵
        if level == 0 or H_final is None:
            H_final = H
        else:
            # 缩放H_final以适应当前层级
            scale_factor = 2 ** level
            H_scaled = H_final.copy()
            H_scaled[0, 2] /= scale_factor
            H_scaled[1, 2] /= scale_factor
            
            # 将当前层级的变换应用到已有的变换上
            H_final = H @ H_scaled
    
    # 如果找不到变换，返回None
    if H_final is None:
        return None, None, None
    
    # 应用最终变换
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 计算变换后的边界
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    # 变换img1的角点
    warp_corners1 = cv2.perspectiveTransform(corners1, H_final)
    
    # 计算边界框
    all_corners = np.vstack([corners2, warp_corners1])
    x_min = np.int32(np.min(all_corners[:, 0, 0]))
    y_min = np.int32(np.min(all_corners[:, 0, 1]))
    x_max = np.int32(np.max(all_corners[:, 0, 0]))
    y_max = np.int32(np.max(all_corners[:, 0, 1]))
    
    # 计算平移矩阵
    T = np.eye(3)
    if x_min < 0:
        T[0, 2] = -x_min
    if y_min < 0:
        T[1, 2] = -y_min
    
    # 应用平移矩阵
    H_with_T = T @ H_final
    
    # 计算输出图像的尺寸
    output_size = (x_max - x_min, y_max - y_min)
    
    # 应用变换
    warped_img1 = cv2.warpPerspective(img1, H_with_T, output_size)
    
    # 为img2创建一个新图像，位置适当偏移
    warped_img2 = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    warped_img2[int(T[1, 2]):int(T[1, 2])+h2, int(T[0, 2]):int(T[0, 2])+w2] = img2
    
    return warped_img1, warped_img2, H_with_T

def preprocess_image(img):
    """将OpenCV图像转换为PyTorch张量"""
    # 转换为float32并归一化到[-1, 1]
    img = img.astype(np.float32) / 127.5 - 1.0
    # HWC -> CHW
    img = np.transpose(img, [2, 0, 1])
    # 添加批次维度
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img)

def tensor_to_image(tensor):
    """将PyTorch张量转换回OpenCV图像"""
    # 移除批次维度，CHW -> HWC
    img = tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
    # 从[-1, 1]转换回[0, 255]
    img = (img + 1) * 127.5
    return img.astype(np.uint8)

def apply_warp(img, H, mesh_motion, grid_h, grid_w):
    """应用单应性变换和网格变形"""
    # 导入所需模块
    import Warp.Codes.utils.torch_homo_transform as torch_homo_transform
    import Warp.Codes.utils.torch_tps_transform as torch_tps_transform
    
    # 获取设备信息
    device = img.device
    
    # 应用单应性变换
    h, w = img.size(2), img.size(3)
    warped_img = torch_homo_transform.transformer(img, H, (h, w))
    
    # 获取网格
    rigid_mesh = get_rigid_mesh(img.size(0), h, w, grid_h, grid_w, device)
    mesh = rigid_mesh + mesh_motion
    
    # 归一化网格
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, h, w)
    norm_mesh = get_norm_mesh(mesh, h, w)
    
    # 应用TPS变换
    mask = torch.ones_like(img)
    warped_img = torch_tps_transform.transformer(torch.cat((warped_img, mask), 1), 
                                              norm_mesh, norm_rigid_mesh, 
                                              (h, w))
    
    return warped_img[:,:3,:,:]

def get_rigid_mesh(batch_size, h, w, grid_h, grid_w, device=None):
    """生成规则网格"""
    mesh = torch.zeros(batch_size, grid_h+1, grid_w+1, 2)
    for i in range(grid_h+1):
        for j in range(grid_w+1):
            mesh[:, i, j, 0] = j * (w / grid_w)
            mesh[:, i, j, 1] = i * (h / grid_h)
    
    if device is not None:
        mesh = mesh.to(device)
    
    return mesh

def get_norm_mesh(mesh, h, w):
    """归一化网格坐标"""
    norm_mesh = mesh.clone()
    norm_mesh[:, :, :, 0] = norm_mesh[:, :, :, 0] / (w/2) - 1
    norm_mesh[:, :, :, 1] = norm_mesh[:, :, :, 1] / (h/2) - 1
    return norm_mesh

def generate_improved_mask(reference_img, target_img):
    """
    生成改进的蒙版，使用多层次渐变和边缘感知技术
    
    Args:
        reference_img: 参考图像，用于计算与目标图像的差异
        target_img: 目标图像，需要为其生成蒙版
        
    Returns:
        改进的蒙版图像
    """
    if reference_img is None or target_img is None:
        return None
    
    # 确保图像具有相同尺寸
    if reference_img.shape != target_img.shape:
        # 调整reference_img到target_img尺寸
        reference_img = cv2.resize(reference_img, (target_img.shape[1], target_img.shape[0]))
    
    # 转换为灰度图
    if len(reference_img.shape) > 2:
        reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    else:
        reference_gray = reference_img
        
    if len(target_img.shape) > 2:
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    else:
        target_gray = target_img
    
    # 计算图像之间的差异
    diff = cv2.absdiff(reference_gray, target_gray)
    
    # 应用阈值提取差异较大的区域
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # 形态学操作去除噪点并填充小孔
    kernel_size = max(3, min(int(min(target_img.shape[0], target_img.shape[1]) / 100), 15))
    if kernel_size % 2 == 0:
        kernel_size += 1  # 确保是奇数
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 计算到图像中心的距离，创建径向梯度
    height, width = mask.shape[:2]
    center_y, center_x = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    
    # 计算到中心的距离
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # 归一化距离到[0,1]范围
    max_dist = np.sqrt(center_x**2 + center_y**2)
    norm_dist = dist_from_center / max_dist
    
    # 创建中心加权掩码 - 中心附近权重较高
    center_weight = 1.0 - 0.7 * norm_dist  # 边缘降低到0.3的权重
    
    # 将二值掩码与中心权重相乘
    weighted_mask = (mask.astype(np.float32) / 255.0) * center_weight
    
    # 应用Sigmoid函数使过渡更平滑
    # Sigmoid公式: 1/(1+exp(-k*(x-x0)))
    # 其中k控制陡度，x0控制中点
    k = 8.0  # 陡度系数
    x0 = 0.5  # 中点
    smooth_mask = 1.0 / (1.0 + np.exp(-k * (weighted_mask - x0)))
    
    # 应用多次高斯模糊创建非常平滑的边缘
    # 第一次使用大核进行整体平滑
    large_kernel = min(31, max(11, int(min(height, width) / 30)))
    if large_kernel % 2 == 0:
        large_kernel += 1
    smooth_mask = cv2.GaussianBlur(smooth_mask, (large_kernel, large_kernel), large_kernel/4)
    
    # 第二次使用中等核进行细节平滑
    smooth_mask = cv2.GaussianBlur(smooth_mask, (15, 15), 5)
    
    # 使用Sobel算子找到边缘区域
    sobel_x = cv2.Sobel(smooth_mask, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smooth_mask, cv2.CV_32F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # 找到需要特别处理的边缘区域
    edge_mask = edge_magnitude > 0.05
    
    # 创建特别平滑的边缘区域版本
    edge_version = cv2.GaussianBlur(smooth_mask, (21, 21), 7)
    
    # 在边缘区域使用更平滑的版本
    final_mask = np.where(edge_mask, edge_version, smooth_mask)
    
    # 再次应用Sigmoid平滑边缘过渡
    final_mask = 1.0 / (1.0 + np.exp(-10.0 * (final_mask - 0.5)))
    
    # 转换回8位灰度图
    final_mask_uint8 = (final_mask * 255).astype(np.uint8)
    
    return final_mask_uint8

def prepare_virtual_composition_data(config, args, mode='train'):
    """
    当无法使用真实数据时，生成虚拟的Composition数据
    
    Args:
        config: 配置字典
        args: 命令行参数
        mode: 'train'或'test'，指定准备训练集还是测试集
    """
    print(f"Generating virtual composition {mode} data...")
    
    # 在data/UDIS-D下创建新的目录结构
    output_base = os.path.join('data', 'UDIS-D', 'composition_data')
    output_path = os.path.join(output_base, mode)
    
    # 创建扭曲图像和掩码的目录
    warp1_dir = os.path.join(output_path, 'warp1')
    warp2_dir = os.path.join(output_path, 'warp2')
    mask1_dir = os.path.join(output_path, 'mask1')
    mask2_dir = os.path.join(output_path, 'mask2')
    
    # 确保目录存在
    for directory in [output_base, output_path, warp1_dir, warp2_dir, mask1_dir, mask2_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 根据process_all_data参数决定生成数量
    if args.process_all_data:
        num_images = 100 if mode == 'train' else 20  # 生成更多的虚拟数据
        print(f"Generating extended virtual dataset with {num_images} images")
    else:
        num_images = 20 if mode == 'train' else 5    # 默认的较小数据集
        print(f"Generating default virtual dataset with {num_images} images")
    
    # 生成虚拟数据
    for i in range(num_images):
        # 创建随机图像
        height, width = 512, 512
        img1 = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # 创建随机掩码 - 使用更自然的形状
        mask1 = np.zeros((height, width), dtype=np.uint8)
        mask2 = np.zeros((height, width), dtype=np.uint8)
        
        # 绘制椭圆作为掩码
        center1 = (np.random.randint(width//4, 3*width//4), np.random.randint(height//4, 3*height//4))
        axes1 = (np.random.randint(50, 150), np.random.randint(50, 150))
        angle1 = np.random.randint(0, 180)
        cv2.ellipse(mask1, center1, axes1, angle1, 0, 360, 255, -1)
        
        center2 = (np.random.randint(width//4, 3*width//4), np.random.randint(height//4, 3*height//4))
        axes2 = (np.random.randint(50, 150), np.random.randint(50, 150))
        angle2 = np.random.randint(0, 180)
        cv2.ellipse(mask2, center2, axes2, angle2, 0, 360, 255, -1)
        
        # 转换为3通道
        mask1_3ch = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
        mask2_3ch = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
        
        # 保存结果
        filename = f"virtual_{i:04d}.jpg"
        cv2.imwrite(os.path.join(warp1_dir, filename), img1)
        cv2.imwrite(os.path.join(warp2_dir, filename), img2)
        cv2.imwrite(os.path.join(mask1_dir, filename), mask1_3ch)
        cv2.imwrite(os.path.join(mask2_dir, filename), mask2_3ch)
    
    print(f"Generated {num_images} virtual image pairs for {mode} dataset")
    
    # 更新配置以指向虚拟数据
    if mode == 'train':
        config['composition']['train']['train_path'] = output_path
    else:  # mode == 'test'
        config['composition']['test']['test_path'] = output_path
    
    return config

def check_and_move_to_device(tensor, target_device):
    """
    检查张量的设备并移动到目标设备
    
    Args:
        tensor: 输入张量
        target_device: 目标设备
        
    Returns:
        在目标设备上的张量
    """
    if tensor is None:
        return None
        
    if not isinstance(tensor, torch.Tensor):
        return tensor
        
    # 检查设备
    if tensor.device != target_device:
        print(f"Warning: Tensor on {tensor.device} moved to {target_device}")
        tensor = tensor.to(target_device)
        
    return tensor

def simple_feature_matching_and_warping(img1, img2):
    """
    简单的特征匹配和图像变形函数
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        
    Returns:
        warped_img1: 变形后的第一张图像
        warped_img2: 变形后的第二张图像
        H: 单应性矩阵
    """
    # 增强图像细节以提高特征点检测效果
    img1_enhanced = cv2.detailEnhance(img1, sigma_s=10, sigma_r=0.15)
    img2_enhanced = cv2.detailEnhance(img2, sigma_s=10, sigma_r=0.15)
    
    # 转换为灰度图
    gray1 = cv2.cvtColor(img1_enhanced, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_enhanced, cv2.COLOR_BGR2GRAY)
    
    # 设置最小关键点数
    min_kp = 50
    
    # 首先尝试SIFT特征检测
    sift = cv2.SIFT_create(nfeatures=500)
    kp1 = sift.detect(gray1, None)
    kp2 = sift.detect(gray2, None)
    kp1 = list(kp1)
    kp2 = list(kp2)
    
    # 如果检测到的关键点数量不足，尝试ORB
    if len(kp1) < min_kp or len(kp2) < min_kp:
        print(f"SIFT找到的关键点不足，尝试ORB。SIFT找到：{len(kp1)}在img1，{len(kp2)}在img2")
        orb = cv2.ORB_create(nfeatures=1000)
        kp1 = orb.detect(gray1, None)
        kp2 = orb.detect(gray2, None)
        kp1 = list(kp1)
        kp2 = list(kp2)
    
    # 如果仍然不足，尝试AKAZE
    if len(kp1) < min_kp or len(kp2) < min_kp:
        print(f"ORB找到的关键点不足，尝试AKAZE。ORB找到：{len(kp1)}在img1，{len(kp2)}在img2")
        akaze = cv2.AKAZE_create()
        kp1 = akaze.detect(gray1, None)
        kp2 = akaze.detect(gray2, None)
        kp1 = list(kp1)
        kp2 = list(kp2)
    
    # 如果关键点数量仍然不足，基于边缘检测添加额外的关键点
    if len(kp1) < min_kp or len(kp2) < min_kp:
        print(f"不足的关键点，增强检测。找到 {len(kp1)} 在img1, {len(kp2)} 在img2")
        # 在边缘区域添加更多关键点
        edges1 = cv2.Canny(gray1, 100, 200)
        edges2 = cv2.Canny(gray2, 100, 200)
        
        # 从边缘提取关键点
        edge_points1 = np.where(edges1 > 0)
        edge_points2 = np.where(edges2 > 0)
        
        # 选择子集作为新关键点（每隔10个像素）
        if len(edge_points1[0]) > 0:
            for i in range(0, len(edge_points1[0]), 10):
                if i < len(edge_points1[0]):
                    y, x = edge_points1[0][i], edge_points1[1][i]
                    kp = cv2.KeyPoint(float(x), float(y), 10)
                    kp1.append(kp)
        
        if len(edge_points2[0]) > 0:
            for i in range(0, len(edge_points2[0]), 10):
                if i < len(edge_points2[0]):
                    y, x = edge_points2[0][i], edge_points2[1][i]
                    kp = cv2.KeyPoint(float(x), float(y), 10)
                    kp2.append(kp)
    
    # 计算特征描述符
    if len(kp1) > 0 and len(kp2) > 0:
        # 使用SIFT计算描述符
        _, des1 = sift.compute(gray1, kp1)
        _, des2 = sift.compute(gray2, kp2)
        
        # 确保描述符不为空
        if des1 is None or des2 is None:
            print("描述符计算失败")
            return img1.copy(), img2.copy(), np.eye(3)
        
        # 特征匹配
        bf = cv2.BFMatcher()
        try:
            matches = bf.knnMatch(des1, des2, k=2)
            
            # 应用Lowe比率测试筛选好的匹配
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            # 只有当找到足够的好匹配点时才继续处理
            if len(good_matches) > min_kp:
                # 提取匹配点坐标
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # 计算单应性矩阵
                H, mask_matches = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is None:
                    print("单应性矩阵计算失败，使用恒等变换")
                    H = np.eye(3)  # 使用恒等变换
                
                # 使用单应性矩阵对img1进行变换
                warped_img1 = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
                warped_img2 = img2.copy()  # 第二张图像保持不变
                
                return warped_img1, warped_img2, H
            else:
                print(f"找到的好匹配点不足: {len(good_matches)}")
                return None, None, None
                
        except Exception as e:
            print(f"匹配过程中出错: {e}")
            return None, None, None
    else:
        print("无法检测到足够的关键点")
        return None, None, None

def prepare_test_composition_data(config, force_prepare=False, use_limited=True, limit_size=100):
    """
    准备用于图像组合测试的数据集
    
    Args:
        config: 配置字典
        force_prepare: 是否强制重新准备数据
        use_limited: 是否使用限制大小的数据集
        limit_size: 限制的数据集大小
    
    Returns:
        更新后的配置字典
    """
    print("==== Preparing Composition TEST Data ====")
    
    # 配置路径
    output_path = 'data/UDIS-D/composition_data/test'
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 创建子目录
    sub_dirs = ['warp1', 'warp2', 'mask1', 'mask2']
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(output_path, sub_dir)
        print(f"Ensured directory exists: {sub_dir_path}")
        os.makedirs(sub_dir_path, exist_ok=True)
    
    # 检查数据集是否已经存在
    warp1_files = glob.glob(os.path.join(output_path, 'warp1', '*.jpg'))
    warp2_files = glob.glob(os.path.join(output_path, 'warp2', '*.jpg'))
    mask1_files = glob.glob(os.path.join(output_path, 'mask1', '*.jpg'))
    mask2_files = glob.glob(os.path.join(output_path, 'mask2', '*.jpg'))
    
    # 如果数据已经准备好，且不强制重新准备，则跳过
    if (len(warp1_files) > 0 and len(warp2_files) > 0 and 
        len(mask1_files) > 0 and len(mask2_files) > 0 and 
        not force_prepare):
        print(f"发现完整的Composition test 数据集:")
        print(f"  - warp1: {len(warp1_files)} 文件")
        print(f"  - warp2: {len(warp2_files)} 文件")
        print(f"  - mask1: {len(mask1_files)} 文件")
        print(f"  - mask2: {len(mask2_files)} 文件")
        print(f"跳过数据生成过程。如需重新生成，请使用--force_prepare参数")
        
        # 更新配置
        config['test'] = output_path
        return config
    
    # 使用CPU进行数据准备
    device = torch.device('cpu')
    print(f"Using device: {device} for data preparation")
    
    # 加载测试图像
    test_path = config.get('test_path', 'data/UDIS-D/testing')
    img1_dir = os.path.join(test_path, 'img1')
    img2_dir = os.path.join(test_path, 'img2')
    
    # 获取所有图像文件
    img1_files = sorted(glob.glob(os.path.join(img1_dir, '*.jpg')))
    img2_files = sorted(glob.glob(os.path.join(img2_dir, '*.jpg')))
    
    # 确保两个目录的文件数相等
    if len(img1_files) != len(img2_files):
        print(f"警告: img1_dir和img2_dir的图像数量不相等。img1: {len(img1_files)}, img2: {len(img2_files)}")
        # 取最小数量
        num_files = min(len(img1_files), len(img2_files))
        img1_files = img1_files[:num_files]
        img2_files = img2_files[:num_files]
    
    print(f"Found {len(img1_files)} images in img1 and {len(img2_files)} images in img2")
    
    # 加载Warp模型（如果存在）
    model = None
    if hasattr(config, 'warp_model_path') and os.path.exists(config.warp_model_path):
        model_path = config.warp_model_path
    else:
        # 尝试查找最新的检查点
        checkpoints = sorted(glob.glob('Warp/model/checkpoint_epoch_*.pth'))
        if checkpoints:
            model_path = checkpoints[-1]  # 使用最新的检查点
            print(f"Loading Warp model from {model_path}")
            
            # 加载模型
            from Warp.warp_model import WarpNet
            model = WarpNet()
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("Model loaded successfully")
        else:
            print("No Warp model checkpoints found. Using traditional warping methods.")
    
    # 处理图像对
    print("Processing images for test dataset...")
    if use_limited:
        print("Using limited dataset size...")
        # 如果需要限制数据集大小
        if limit_size < len(img1_files):
            img1_files = img1_files[:limit_size]
            img2_files = img2_files[:limit_size]
        print(f"Limiting test dataset to {len(img1_files)} images")
    
    # 使用tqdm创建进度条
    pbar = tqdm(total=len(img1_files), desc="Generating test data")
    num_processed = 0
    
    for i, (img1_file, img2_file) in enumerate(zip(img1_files, img2_files)):
        # 使用改进的process_image_pair_simple函数处理图像
        result = process_image_pair_simple(
            img1_file, img2_file, output_path, model
        )
        
        if not result:
            print(f"Error processing image pair: {img1_file}, {img2_file}")
            continue
        
        # 成功处理，增加计数    
        num_processed += 1
            
        # 更新进度条
        pbar.update(1)
    
    # 关闭进度条
    pbar.close()
    
    # 打印统计信息
    print(f"Composition test data preparation complete. Successfully processed {num_processed}/{len(img1_files)} image pairs.")
    print(f"Data saved to: {output_path}")
    
    # 更新配置
    config['test'] = output_path
    print(f"Updated config['test'] path: {output_path}")
    print("==== Composition TEST Data Preparation Completed ====")
    
    return config

def setup_distributed(rank, world_size, timeout_sec=1800):
    """
    设置分布式训练环境，添加超时处理
    """
    print(f"[进程 {rank}] 设置分布式环境，超时时间: {timeout_sec}秒")
    
    # 从环境变量获取地址和端口
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    print(f"[进程 {rank}] 使用 MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # 初始化进程组，添加超时设置
    try:
        import datetime
        timeout = datetime.timedelta(seconds=timeout_sec)
        dist.init_process_group(
            "nccl", 
            init_method="env://",
            rank=rank, 
            world_size=world_size,
            timeout=timeout
        )
        print(f"[进程 {rank}] 分布式进程组初始化成功!")
        return True
    except Exception as e:
        print(f"[进程 {rank}] 初始化分布式进程组失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_distributed():
    """
    清理分布式训练环境
    """
    try:
        if dist.is_initialized():
            print(f"[进程 {dist.get_rank()}] 清理分布式环境")
            dist.destroy_process_group()
            print(f"[进程 {dist.get_rank()}] 分布式环境已清理")
    except:
        print("清理分布式环境出错，可能已被清理或未初始化")

def custom_collate_fn(batch):
    if not batch:
        return {}
    
    # 创建各字段的列表
    keys = batch[0].keys()
    collated_batch = {k: [] for k in keys}
    
    # 确定批次中的最大高度和宽度
    max_h, max_w = 0, 0
    for sample in batch:
        for k in ['base_image', 'warp_image']:
            if k in sample:
                max_h = max(max_h, sample[k].shape[1])
                max_w = max(max_w, sample[k].shape[2])
    
    # 将最大高度和宽度调整为32的倍数
    max_h = ((max_h + 31) // 32) * 32
    max_w = ((max_w + 31) // 32) * 32
    
    # 限制最大尺寸
    if max_h > 1024 or max_w > 1024:
        print(f"警告: 批次尺寸过大 ({max_h}x{max_w})，可能导致内存不足。")
        max_h = min(max_h, 1024)
        max_w = min(max_w, 1024)
        print(f"已限制批次尺寸为: {max_h}x{max_w}")
    
    # 对每个样本进行填充
    for sample in batch:
        for key in keys:
            tensor = sample[key]
            # 对图像和掩码使用不同的填充模式
            if 'mask' in key:
                # 掩码使用常数0填充
                padded_tensor = pad_tensor(tensor, max_h, max_w, value=0)
            else:
                # 图像使用常数0填充
                padded_tensor = pad_tensor(tensor, max_h, max_w, value=0)
            collated_batch[key].append(padded_tensor)
    
    # 将列表转换为批次张量
    for key in keys:
        collated_batch[key] = torch.stack(collated_batch[key], dim=0)
    
    return collated_batch

def pad_tensor(tensor, target_h, target_w, value=0):
    """填充张量到目标大小"""
    c, h, w = tensor.shape
    
    # 计算需要的填充量
    pad_h = target_h - h
    pad_w = target_w - w
    
    # 如果不需要填充，直接返回
    if pad_h <= 0 and pad_w <= 0:
        return tensor
    
    # 计算填充量 (左, 右, 上, 下)
    padding = (0, pad_w, 0, pad_h)
    
    # 应用填充
    padded_tensor = F.pad(tensor, padding, mode='constant', value=value)
    
    return padded_tensor

def modify_composition_network_if_needed():
    """检查并修改Composition网络模型以符合设计思路"""
    print("跳过Composition网络模型修改，使用内置的实现")
    return  # 直接返回，跳过模型修改
    
    try:
        from Composition.Codes.enhanced_network import ImprovedDiffusionComposition
        
        # 保存原始类定义
        original_init = ImprovedDiffusionComposition.__init__
        original_forward = ImprovedDiffusionComposition.forward
        
        # 检查是否需要修改
        if not hasattr(ImprovedDiffusionComposition, 'extract_features'):
            print("正在调整Composition网络模型以优化蒙版生成功能...")
            
            # 添加特征提取方法
            def extract_features(self, base_image, warp_image, base_mask, warp_mask):
                # 实现分层特征提取
                # 合并输入进行特征提取
                combined = torch.cat([base_image, warp_image, base_mask, warp_mask], dim=1)
                
                # 使用现有的特征提取网络
                if hasattr(self, 'encoder'):
                    features = self.encoder(combined)
                elif hasattr(self, 'feature_extractor'):
                    features = self.feature_extractor(combined)
                elif hasattr(self, 'channel_adapter'):
                    # 使用channel_adapter和初始层进行特征提取
                    x = self.channel_adapter(combined)
                    # 初始化时间步为0
                    t = torch.zeros(base_image.shape[0], device=base_image.device, dtype=torch.long)
                    try:
                        # 尝试使用time_mlp进行编码
                        t_emb = self.time_mlp(t)
                        # 使用下采样层进行特征提取
                        features = self.down1(x, t_emb)
                    except Exception as e:
                        print(f"使用time_mlp进行编码时出错: {e}")
                        # 创建一个默认的时间编码
                        t_emb = torch.zeros((base_image.shape[0], 128), device=base_image.device)
                        # 使用下采样层进行特征提取
                        try:
                            features = self.down1(x, t_emb)
                        except Exception as e:
                            print(f"使用下采样层进行特征提取时出错: {e}")
                            # 退回到简单的卷积特征提取
                            if hasattr(self, 'conv'):
                                features = self.conv(x)
                            else:
                                # 最后手段 - 返回输入作为特征
                                features = x
                else:
                    # 无法提取特征，返回None
                    print("警告：模型没有可用的特征提取器")
                    features = None
                
                return features
            
            # 替换forward方法使其专注于生成蒙版
            def new_forward(self, x, t=None, mode='train'):
                # 解包输入
                if isinstance(x, dict):
                    base_image = x['base_image']
                    warp_image = x['warp_image']
                    base_mask = x['base_mask']
                    warp_mask = x['warp_mask']
                elif isinstance(x, (list, tuple)) and len(x) == 4:
                    # 兼容原有接口，假设输入是四个张量的列表或元组
                    base_image, warp_image, base_mask, warp_mask = x
                # 兼容在train.py中的直接调用方式: model(warp1, warp2, mask1, mask2)
                elif torch.is_tensor(x) and torch.is_tensor(t) and len(x.shape) == 4 and len(t.shape) == 4:
                    warp_image, base_image, warp_mask, base_mask = x, t, mode, None
                    if base_mask is None and isinstance(warp_mask, torch.Tensor):
                        # 尝试创建匹配的base_mask
                        base_mask = torch.ones_like(warp_mask)
                    t = None
                    mode = 'inference'
                else:
                    # 兼容原有接口，假设输入是四个张量的列表或元组
                    base_image, warp_image, base_mask, warp_mask = x
                
                # 提取特征
                features = self.extract_features(base_image, warp_image, base_mask, warp_mask)
                
                # 根据模式决定处理流程
                if mode == 'train' and t is not None:
                    # 训练模式：生成蒙版并添加噪声
                    # 扩散模型训练逻辑
                    try:
                        # 尝试调用原始forward方法
                        original_forward = super(ImprovedDiffusionComposition, self).forward
                        output = original_forward(x, t)
                        return output
                    except Exception as e:
                        print(f"调用原始forward方法失败: {e}")
                        # 创建一个简单的掩码和结果
                        mask = self.generate_simple_mask(base_image, warp_image)
                        result = self.apply_mask(base_image, warp_image, mask)
                        return mask, result
                else:
                    # 推理模式：生成最终蒙版
                    
                    # 使用扩散模型采样生成蒙版 - 如果有sample方法
                    if hasattr(self, 'simple_sample') and features is not None and callable(self.simple_sample):
                        try:
                            mask = self.simple_sample(base_image.shape[0], base_image.shape[2:], features)
                        except Exception as e:
                            print(f"调用simple_sample方法失败: {e}")
                            mask = self.generate_simple_mask(base_image, warp_image)
                    else:
                        # 如果没有特征或sample方法，使用简单的方法生成掩码
                        mask = self.generate_simple_mask(base_image, warp_image)
                    
                    # 应用掩码进行拼接
                    result = self.apply_mask(base_image, warp_image, mask)
                    
                    # 对于原来的调用方式，返回与之匹配的结果
                    if isinstance(t, torch.Tensor) and len(t.shape) == 4:
                        return mask, result
                    
                    # 确保mask的尺寸与base_image匹配
                    if mask.shape[2:] != base_image.shape[2:]:
                        mask = torch.nn.functional.interpolate(
                            mask, 
                            size=base_image.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # 对于新的调用方式，返回字典
                    return {
                        'result': result,
                        'mask': mask,
                        'masked_warp': warp_image * mask  # 这里使用调整后的mask
                    }
                    
            # 添加辅助方法生成简单掩码
            def generate_simple_mask(self, base_image, warp_image):
                """生成一个简单的掩码，在没有扩散模型时使用"""
                # 基于两张图片的差异创建一个简单的掩码
                # 计算亮度差异
                base_gray = base_image.mean(dim=1, keepdim=True)
                warp_gray = warp_image.mean(dim=1, keepdim=True)
                diff = torch.abs(base_gray - warp_gray)
                
                # 使用简单阈值法
                mask = (diff > 0.2).float()
                
                # 平滑掩码边缘
                kernel_size = min(9, min(base_image.shape[2], base_image.shape[3]) // 10)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                if kernel_size >= 3:
                    mask = torch.nn.functional.avg_pool2d(
                        mask,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size//2
                    )
                
                return mask
                
            # 添加辅助方法应用掩码
            def apply_mask(self, base_image, warp_image, mask):
                """应用掩码拼接两张图片"""
                # 确保掩码在[0,1]范围内
                mask = torch.sigmoid(mask) if not torch.all((mask >= 0) & (mask <= 1)) else mask
                
                # 应用掩码
                masked_warp = warp_image * mask
                inverse_mask = 1.0 - mask
                result = masked_warp + base_image * inverse_mask
                
                return result
            
            # 添加方法到类
            ImprovedDiffusionComposition.extract_features = extract_features
            ImprovedDiffusionComposition.forward = new_forward
            
            # 添加辅助方法
            def generate_simple_mask(self, base_image, warp_image):
                """生成一个简单的掩码，在没有扩散模型时使用"""
                # 基于两张图片的差异创建一个简单的掩码
                # 计算亮度差异
                base_gray = base_image.mean(dim=1, keepdim=True)
                warp_gray = warp_image.mean(dim=1, keepdim=True)
                diff = torch.abs(base_gray - warp_gray)
                
                # 使用简单阈值法
                mask = (diff > 0.2).float()
                
                # 平滑掩码边缘
                kernel_size = min(9, min(base_image.shape[2], base_image.shape[3]) // 10)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                if kernel_size >= 3:
                    mask = torch.nn.functional.avg_pool2d(
                        mask,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size//2
                    )
                
                return mask
                
            def apply_mask(self, base_image, warp_image, mask):
                """应用掩码拼接两张图片"""
                # 确保掩码在[0,1]范围内
                mask = torch.sigmoid(mask) if not torch.all((mask >= 0) & (mask <= 1)) else mask
                
                # 应用掩码
                masked_warp = warp_image * mask
                inverse_mask = 1.0 - mask
                result = masked_warp + base_image * inverse_mask
                
                return result
            
            # 修改sample方法签名的包装器
            def simple_sample(self, batch_size, shape, features=None):
                """简化的sample方法，兼容原始的sample接口"""
                # 从模型输入或特征中获取基本参数
                device = next(self.parameters()).device
                if features is not None:
                    # 使用简单掩码替代sample
                    base_image = features[:, :3]
                    warp_image = features[:, 3:6]
                    # 生成一个与输入图像相同尺寸的掩码
                    mask = self.generate_simple_mask(base_image, warp_image)
                else:
                    # 确保shape是二元组
                    if not isinstance(shape, (list, tuple)):
                        shape = (shape, shape)
                    # 创建随机掩码
                    mask = torch.rand((batch_size, 1, shape[0], shape[1]), device=device)
                    mask = (mask > 0.5).float()
                return mask
                
            def apply_mask(self, base_image, warp_image, mask):
                """应用掩码拼接两张图片"""
                # 确保掩码在[0,1]范围内
                mask = torch.sigmoid(mask) if not torch.all((mask >= 0) & (mask <= 1)) else mask
                
                # 检查尺寸是否匹配，如果不匹配则调整mask尺寸
                if mask.shape[2:] != base_image.shape[2:]:
                    mask = torch.nn.functional.interpolate(
                        mask, 
                        size=base_image.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # 应用掩码
                masked_warp = warp_image * mask
                inverse_mask = 1.0 - mask
                result = masked_warp + base_image * inverse_mask
                
                return result
            
            # 添加辅助方法到类
            ImprovedDiffusionComposition.generate_simple_mask = generate_simple_mask
            ImprovedDiffusionComposition.apply_mask = apply_mask
            ImprovedDiffusionComposition.simple_sample = simple_sample
            
            print("Composition网络模型已调整以优化蒙版生成功能")
    
    except Exception as e:
        print(f"修改Composition网络模型时出错: {e}")
        print("建议直接修改Composition/Codes/network.py中的ImprovedDiffusionComposition类")

# 在调用训练或测试前执行
modify_composition_network_if_needed()

def main():
    # 解析命令行参数
    import argparse
    import os
    import sys
    import torch
    import datetime
    import numpy as np
    parser = argparse.ArgumentParser(description='UDTATIS: Unsupervised Deep Image Stitching with Enhanced Diffusion')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'end2end'], default='train', 
                        help='模式: train, test, 或 end2end')
    parser.add_argument('--part', type=str, choices=['warp', 'composition', 'all'], default='all',
                        help='运行哪个部分: warp, composition, 或 all')
    parser.add_argument('--virtual', action='store_true', help='使用虚拟数据进行测试/训练')
    parser.add_argument('--force_virtual', action='store_true', help='强制使用虚拟数据生成')
    parser.add_argument('--pretrained', type=str, help='预训练模型路径')
    parser.add_argument('--prepare_only', action='store_true', help='只准备数据，不进行训练')
    parser.add_argument('--force_prepare', action='store_true', help='强制重新生成数据，即使已存在')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备 cuda | cpu')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='指定批处理大小，覆盖配置文件中的值')
    parser.add_argument('--img_size', type=int, default=None,
                        help='指定图像大小，为None则保持原始分辨率')
    parser.add_argument('--workers', type=int, default=4, help='数据加载器的工作线程数')
    parser.add_argument('--warp_epochs', type=int, default=40, help='Warp模型的训练轮数')
    parser.add_argument('--comp_epochs', type=int, default=None,
                        help='Composition训练的轮数')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--out_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--preserve_resolution', action='store_true', help='保持图像原始分辨率，不进行缩放')
    parser.add_argument('--process_all_data', action='store_true', default=False,
                        help='处理全部数据，不进行筛选')
    parser.add_argument('--no_augment', action='store_true', help='禁用数据增强，对拼接任务很重要')
    parser.add_argument('--test_train_test', action='store_true', help='测试训练过程中的测试环节是否正常工作')
    parser.add_argument('--model_path', type=str, help='用于测试的模型路径')
    
    # 添加分布式训练参数
    parser.add_argument('--distributed', action='store_true', help='使用分布式训练')
    parser.add_argument('--world_size', type=int, default=1, help='进程总数')
    parser.add_argument('--rank', type=int, default=0, help='当前进程的排名')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:23456',
                        help='用于设置分布式训练的URL')
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help='分布式后端')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='分布式训练的本地rank')
    parser.add_argument('--use_real_data', action='store_true', default=True,
                        help='强制使用真实数据，不回退到虚拟数据')
    parser.add_argument('--data_dir', type=str, help='数据目录路径')
    parser.add_argument('--max_timeout', type=int, default=1800, help='分布式通信的最大超时时间(秒)')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp', help='禁用自动混合精度训练')
    parser.add_argument('--test_freq', type=int, default=10, help='测试频率（训练期间每隔多少轮次进行一次测试）')
    parser.add_argument('--overlap_based_stitching', action='store_true', default=False,
                   help='使用基于重叠区域的裁剪与拼接，保留原始分辨率')
    parser.add_argument('--restore_original_size', action='store_true', default=False,
                   help='测试时恢复到原始图像尺寸')
    
    # 在正式解析参数前，将命令行中的--local-rank转换为--local_rank
    import sys
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--local-rank='):
            rank_value = arg.split('=')[1]
            sys.argv[i] = f'--local_rank={rank_value}'
        elif arg == '--local-rank' and i + 1 < len(sys.argv):
            sys.argv[i] = '--local_rank'
    
    args = parser.parse_args()
    
    # 处理分布式训练环境变量
    # torch.distributed.launch使用--local-rank，而我们使用--local_rank
    # 检查环境变量以获取正确的local_rank值
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = True
        print(f"从环境变量获取local_rank: {args.local_rank}")
    
    # 读取配置文件
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {args.config} 未找到。请确保文件存在或使用--config指定正确路径。")
        return
    except json.JSONDecodeError:
        print(f"配置文件 {args.config} 格式错误。请确保它是有效的JSON文件。")
        return
    
    # 添加root_dir到配置
    config['root_dir'] = os.path.abspath('.')
    
    # 确保关键目录存在
    create_directories(config)
    
    # 检查是否有GPU可用
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} device(s).")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        
        # 打印更多GPU信息
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        print(f"当前GPU详情:")
        print(f" - 名称: {props.name}")
        print(f" - 总内存: {props.total_memory / 1e9:.2f} GB")
        print(f" - 多处理器数量: {props.multi_processor_count}")
        print(f" - CUDA算力: {props.major}.{props.minor}")
        print(f" - CUDA版本: {torch.version.cuda}")
        print(f" - cuDNN版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        print(f" - cuDNN已启用: {torch.backends.cudnn.enabled}")
        print(f" - cuDNN基准模式: {torch.backends.cudnn.benchmark}")
        
        # 启用基准模式以提高性能
        torch.backends.cudnn.benchmark = True
        print(f"已启用cuDNN基准模式，以提高性能")
    else:
        print("CUDA is not available. Will use CPU if configured.")
        # 如果没有GPU但配置使用GPU，发出警告
        if any(config[part]['train']['gpu'] != '-1' for part in ['warp', 'composition']):
            warnings.warn("配置文件指定使用GPU，但未检测到可用GPU。将尝试使用CPU。")
            # 更新配置以使用CPU
            for part in ['warp', 'composition']:
                config[part]['train']['gpu'] = '-1'
                config[part]['test']['gpu'] = '-1'
    
    # 如果指定了force_virtual参数，直接生成虚拟数据
    if args.force_virtual:
        print("Forcing virtual data generation")
        if args.mode == 'train':
            config = prepare_virtual_composition_data(config, args, mode='train')
        else:
            config = prepare_virtual_composition_data(config, args, mode='test')
        print("Virtual data generation completed")
        return
    
    # 根据模式和部分运行相应的函数
    if args.mode == 'train':
        if args.part in ['warp', 'all']:
            train_warp(args, config)
        
        # 为Composition准备数据并训练
        if args.part in ['composition', 'all'] or args.prepare_only:
            # 准备数据
            config = prepare_composition_data(config, args, mode='train', force_prepare=args.force_prepare)
            
            # 如果不是只准备数据，则进行训练
            if not args.prepare_only and args.part in ['composition', 'all']:
                # 创建Composition训练参数
                from argparse import Namespace
                train_args = Namespace()
                
                # 设置各种路径参数
                train_args.data_dir = config['composition']['train']['train_path']
                
                # 创建唯一的TensorBoard日志目录
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                train_args.log_dir = os.path.join(config['composition']['train']['summary_path'], f'run_{timestamp}')
                
                # 确保日志目录存在
                os.makedirs(train_args.log_dir, exist_ok=True)
                
                # 清除旧的日志文件
                import glob
                old_logs = glob.glob(os.path.join(config['composition']['train']['summary_path'], 'events.out.*'))
                for log in old_logs:
                    try:
                        print(f"删除旧日志文件: {log}")
                        os.remove(log)
                    except Exception as e:
                        print(f"删除旧日志文件时出错: {e}")
                
                # 添加必要的参数
                train_args.use_diffusion = True
                train_args.use_enhanced_diffusion = True
                train_args.diffusion_steps = 500  # 减少步数从1000到500
                train_args.norm_type = 'imagenet'
                train_args.vis_freq = 10
                train_args.distributed = False
                train_args.batch_size = args.batch_size if args.batch_size else 8
                train_args.img_size = args.img_size if args.img_size else 256
                train_args.epochs = args.comp_epochs if args.comp_epochs else 50
                train_args.optimizer = 'Adam'  # 修正为正确的大小写 'Adam' 而不是 'adam'
                train_args.augment = False  # 始终禁用数据增强，拼接任务不适合随机变换
                train_args.use_virtual = False if args.use_real_data else args.virtual
                train_args.num_workers = args.workers if args.workers else 2
                train_args.gpu = '0'
                train_args.sync_bn = False
                train_args.lr = 1e-4  # 设置初始学习率为1e-4
                train_args.weight_decay = 1e-5  # 添加缺失的weight_decay参数
                train_args.clip_grad = 0.5  # 添加梯度裁剪值
                train_args.grad_accum_steps = 4  # 增加梯度累积步数以提高稳定性
                train_args.scheduler = 'cosine_warmup'  # 使用余弦预热学习率调度器
                train_args.warm_up_epochs = 5  # 减少预热轮数，从20降到5
                train_args.boundary_loss_weight = 0.5  # 降低边界损失权重
                train_args.diffusion_loss_weight = 0.5  # 降低扩散损失权重
                train_args.test_freq = args.test_freq if hasattr(args, 'test_freq') else 10  # 测试频率
                train_args.resume = None  # 添加resume参数，默认不从检查点恢复
                train_args.use_amp = False  # 禁用自动混合精度训练以提高稳定性
                train_args.save_freq = 1  # 每个epoch保存一次模型
                train_args.detect_anomaly = True  # 启用异常检测
                train_args.freeze_layers = True  # 启用选择性网络冻结
                train_args.overlap_based_stitching = args.overlap_based_stitching  # 添加这一行，传递命令行参数

                # 检查是否可以使用GPU
                if torch.cuda.is_available():
                    train_args.device = 'cuda'
                else:
                    train_args.device = 'cpu'
                    
                # 打印训练参数
                print("Composition 训练参数:")
                for key, value in vars(train_args).items():
                    print(f"  {key}: {value}")
                
                # 导入并调用训练函数
                try:
                    # 创建修改后的TrainDataset类以适应训练脚本
                    import types
                    from Composition.Codes.dataset import TrainDataset as OrigTrainDataset
                    
                    # 创建自定义的TrainDataset类
                    class CustomTrainDataset(OrigTrainDataset):
                        def __init__(self, data_path, image_size=512, augment=True, norm_type='imagenet', is_test=False, use_virtual=False):
                            # 继承父类初始化
                            super().__init__(data_path, use_virtual=use_virtual, image_size=image_size, 
                                            augment=augment, norm_type=norm_type, is_test=is_test)
                            
                            # 标准图像尺寸，但仅用于不一致尺寸的对齐，不强制所有图像都调整为此尺寸
                            self.standard_size = (512, 512)  # 使用512x512而不是256x256
                            
                            # 确保转换为RGB和规范化的函数
                            self.to_tensor = transforms.ToTensor()
                            if norm_type == 'imagenet':
                                self.normalize = transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]
                                )
                            else:  # 使用 0-1 归一化
                                self.normalize = lambda x: x
                    
                        def __getitem__(self, index):
                            # 获取原始数据
                            data = super(CustomTrainDataset, self).__getitem__(index)
                            
                            # 从字典中提取数据
                            warp1 = data['warp1']
                            warp2 = data['warp2']
                            mask1 = data['mask1']
                            mask2 = data['mask2']                           

                            
                            # 确保通道数正确
                            if warp1.size(0) == 1:
                                warp1 = warp1.repeat(3, 1, 1)
                                
                            if warp2.size(0) == 1:
                                warp2 = warp2.repeat(3, 1, 1)
                            
                            # 确保掩码是单通道的
                            if mask1.size(0) > 1:
                                mask1 = mask1[:1, :, :]
                                
                            if mask2.size(0) > 1:
                                mask2 = mask2[:1, :, :]
                            
                            # 明确角色: warp2是基准图像，warp1需要被拼接到warp2上
                            base_image = warp2     # 基准图像
                            warp_image = warp1     # 需要拼接的图像
                            base_mask = mask2      # 基准图像掩码
                            warp_mask = mask1      # 需要拼接图像的掩码
                            
                            # 创建GT图像用于训练评估
                            gt = (warp_image + base_image) / 2.0
                            
                            return {
                                'base_image': base_image,
                                'warp_image': warp_image,
                                'base_mask': base_mask,
                                'warp_mask': warp_mask,
                                'gt': gt
                            }
                    
                    # 添加自定义的collate函数，确保批处理中的所有图像尺寸一致
                    def custom_collate_fn(batch):
                        # 从batch中获取第一个样本
                        if not batch:
                            return []
                        
                        # 检查第一个样本的类型
                        first_sample = batch[0]
                        is_dict_type = isinstance(first_sample, dict)
                        
                        # 为每个张量类型创建一个列表
                        base_image_list, warp_image_list, base_mask_list, warp_mask_list, gt_list = [], [], [], [], []
                        
                        # 首先确定批次中的最大高度和宽度
                        max_h, max_w = 0, 0
                        for sample in batch:
                            # 根据样本类型提取数据
                            if is_dict_type:
                                # 样本是字典类型
                                if 'base_image' in sample and 'warp_image' in sample:
                                    base_image = sample['base_image']
                                    warp_image = sample['warp_image']
                                elif 'warp1' in sample and 'warp2' in sample:
                                    # 处理老的命名方式
                                    base_image = sample['warp2']  # 基准图像
                                    warp_image = sample['warp1']  # 要拼接的图像
                                else:
                                    print("警告：样本字典缺少必要的键")
                                    print(f"可用的键: {sample.keys()}")
                                    # 创建默认值
                                    shape = (3, 256, 256)
                                    base_image = torch.zeros(shape)
                                    warp_image = torch.zeros(shape)
                            else:
                                # 尝试作为元组处理
                                try:
                                    if len(sample) >= 4:
                                        warp_image, base_image = sample[0], sample[1]
                                    else:
                                        # 创建默认值
                                        print(f"警告：元组样本长度不足, 实际长度: {len(sample)}")
                                        shape = (3, 256, 256)
                                        base_image = torch.zeros(shape)
                                        warp_image = torch.zeros(shape)
                                except Exception as e:
                                    print(f"处理样本时出错: {e}")
                                    # 创建默认值
                                    shape = (3, 256, 256)
                                    base_image = torch.zeros(shape)
                                    warp_image = torch.zeros(shape)
                            
                            # 更新最大高度和宽度
                            max_h = max(max_h, base_image.shape[1], warp_image.shape[1])
                            max_w = max(max_w, base_image.shape[2], warp_image.shape[2])
                        
                        # 将最大高度和宽度调整为32的倍数，以便更好地处理网络下采样
                        max_h = ((max_h + 31) // 32) * 32
                        max_w = ((max_w + 31) // 32) * 32
                        
                        if max_h > 1024 or max_w > 1024:
                            print(f"警告: 批次尺寸过大 ({max_h}x{max_w})，可能导致内存不足。考虑降低批次大小。")
                            # 强制限制最大尺寸为1024x1024
                            max_h = min(max_h, 1024)
                            max_w = min(max_w, 1024)
                            print(f"已限制批次尺寸为: {max_h}x{max_w}")
                        
                        # 遍历batch中的每个样本，进行填充
                        for sample in batch:
                            # 根据样本类型提取数据
                            if is_dict_type:
                                # 样本是字典类型
                                if 'base_image' in sample and 'warp_image' in sample:
                                    base_image = sample['base_image']
                                    warp_image = sample['warp_image']
                                    base_mask = sample.get('base_mask', torch.ones((1, base_image.shape[1], base_image.shape[2])))
                                    warp_mask = sample.get('warp_mask', torch.ones((1, warp_image.shape[1], warp_image.shape[2])))
                                    gt = sample.get('gt', (base_image + warp_image) / 2.0)
                                elif 'warp1' in sample and 'warp2' in sample:
                                    # 处理老的命名方式
                                    warp_image = sample['warp1']
                                    base_image = sample['warp2']
                                    warp_mask = sample.get('mask1', torch.ones((1, warp_image.shape[1], warp_image.shape[2])))
                                    base_mask = sample.get('mask2', torch.ones((1, base_image.shape[1], base_image.shape[2])))
                                    gt = (warp_image + base_image) / 2.0
                                else:
                                    # 创建默认值
                                    shape = (3, 256, 256)
                                    base_image = torch.zeros(shape)
                                    warp_image = torch.zeros(shape)
                                    base_mask = torch.ones((1, shape[1], shape[2]))
                                    warp_mask = torch.ones((1, shape[1], shape[2]))
                                    gt = torch.zeros(shape)
                            else:
                                # 尝试作为元组处理
                                try:
                                    if len(sample) >= 7:
                                        # 元组格式: img1, img2, mask1, mask2, gt, warp1, warp2
                                        img1, img2, mask1, mask2, gt, warp1, warp2 = sample[:7]
                                except Exception as e:
                                    print(f"处理样本时出错: {e}")
                                    continue
                            
                            # 确保掩码是单通道的
                            if base_mask.dim() == 2:
                                base_mask = base_mask.unsqueeze(0)
                            if warp_mask.dim() == 2:
                                warp_mask = warp_mask.unsqueeze(0)
                            
                            # 创建有效区域掩码 - 用于标识非填充区域
                            valid_h, valid_w = base_image.shape[1], base_image.shape[2]
                            valid_mask = torch.zeros(1, max_h, max_w)
                            valid_mask[:, :valid_h, :valid_w] = 1.0
                            
                            # 填充图像到统一大小
                            base_image_padded = pad_tensor(base_image, max_h, max_w)
                            warp_image_padded = pad_tensor(warp_image, max_h, max_w)
                            base_mask_padded = pad_tensor(base_mask, max_h, max_w) * valid_mask  # 使用有效区域掩码
                            warp_mask_padded = pad_tensor(warp_mask, max_h, max_w) * valid_mask  # 使用有效区域掩码
                            gt_padded = pad_tensor(gt, max_h, max_w)
                            
                            # 添加到对应列表
                            base_image_list.append(base_image_padded)
                            warp_image_list.append(warp_image_padded)
                            base_mask_list.append(base_mask_padded)
                            warp_mask_list.append(warp_mask_padded)
                            gt_list.append(gt_padded)
                        
                        # 使用torch.stack合并处理后的张量
                        base_image_batch = torch.stack(base_image_list, dim=0)
                        warp_image_batch = torch.stack(warp_image_list, dim=0)
                        base_mask_batch = torch.stack(base_mask_list, dim=0)
                        warp_mask_batch = torch.stack(warp_mask_list, dim=0)
                        gt_batch = torch.stack(gt_list, dim=0)
                        
                        # 返回字典形式的批次数据
                        return {
                            'base_image': base_image_batch,
                            'warp_image': warp_image_batch,
                            'base_mask': base_mask_batch,
                            'warp_mask': warp_mask_batch,
                            'gt': gt_batch
                        }
                    
                    # 辅助函数：填充张量到指定大小
                    def pad_tensor(tensor, target_h, target_w):
                        # 获取当前张量维度
                        c, h, w = tensor.shape
                        
                        # 计算需要填充的量
                        pad_h = target_h - h
                        pad_w = target_w - w
                        
                        # 如果不需要填充，直接返回
                        if pad_h <= 0 and pad_w <= 0:
                            return tensor
                        
                        # 计算填充量 (左, 右, 上, 下)
                        padding = (0, pad_w, 0, pad_h)
                        
                        # 使用常量0进行填充
                        padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
                        
                        return padded_tensor
                    
                    # 替换Composition.Codes.train中使用的TrainDataset
                    import sys
                    from Composition.Codes import train
                    train.TrainDataset = CustomTrainDataset
                    
                    # 保存原始的DataLoader创建方法
                    original_dataloader = torch.utils.data.DataLoader
                    
                    # 修改DataLoader，使其使用我们的自定义collate_fn
                    def patched_dataloader(*args, **kwargs):
                        # 将自定义collate_fn添加到kwargs中
                        if 'collate_fn' not in kwargs:
                            kwargs['collate_fn'] = custom_collate_fn
                        return original_dataloader(*args, **kwargs)
                    
                    # 替换torch.utils.data中的DataLoader
                    torch.utils.data.DataLoader = patched_dataloader
                    
                    # 导入并调用训练函数
                    from Composition.Codes.train import train
                    from torch.utils.data import DataLoader
                    
                    # 训练模型
                    gpu_id = int(train_args.gpu) if train_args.gpu != '-1' else 0
                    
                    # 修正train函数的调用，提供所有需要的参数
                    if hasattr(train_args, 'distributed') and train_args.distributed:
                        world_size = train_args.world_size if hasattr(train_args, 'world_size') else 1
                    else:
                        world_size = 1
                    
                    # 为分布式训练正确设置设备
                    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
                    
                    # 创建自定义数据集和修改DataLoader逻辑
                    train_dataset = CustomTrainDataset(
                        train_args.data_dir,
                        image_size=train_args.img_size,
                        augment=True,
                        norm_type=train_args.norm_type,
                        is_test=False,
                        use_virtual=train_args.use_virtual if hasattr(train_args, 'use_virtual') else False
                    )

                    # 使用自定义的collate_fn
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=train_args.batch_size,
                        shuffle=True,
                        num_workers=train_args.num_workers,
                        collate_fn=custom_collate_fn
                    )
                    
                    # 使用更新后的train_Composition函数，而不是直接调用导入的train函数
                    # 这样可以使用我们实现的所有改进和修复
                    print("使用改进的Composition训练函数...")
                    # 将train_args转换为合适的调试参数
                    train_Composition(config, debug_mode=train_args)
                
                except Exception as e:
                    print(f"Error in Composition training setup: {e}")
                    import traceback
                    traceback.print_exc()
                
    elif args.mode == 'test':
        if args.part in ['warp', 'all']:
            test_warp(args, config)
            
        # 为Composition准备测试数据并测试
        if args.part in ['composition', 'all'] or args.prepare_only:
            # 准备数据
            config = prepare_composition_data(config, args, mode='test', force_prepare=args.force_prepare)
            
            # 如果不是只准备数据，则进行测试
            if not args.prepare_only and args.part in ['composition', 'all']:
                config = test_composition(args, config)
                
    elif args.mode == 'end2end':
        end_to_end_test(args, config)
    
if __name__ == '__main__':
    main() 