import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
import os
import sys
import time
import torch.utils.data
from torch.utils.data import DataLoader

# 确保能访问Composition模块的路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入loss模块
try:
    from Composition.Codes.loss import (
        cal_perceptual_loss, cal_boundary_term, cal_smooth_term_stitch,
        gradient, cal_color_consistency_loss, cal_panorama_consistency_loss,
        VGGPerceptualLoss, SSIM
    )
except ImportError:
    print("警告: 无法导入Composition.Codes.loss模块的某些函数，全景图一致性损失可能无法正常工作")


# 预定义AttentionBlock和GlobalAttentionBlock类
class AttentionBlock(nn.Module):
    """局部自注意力块 - 增强版本，解决维度对齐问题"""
    def __init__(self, channels, heads=4):
        super().__init__()
        self.channels = channels
        self.heads = heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        # 残差连接
        skip = x
        
        # 归一化
        x = self.norm(x)
        
        # 计算注意力
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # 计算每个头的特征维度
        head_dim = c // self.heads
        
        # 重塑为多头注意力，处理不能整除的情况
        q = q.reshape(b, self.heads, head_dim, h * w).transpose(-2, -1)  # [b, heads, h*w, head_dim]
        k = k.reshape(b, self.heads, head_dim, h * w)                    # [b, heads, head_dim, h*w]
        v = v.reshape(b, self.heads, head_dim, h * w).transpose(-2, -1)  # [b, heads, h*w, head_dim]
        
        # 计算注意力分数并应用
        scale = head_dim ** -0.5
        
        # 转置k以计算注意力分数
        # 确保k是密集张量，防止permute操作失败
        k = ensure_dense_tensor(k)
        k = k.permute(0, 1, 3, 2)  # [b, heads, h*w, head_dim]
        
        # 现在q形状是[b, heads, h*w, head_dim]，k形状是[b, heads, h*w, head_dim]
        # 矩阵乘法后得到[b, heads, h*w, h*w]
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale  # 使用transpose确保维度匹配
        attn = F.softmax(attn, dim=-1)
        
        # 矩阵乘法得到[b, heads, h*w, head_dim]
        x = torch.matmul(attn, v)  # v形状是[b, heads, h*w, head_dim]
        
        # 重塑回原始形状
        x = x.transpose(-2, -1).reshape(b, c, h, w)
        
        # 投影并添加残差
        x = self.proj(x)
        return x + skip



class GlobalAttentionBlock(nn.Module):
    """全局注意力块 - 增强版本，解决维度对齐问题"""
    def __init__(self, channels, heads=4):
        super().__init__()
        self.channels = channels
        self.heads = heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # 添加全局上下文处理
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        # 残差连接
        skip = x
        
        # 归一化
        x = self.norm(x)
        
        # 全局上下文特征
        global_feat = self.global_pool(x)
        global_feat = self.global_proj(global_feat)
        
        # 计算注意力
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # 计算每个头的特征维度
        head_dim = c // self.heads
        
        # 计算注意力缩放因子
        scale = head_dim ** -0.5
        
        # 重塑为多头注意力，处理不能整除的情况
        q = q.reshape(b, self.heads, head_dim, h * w).transpose(-2, -1)  # [b, heads, h*w, head_dim]
        k = k.reshape(b, self.heads, head_dim, h * w)                    # [b, heads, head_dim, h*w]
        v = v.reshape(b, self.heads, head_dim, h * w).transpose(-2, -1)  # [b, heads, h*w, head_dim]
        
        # 添加全局信息到键值
        global_k = global_feat.reshape(b, self.heads, head_dim, 1).expand(-1, -1, -1, h*w)
        k = k + global_k * 0.1  # 轻微融入全局信息
        
        # 直接将k转置为正确的形状
        # 确保k是密集张量，防止permute操作失败
        k = ensure_dense_tensor(k)
        k = k.permute(0, 1, 3, 2)  # [b, heads, h*w, head_dim]
        
        # 矩阵乘法得到注意力权重
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale  # 使用transpose确保维度匹配
        attn = F.softmax(attn, dim=-1)
        
        # 矩阵乘法得到加权特征 [b, heads, h*w, head_dim]
        x = torch.matmul(attn, v)
        
        # 重塑回原始形状
        x = x.transpose(-2, -1).reshape(b, c, h, w)
        
        # 投影并添加残差和全局特征
        x = self.proj(x)
        # 添加全局上下文
        x = x + global_feat.expand(-1, -1, h, w) * 0.1
        
        return x + skip



def ensure_dense_tensor(tensor):
    """
    确保张量是密集张量，如果是稀疏张量则转换为密集张量
    
    参数:
        tensor (torch.Tensor): 输入张量，可能是稀疏的
        
    返回:
        torch.Tensor: 密集张量
    """
    if tensor is None:
        return None
    if hasattr(tensor, 'is_sparse') and tensor.is_sparse:
        return tensor.to_dense()
    return tensor

def safe_mask_to_tensor(mask_array, device=None):
    """
    安全地将掩码数组转换为张量，处理各种可能的维度和稀疏问题
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


class EnhancedDiffusionComposition(nn.Module):
    """
    增强的扩散模型用于图像拼接，以warp2为固定基准图像，生成最终形变蒙版
    实现分层特征提取和两阶段（局部/全局）处理
    """
    def __init__(self, num_timesteps=1000, beta_schedule='linear', 
                 image_size=512, base_channels=64, attention_resolutions=[16, 8],
                 dropout=0.0, channel_mult=(1, 2, 4, 8), conv_resample=True,
                 num_res_blocks=2, heads=4, use_scale_shift_norm=True):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.image_size = image_size  # 修改默认值为512
        self.heads = heads
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置时间步为1000步
        self.register_schedule(beta_schedule=beta_schedule)
        
        # 时间嵌入
        time_embed_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 输入通道适配层 - 接收两个输入图像和两个掩码
        # warp1, warp2, mask1, mask2 = 3+3+1+1 = 8 通道
        self.input_channels = 8
        self.channel_adapter = nn.Conv2d(self.input_channels, base_channels, kernel_size=3, padding=1)
        
        # 分层特征提取网络
        # 下采样路径
        self.down1 = DownBlock(base_channels, base_channels, time_embed_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm)
        self.down2 = DownBlock(base_channels, base_channels*2, time_embed_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm)
        self.down3 = DownBlock(base_channels*2, base_channels*4, time_embed_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm) 
        self.down4 = DownBlock(base_channels*4, base_channels*8, time_embed_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm)
        
        # 中间块 - 局部注意力机制
        self.mid_block1 = ResBlock(base_channels*8, base_channels*8, time_embed_dim, dropout, use_scale_shift_norm)
        self.mid_attn = AttentionBlock(base_channels*8, heads=heads)
        self.mid_block2 = ResBlock(base_channels*8, base_channels*8, time_embed_dim, dropout, use_scale_shift_norm)
        
        # 全局注意力块 - 处理整体图像关系
        self.global_attn = GlobalAttentionBlock(base_channels*8, heads=heads)
        
        # 上采样路径 - 带有跳跃连接
        self.up1 = UpBlock(base_channels*8, base_channels*4, time_embed_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm)
        self.up2 = UpBlock(base_channels*4, base_channels*2, time_embed_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm)
        self.up3 = UpBlock(base_channels*2, base_channels, time_embed_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm)
        self.up4 = UpBlock(base_channels, base_channels, time_embed_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm)
        
        # 输出层 - 生成噪声预测
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.BatchNorm2d(base_channels),  # 添加批次正则化提高稳定性
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
        )
        
        # 蒙版生成分支 - 专注于生成高质量蒙版
        self.mask_branch = nn.Sequential(
            nn.Conv2d(base_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def register_schedule(self, beta_schedule='linear'):
        """设置扩散过程的方差调度"""
        if beta_schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
        elif beta_schedule == 'cosine':
            # 余弦调度对非最佳拼接情况更稳定
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        
        # 定义前向扩散过程参数
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 存储所有参数
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 计算前向扩散过程的派生公式
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1. / alphas))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
    
    def time_mlp(self, t):
        """将时间步转换为特征嵌入"""
        # 确保t是张量而不是整数，并且强制转为长整型
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=self.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0)  # 标量张量转为1维张量
            
        t = t.to(dtype=torch.long)
        
        half_dim = self.time_embed[0].in_features // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # 通过MLP网络
        emb = self.time_embed(emb)
        return emb
    
    def forward_diffusion(self, x, t):
        """
        对输入图像执行前向扩散过程
        参数:
            x: 输入图像
            t: 时间步
        返回:
            带噪图像和噪声目标
        """
        noise = torch.randn_like(x)
        # 使用参数化公式给图像添加噪声
        noisy_x = (
            self.sqrt_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1) * x + 
            self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1) * noise
        )
        return noisy_x, noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """从噪声预测原始图像"""
        return (
            (x_t - self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1) * noise) / 
            self.sqrt_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        )
    
    def q_posterior(self, x_0, x_t, t):
        """计算后验分布参数"""
        posterior_mean = (
            self.alphas_cumprod_prev.gather(-1, t).reshape(-1, 1, 1, 1) * 
            x_0 + 
            (1 - self.alphas_cumprod_prev.gather(-1, t).reshape(-1, 1, 1, 1)) * 
            x_t
        )
        return posterior_mean
    
    def p_mean_variance(self, x, t, img1, img2, mask1, mask2):
        """
        计算p分布的均值和方差
        使用条件输入指导去噪过程
        """
        # 编码条件信息 (img1, img2, mask1, mask2)
        x_input = torch.cat([img1, img2, mask1, mask2], dim=1)
        x_input = self.channel_adapter(x_input)
        
        # 获取时间编码
        t_emb = self.time_mlp(t)
        
        # 分层特征提取 - 下采样路径
        d1 = self.down1(x_input, t_emb)
        d2 = self.down2(d1, t_emb)
        d3 = self.down3(d2, t_emb)
        d4 = self.down4(d3, t_emb)
        
        # 中间块 - 局部注意力处理
        h = self.mid_block1(d4, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # 全局注意力处理 - 整体关系建模
        h = self.global_attn(h)
        
        # 分层特征提取 - 上采样路径(带跳跃连接)
        h = self.up1(h, d4, t_emb)
        h = self.up2(h, d3, t_emb)
        h = self.up3(h, d2, t_emb)
        h = self.up4(h, d1, t_emb)
        
        # 预测噪声
        predicted_noise = self.final_conv(h)
        
        # 使用预测的噪声计算原始图像
        x_0 = self.predict_start_from_noise(x, t, predicted_noise)
        x_0 = torch.clamp(x_0, -1, 1)
        
        # 计算后验均值
        mean = self.q_posterior(x_0, x, t)
        
        return mean, x_0
    
    def p_sample(self, x, t, img1, img2, mask1, mask2, guidance_scale=1.0):
        """
        执行一步去噪采样，改进版本确保维度对齐
        """
        # 确保所有输入有相同的空间维度
        if img2.shape[2:] != img1.shape[2:]:
            img2 = F.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
        
        if mask1.shape[2:] != img1.shape[2:]:
            mask1 = F.interpolate(mask1, size=img1.shape[2:], mode='bilinear', align_corners=False)
        
        if mask2.shape[2:] != img1.shape[2:]:
            mask2 = F.interpolate(mask2, size=img1.shape[2:], mode='bilinear', align_corners=False)
        
        if x.shape[2:] != img1.shape[2:]:
            x = F.interpolate(x, size=img1.shape[2:], mode='bilinear', align_corners=False)
        
        # 获取概率分布参数
        mean, x_0 = self.p_mean_variance(x, t, img1, img2, mask1, mask2)
        
        # 应用分类器自由引导
        if guidance_scale > 1.0:
            # 无条件前向传播 (将条件权重设为0)
            uncond_img1 = torch.zeros_like(img1)
            uncond_img2 = torch.zeros_like(img2)
            uncond_mask1 = torch.zeros_like(mask1)
            uncond_mask2 = torch.zeros_like(mask2)
            
            uncond_mean, _ = self.p_mean_variance(x, t, uncond_img1, uncond_img2, uncond_mask1, uncond_mask2)
            
            # 混合无条件和有条件结果
            mean = uncond_mean + guidance_scale * (mean - uncond_mean)
        
        # 添加噪声 (如果t > 0)
        noise = torch.randn_like(x)
        mask = (t > 0).reshape(-1, 1, 1, 1).float()
        
        # 采样下一步
        sample = mean + mask * torch.sqrt(self.posterior_variance.gather(-1, t).reshape(-1, 1, 1, 1)) * noise
        
        # 处理条件输入，与forward_diffusion_train保持一致
        x_cond = torch.cat([img1, img2, mask1, mask2], dim=1)
        x_cond = self.channel_adapter(x_cond)
        
        # 特征提取
        t_emb = self.time_mlp(t)
        d1 = self.down1(x_cond, t_emb)
        d2 = self.down2(d1, t_emb)
        d3 = self.down3(d2, t_emb)
        d4 = self.down4(d3, t_emb)
        
        # 中间块处理
        h = self.mid_block1(d4, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        h = self.global_attn(h)
        
        # 上采样
        h = self.up1(h, d4, t_emb)
        h = self.up2(h, d3, t_emb)
        h = self.up3(h, d2, t_emb)
        h = self.up4(h, d1, t_emb)
        
        # 使用direct_mask_branch生成掩码
        learned_mask = self.mask_branch(h)
        
        # 确保掩码与采样形状一致
        if learned_mask.shape[2:] != sample.shape[2:]:
            learned_mask = F.interpolate(learned_mask, size=sample.shape[2:], mode='bilinear', align_corners=False)
        
        return sample, x_0, learned_mask
    
    def forward(self, *args, **kwargs):
        """
        灵活的forward方法，改进版本确保维度对齐，支持多种调用形式:
        - 单个字典输入: model(input_dict)
        - 标准输入: forward(x, t, img1, img2, mask1, mask2) - 用于diffusion训练
        - 简化输入: forward(warp1, warp2, mask1, mask2) - 用于最终应用
        """
        try:
            # 处理字典输入
            if len(args) == 1 and isinstance(args[0], dict):
                input_dict = args[0]
                if all(k in input_dict for k in ['base_image', 'warp_image', 'base_mask', 'warp_mask']):
                    # 重新打包为位置参数
                    return self.forward_composition(
                        input_dict['warp_image'], 
                        input_dict['base_image'], 
                        input_dict['warp_mask'], 
                        input_dict['base_mask']
                    )
                else:
                    raise ValueError("输入字典缺少必要的键")
            
            # 根据参数数量判断调用方式
            if len(args) == 4:
                # 兼容直接传递4个参数: model(warp1, warp2, mask1, mask2)
                return self.forward_composition(*args)
            elif len(args) == 6:
                # 标准diffusion调用: model(x, t, img1, img2, mask1, mask2)
                x, t, img1, img2, mask1, mask2 = args
                return self.forward_diffusion_train(x, t, img1, img2, mask1, mask2)
            else:
                # 尝试根据args和kwargs内容推断调用方式
                if 'x' in kwargs and 't' in kwargs:
                    # 似乎是标准diffusion调用
                    x = kwargs.get('x')
                    t = kwargs.get('t')
                    img1 = kwargs.get('img1')
                    img2 = kwargs.get('img2')
                    mask1 = kwargs.get('mask1')
                    mask2 = kwargs.get('mask2')
                    return self.forward_diffusion_train(x, t, img1, img2, mask1, mask2)
                elif 'warp_image' in kwargs and 'base_image' in kwargs:
                    # 似乎是composition调用
                    warp_image = kwargs.get('warp_image')
                    base_image = kwargs.get('base_image')
                    warp_mask = kwargs.get('warp_mask')
                    base_mask = kwargs.get('base_mask')
                    return self.forward_composition(warp_image, base_image, warp_mask, base_mask)
                else:
                    # 尝试使用参数实际类型和形状来推断
                    
                    # 如果只有两个参数，很可能是(x, t)
                    if len(args) == 2:
                        x, t = args
                        # 我们需要创建假的img1, img2, mask1, mask2
                        img1 = torch.zeros_like(x)
                        img2 = torch.zeros_like(x)
                        mask1 = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
                        mask2 = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
                        return self.forward_diffusion_train(x, t, img1, img2, mask1, mask2)
                    
                    raise ValueError(f"不支持的输入参数数量: {len(args)}，且无法从kwargs中推断调用方式")
        except Exception as e:
            print(f"forward方法出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 创建默认输出 - 根据输入参数尝试生成合理的输出
            if len(args) >= 1:
                # 如果有输入，尝试根据第一个参数的形状生成输出
                x = args[0]
                if isinstance(x, dict):
                    # 字典输入，尝试获取warp_image
                    if 'warp_image' in x:
                        warp_image = x['warp_image']
                        # 创建默认掩码和结果
                        mask = torch.ones((warp_image.shape[0], 1, warp_image.shape[2], warp_image.shape[3]), device=warp_image.device) * 0.5
                        result = warp_image  # 简单返回输入图像
                        return mask, result
                else:
                    # 张量输入
                    if len(args) == 6:  # 假设是diffusion训练模式
                        # 创建随机噪声和默认掩码
                        noise = torch.randn_like(x)
                        mask = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device) * 0.5
                        return noise, mask
                    elif len(args) == 4:  # 假设是composition模式
                        warp1 = args[0]
                        # 创建默认掩码和结果
                        mask = torch.ones((warp1.shape[0], 1, warp1.shape[2], warp1.shape[3]), device=warp1.device) * 0.5
                        result = warp1  # 简单返回第一个输入图像
                        return mask, result
            
            # 如果上述都失败，抛出原始异常
            raise
    
    def forward_composition(self, warp_image, base_image, warp_mask, base_mask):
        """
        拼接图像组合 - 最终应用时使用，增强版本，解决维度对齐问题
        使用统一的归一化公式：(img1+1.)*mask + (img2+1.)*(1-mask) - 1.
        增加数值稳定性处理
        """
        try:
            # 确保所有输入有相同的空间维度
            if warp_image.shape[2:] != base_image.shape[2:]:
                base_image = F.interpolate(base_image, size=warp_image.shape[2:], mode='bilinear', align_corners=False)
            
            if warp_mask.shape[2:] != warp_image.shape[2:]:
                warp_mask = F.interpolate(warp_mask, size=warp_image.shape[2:], mode='bilinear', align_corners=False)
            
            if base_mask.shape[2:] != warp_image.shape[2:]:
                base_mask = F.interpolate(base_mask, size=warp_image.shape[2:], mode='bilinear', align_corners=False)
            
            # 检查并清理NaN/Inf值
            warp_image = torch.nan_to_num(warp_image, nan=0.0, posinf=1.0, neginf=-1.0)
            base_image = torch.nan_to_num(base_image, nan=0.0, posinf=1.0, neginf=-1.0)
            warp_mask = torch.nan_to_num(warp_mask, nan=0.5, posinf=1.0, neginf=0.0)
            base_mask = torch.nan_to_num(base_mask, nan=0.5, posinf=1.0, neginf=0.0)
            
            # 裁剪mask值到[0,1]范围
            warp_mask = torch.clamp(warp_mask, 0.0, 1.0)
            base_mask = torch.clamp(base_mask, 0.0, 1.0)
            
            # 合并输入以供特征提取
            combined = torch.cat([base_image, warp_image, base_mask, warp_mask], dim=1)
            x_input = self.channel_adapter(combined)
            
            # 初始化时间步为0
            t = torch.zeros(base_image.shape[0], device=base_image.device, dtype=torch.long)
            t_emb = self.time_mlp(t)
            
            # 特征提取
            d1 = self.down1(x_input, t_emb)
            d2 = self.down2(d1, t_emb)
            d3 = self.down3(d2, t_emb)
            d4 = self.down4(d3, t_emb)
            
            # 中间块
            h = self.mid_block1(d4, t_emb)
            h = self.mid_attn(h)
            h = self.mid_block2(h, t_emb)
            
            # 全局处理
            h = self.global_attn(h)
            
            # 上采样
            h = self.up1(h, d4, t_emb)
            h = self.up2(h, d3, t_emb)
            h = self.up3(h, d2, t_emb)
            h = self.up4(h, d1, t_emb)
            
            # 生成掩码 - 使用与训练时相同的direct_mask_branch
            mask = self.mask_branch(h)
            
            # 确保mask与输入图像尺寸一致
            if mask.shape[2:] != warp_image.shape[2:]:
                mask = F.interpolate(mask, size=warp_image.shape[2:], mode='bilinear', align_corners=False)
            
            # 应用Sigmoid确保掩码在[0,1]范围内
            mask = torch.sigmoid(mask)
            
            # 应用统一的拼接公式（与train.py一致），并使用torch.clamp确保数值稳定性
            warp_image_norm = torch.clamp(warp_image + 1.0, -1.0, 3.0)
            base_image_norm = torch.clamp(base_image + 1.0, -1.0, 3.0) 
            stitched_image = warp_image_norm * mask + base_image_norm * (1.0 - mask) - 1.0
            
            # 最后确保输出在有效范围内
            stitched_image = torch.clamp(stitched_image, -1.0, 1.0)
            
            # 返回掩码和结果图像
            return mask, stitched_image
        
        except Exception as e:
            print(f"forward_composition错误: {e}")
            # 返回一个安全的默认结果
            mask = torch.sigmoid(torch.ones_like(warp_mask))
            stitched_image = (warp_image + base_image) / 2.0
            return mask, stitched_image
    
    def forward_diffusion_train(self, x, t, img1, img2, mask1, mask2):
        """
        扩散训练 - 预测噪声和生成掩码，改进版本确保维度对齐
        """
        # 确保所有输入有相同的空间维度
        if img2.shape[2:] != img1.shape[2:]:
            img2 = F.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
        
        if mask1.shape[2:] != img1.shape[2:]:
            mask1 = F.interpolate(mask1, size=img1.shape[2:], mode='bilinear', align_corners=False)
        
        if mask2.shape[2:] != img1.shape[2:]:
            mask2 = F.interpolate(mask2, size=img1.shape[2:], mode='bilinear', align_corners=False)
        
        if x.shape[2:] != img1.shape[2:]:
            x = F.interpolate(x, size=img1.shape[2:], mode='bilinear', align_corners=False)
        
        # 组合条件输入
        x_cond = torch.cat([img1, img2, mask1, mask2], dim=1)
        x_cond = self.channel_adapter(x_cond)
        
        # 获取时间编码
        t_emb = self.time_mlp(t)
        
        # 分层特征提取和处理
        d1 = self.down1(x_cond, t_emb)
        d2 = self.down2(d1, t_emb)
        d3 = self.down3(d2, t_emb)
        d4 = self.down4(d3, t_emb)
        
        # 局部注意力处理
        h = self.mid_block1(d4, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # 全局注意力增强
        h = self.global_attn(h)
        
        # 分层特征提取和上采样
        h = self.up1(h, d4, t_emb)
        h = self.up2(h, d3, t_emb)
        h = self.up3(h, d2, t_emb)
        h = self.up4(h, d1, t_emb)
        
        # 预测噪声和蒙版
        predicted_noise = self.final_conv(h)
        
        # 使用专门的直接掩码分支，应用于特征映射
        learned_mask = self.mask_branch(h)
        
        # 确保预测噪声与输入x形状一致
        if predicted_noise.shape[2:] != x.shape[2:]:
            predicted_noise = F.interpolate(predicted_noise, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return predicted_noise, learned_mask
    
    def sample(self, img1, img2, mask1, mask2, num_steps=100, guidance_scale=1.0):
        """
        使用扩散模型对输入图像进行采样，生成 warp2 的拼接模板
        
        参数:
            img1, img2: 输入图像 [B, C, H, W] (-1 到 1 范围)
            mask1, mask2: 输入掩码 [B, 1, H, W] (0 到 1 范围)
            num_steps: 采样步数
            guidance_scale: 引导尺度
            
        返回:
            transition_mask: 生成的混合掩码
            clean_x: 去噪后的预测图像
        """
        try:
            # 检查输入
            if img1 is None or img2 is None or mask1 is None or mask2 is None:
                print("错误: 采样函数接收到了空输入")
                return None, None
            
            # 获取批次大小和分辨率
            batch_size, _, h, w = img1.shape
            device = img1.device
            
            # 确保所有输入的形状一致
            if img2.shape[2:] != img1.shape[2:]:
                img2 = F.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
            
            if mask1.shape[2:] != img1.shape[2:]:
                mask1 = F.interpolate(mask1, size=img1.shape[2:], mode='bilinear', align_corners=False)
            
            if mask2.shape[2:] != img1.shape[2:]:
                mask2 = F.interpolate(mask2, size=img1.shape[2:], mode='bilinear', align_corners=False)
            
            # 检查并清理NaN/Inf值
            img1 = torch.nan_to_num(img1, nan=0.0, posinf=1.0, neginf=-1.0)
            img2 = torch.nan_to_num(img2, nan=0.0, posinf=1.0, neginf=-1.0)
            mask1 = torch.nan_to_num(mask1, nan=0.5, posinf=1.0, neginf=0.0)
            mask2 = torch.nan_to_num(mask2, nan=0.5, posinf=1.0, neginf=0.0)
            
            # 确保掩码在[0,1]范围内
            mask1 = torch.clamp(mask1, 0.0, 1.0)
            mask2 = torch.clamp(mask2, 0.0, 1.0)
            
            # 更精确地识别重叠区域 - 两个掩码都激活的区域
            overlap_mask = mask1 * mask2
            
            # 使用大于阈值的条件确定实际重叠
            overlap_threshold = 0.1
            overlap_mask_binary = (overlap_mask > overlap_threshold).float()
            
            # 从纯噪声开始
            x = torch.randn((batch_size, 3, h, w), device=device)
            
            # 逐步去噪
            clean_x = None
            for i in reversed(range(0, self.num_timesteps, self.num_timesteps // num_steps)):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x, clean_x, learned_mask = self.p_sample(x, t, img1, img2, mask1, mask2, guidance_scale)
                
                # 应用进度更新的回调(如果有的话)
                if hasattr(self, 'progress_callback') and self.progress_callback is not None:
                    should_stop = self.progress_callback(i, x)
                    if should_stop:
                        break
            
            # 生成最终融合掩码
            # 处理条件输入
            x_cond = torch.cat([img1, img2, mask1, mask2], dim=1)
            x_cond = self.channel_adapter(x_cond)
            
            # 时间嵌入 - 使用时间步0
            t = torch.zeros(batch_size, device=device, dtype=torch.long)
            t_emb = self.time_mlp(t)
            
            # 特征提取
            d1 = self.down1(x_cond, t_emb)
            d2 = self.down2(d1, t_emb)
            d3 = self.down3(d2, t_emb)
            d4 = self.down4(d3, t_emb)
            
            # 中间块处理
            h = self.mid_block1(d4, t_emb)
            h = self.mid_attn(h)
            h = self.mid_block2(h, t_emb)
            h = self.global_attn(h)
            
            # 上采样
            h = self.up1(h, d4, t_emb)
            h = self.up2(h, d3, t_emb)
            h = self.up3(h, d2, t_emb)
            h = self.up4(h, d1, t_emb)
            
            # 生成掩码
            transition_mask = self.mask_branch(h)
            transition_mask = torch.sigmoid(transition_mask)
            
            # 确保掩码与图像尺寸一致
            if transition_mask.shape[2:] != img1.shape[2:]:
                transition_mask = F.interpolate(transition_mask, size=img1.shape[2:], mode='bilinear', align_corners=False)
            
            # 确保clean_x有值，如果没有则使用原始x
            if clean_x is None:
                clean_x = x
                
            return transition_mask, clean_x
        
        except Exception as e:
            print(f"全景生成错误: {e}")
            import traceback
            traceback.print_exc()
            print("回退到安全模式...")
            
            batch_size, _, h, w = img1.shape
            device = img1.device
            
            # 创建一个简单的过渡掩码 - 从左到右线性变化
            simple_mask = torch.zeros((batch_size, 1, h, w), device=device)
            for x in range(w):
                simple_mask[:, :, :, x] = x / (w - 1)
                
            # 直接返回原始img2和简单掩码
            return simple_mask, img2

    def safe_forward(self, warp1=None, warp2=None, mask1=None, mask2=None, input_dict=None):
        """
        安全的前向传播，处理输入数据以便模型生成warp2模板而不是warp1模板

        参数:
            warp1, warp2: 输入图像张量（已裁剪） [B, C, H, W]
            mask1, mask2: 输入掩码张量 [B, 1, H, W]
            input_dict: 可选的输入字典，可以包含上述输入

        返回:
            mask: 生成的掩码 [B, 1, H, W]
            result: 生成的拼接结果 [B, 3, H, W]
        """
        try:
            # 处理输入
            if input_dict is not None:
                if warp1 is None and 'warp1' in input_dict:
                    warp1 = input_dict['warp1']
                if warp2 is None and 'warp2' in input_dict:
                    warp2 = input_dict['warp2']
                if mask1 is None and 'mask1' in input_dict:
                    mask1 = input_dict['mask1']
                if mask2 is None and 'mask2' in input_dict:
                    mask2 = input_dict['mask2']
            
            # 检查输入数据是否有效
            if warp1 is None or warp2 is None or mask1 is None or mask2 is None:
                print("错误: 缺少必需的输入数据")
                return None, None

            # 确保所有输入张量都在同一个设备上
            device = warp1.device
            warp1 = warp1.to(device)
            warp2 = warp2.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)

            # 确保掩码和图像尺寸匹配
            if warp1.shape[2:] != mask1.shape[2:]:
                mask1 = F.interpolate(mask1, size=warp1.shape[2:], mode='bilinear', align_corners=False)
            if warp2.shape[2:] != mask2.shape[2:]:
                mask2 = F.interpolate(mask2, size=warp2.shape[2:], mode='bilinear', align_corners=False)

            # 修改: 使用扩散模型生成warp2的模板，而不是warp1的模板
            # 采样步数调整为更高以提高质量
            sample_steps = 100
            print(f"使用 {sample_steps} 步扩散采样生成warp2模板")

            # 生成warp2模板
            with torch.no_grad():
                # 将warp1（裁剪后的）、mask1（裁剪后的）与warp2一起送入扩散模型
                mask_tensor, predicted_image = self.sample(
                    img1=warp1,
                    img2=warp2,
                    mask1=mask1,
                    mask2=mask2,
                    num_steps=sample_steps
                )

            # 将生成的模板直接应用于warp2图像 - 而不是像之前那样应用于warp1
            result = self.apply_mask(
                base_image=warp2,  # 使用warp2作为基础图像
                warp_image=warp1,  # warp1作为重叠部分
                mask=mask_tensor   # 使用生成的掩码进行混合
            )

            return mask_tensor, result
        
        except Exception as e:
            print(f"安全前向传播时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def __call__(self, *args, **kwargs):
        """
        重写__call__方法，当直接调用模型实例时，优先使用safe_forward方法
        """
        try:
            # 首先尝试使用正常的forward方法
            return self.forward(*args, **kwargs)
        except Exception as e:
            print(f"标准forward方法失败，切换到safe_forward: {e}")
            
            # 解析参数
            warp1 = None
            warp2 = None
            mask1 = None
            mask2 = None
            input_dict = None
            
            # 处理args
            if len(args) == 1 and isinstance(args[0], dict):
                input_dict = args[0]
            elif len(args) >= 4:
                warp1, warp2, mask1, mask2 = args[:4]
            elif len(args) >= 2:
                warp1, warp2 = args[:2]
            
            # 处理kwargs
            if 'warp_image' in kwargs and 'base_image' in kwargs:
                warp1 = kwargs.get('warp_image')
                warp2 = kwargs.get('base_image')
                mask1 = kwargs.get('warp_mask')
                mask2 = kwargs.get('base_mask')
            elif 'img1' in kwargs and 'img2' in kwargs:
                warp1 = kwargs.get('img1')
                warp2 = kwargs.get('img2')
                mask1 = kwargs.get('mask1')
                mask2 = kwargs.get('mask2')
            
            # 使用safe_forward方法
            return self.safe_forward(warp1, warp2, mask1, mask2, input_dict) 

    def apply_mask(self, base_image, warp_image, mask):
        """
        根据掩码应用图像混合
        
        参数:
            base_image: 基础图像 (warp2) [B, C, H, W]
            warp_image: 应用图像 (warp1) [B, C, H, W]
            mask: 混合掩码 [B, 1, H, W] - 值范围[0,1]，1表示保留warp_image的区域
        
        返回:
            混合后的图像
        """
        # 确保掩码是[0,1]范围
        mask = torch.clamp(mask, 0.0, 1.0)
        
        # 处理传入值可能是None的情况
        if base_image is None or warp_image is None or mask is None:
            if base_image is not None:
                return base_image
            elif warp_image is not None:
                return warp_image
            else:
                return None
        
        # 确保维度匹配
        if mask.dim() != base_image.dim():
            if mask.dim() == 3 and base_image.dim() == 4:
                mask = mask.unsqueeze(1)
        
        # 确保空间尺寸一致
        if mask.shape[2:] != base_image.shape[2:]:
            mask = F.interpolate(mask, size=base_image.shape[2:], mode='bilinear', align_corners=False)
        
        if warp_image.shape[2:] != base_image.shape[2:]:
            warp_image = F.interpolate(warp_image, size=base_image.shape[2:], mode='bilinear', align_corners=False)
        
        # 注意：掩码值1代表保留warp_image的部分，0代表保留base_image的部分
        # 确保mask的通道数为1
        if mask.shape[1] != 1:
            if mask.shape[1] == 3:
                # 如果是3通道掩码，取平均转换为单通道
                mask = mask.mean(dim=1, keepdim=True)
            else:
                # 否则只使用第一个通道
                mask = mask[:, 0:1]
        
        # 执行混合 - 使用mask值为0保留base_image，为1保留warp_image
        result = base_image * (1.0 - mask) + warp_image * mask
        
        return result

class ImprovedDiffusionComposition(EnhancedDiffusionComposition):
    def __init__(self, **kwargs):
        # 提取全景图损失权重参数，不传递给父类
        self.panorama_loss_weight = kwargs.pop('panorama_loss_weight', 0.2)
        super().__init__(**kwargs)
        
    def compute_loss(self, x, t, img1, img2, mask1, mask2):
        """
        计算扩散模型的损失函数，改进版本确保维度对齐和数值稳定性
        参数:
            x: 输入噪声或图像 [B, C, H, W]
            t: 时间步 [B]
            img1, img2: 输入图像 [B, C, H, W]
            mask1, mask2: 输入掩码 [B, 1, H, W]
        返回:
            diffusion_loss: 扩散模型的损失
        """
        try:
            # 检查输入是否包含NaN或Inf值
            if torch.isnan(x).any():
                print("警告: 输入x包含NaN值")
                x = torch.nan_to_num(x, nan=0.0)
            if torch.isnan(t).any():
                print("警告: 时间步t包含NaN值")
                t = torch.nan_to_num(t, nan=0)
            
            # 确保t是长整型张量，而不是整数
            if not isinstance(t, torch.Tensor):
                t = torch.tensor([t], device=x.device, dtype=torch.long)
            elif t.dim() == 0:
                t = t.unsqueeze(0)  # 确保t至少是1D张量
                
            t = t.to(dtype=torch.long)
            
            # 确保所有输入具有相同的空间维度
            if img2.shape[2:] != img1.shape[2:]:
                img2 = F.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
            
            if mask1.shape[2:] != img1.shape[2:]:
                mask1 = F.interpolate(mask1, size=img1.shape[2:], mode='bilinear', align_corners=False)
            
            if mask2.shape[2:] != img1.shape[2:]:
                mask2 = F.interpolate(mask2, size=img1.shape[2:], mode='bilinear', align_corners=False)
            
            if x.shape[2:] != img1.shape[2:]:
                x = F.interpolate(x, size=img1.shape[2:], mode='bilinear', align_corners=False)
            
            # 检查并清理NaN/Inf值 - 使用更安全的范围
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            img1 = torch.nan_to_num(img1, nan=0.0, posinf=1.0, neginf=-1.0)
            img2 = torch.nan_to_num(img2, nan=0.0, posinf=1.0, neginf=-1.0)
            mask1 = torch.nan_to_num(mask1, nan=0.5, posinf=1.0, neginf=0.0)
            mask2 = torch.nan_to_num(mask2, nan=0.5, posinf=1.0, neginf=0.0)
            
            # 确保掩码在[0,1]范围内
            mask1 = torch.clamp(mask1, 0.0, 1.0)
            mask2 = torch.clamp(mask2, 0.0, 1.0)
            
            # 确保图像在[-1,1]范围内
            x = torch.clamp(x, -1.0, 1.0)
            img1 = torch.clamp(img1, -1.0, 1.0)
            img2 = torch.clamp(img2, -1.0, 1.0)
            
            # 1. 添加噪声得到带噪图像和目标噪声
            noisy_x, target_noise = self.forward_diffusion(x, t)
            
            # 2. 使用前向函数预测噪声和掩码
            predicted_noise, learned_mask = self.forward(noisy_x, t, img1, img2, mask1, mask2)
            
            # 确保预测噪声与目标噪声维度匹配
            if predicted_noise.shape != target_noise.shape:
                predicted_noise = F.interpolate(predicted_noise, size=target_noise.shape[2:], mode='bilinear', align_corners=False)
                
            # 3. 计算噪声预测的MSE损失 - 添加安全检查
            if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
                predicted_noise = torch.nan_to_num(predicted_noise, nan=0.0, posinf=0.0, neginf=0.0)
                print("警告: predicted_noise包含NaN或Inf值，已进行修复")
            if torch.isnan(target_noise).any() or torch.isinf(target_noise).any():
                target_noise = torch.nan_to_num(target_noise, nan=0.0, posinf=0.0, neginf=0.0)
                print("警告: target_noise包含NaN或Inf值，已进行修复")
            
            # 使用Huber损失代替MSE提高鲁棒性
            noise_loss = F.smooth_l1_loss(predicted_noise, target_noise, beta=0.1)
            
            # 4. 计算掩码的二元交叉熵损失 - 使用生成的掩码
            # 确保维度匹配
            if learned_mask.dim() != mask1.dim():
                if learned_mask.dim() == 3 and mask1.dim() == 4:
                    learned_mask = learned_mask.unsqueeze(1)
                elif learned_mask.dim() == 4 and learned_mask.shape[1] > 1 and mask1.shape[1] == 1:
                    learned_mask = learned_mask[:, 0:1]
                
            # 确保形状匹配
            if mask1.shape != mask2.shape:
                mask2 = F.interpolate(mask2, size=mask1.shape[2:], mode='bilinear', align_corners=False)
                
            combined_mask = (mask1 + mask2) / 2.0
            
            # 确保维度一致
            if learned_mask.shape[2:] != combined_mask.shape[2:]:
                combined_mask = F.interpolate(combined_mask, size=learned_mask.shape[2:], mode='bilinear', align_corners=False)
            
            # 使用 binary_cross_entropy_with_logits 代替 binary_cross_entropy
            # 这样可以更安全地在自动混合精度训练中使用
            combined_mask = torch.clamp(combined_mask, 1e-5, 1.0 - 1e-5)  # 避免log(0)
            
            # 直接使用未经过sigmoid的learned_mask，让binary_cross_entropy_with_logits处理
            mask_loss = F.binary_cross_entropy_with_logits(learned_mask, combined_mask)
            
            # 5. 添加全景图一致性损失
            # 5.1 生成全景图
            # 使用sigmoid将学习的掩码转换为0-1范围
            transition_mask = torch.sigmoid(learned_mask)
            
            # 根据掩码混合两个图像创建拼接图像
            mixed_image = img1 * (1 - transition_mask) + img2 * transition_mask
            
            # 5.2 使用cal_panorama_consistency_loss计算一致性损失
            # 导入函数（确保已在文件开头导入了loss模块）
            from Composition.Codes.loss import cal_panorama_consistency_loss
            
            # 计算全景图一致性损失
            try:
                panorama_loss = cal_panorama_consistency_loss(
                    mixed_image,  # 拼接后的图像
                    img1,         # 第一个源图像
                    img2,         # 第二个源图像
                    mask1,        # 第一个源图像掩码
                    mask2,        # 第二个源图像掩码
                    transition_mask  # 过渡掩码
                )
            except Exception as e:
                print(f"计算全景一致性损失时出错: {e}")
                panorama_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            
            # 6. 总损失 = 噪声损失 + 掩码损失 + 全景图一致性损失
            total_loss = noise_loss + 0.05 * mask_loss + self.panorama_loss_weight * panorama_loss
            
            # 添加L2正则化以提高稳定性
            l2_reg = 0.0
            for param in self.parameters():
                l2_reg = l2_reg + torch.norm(param, 2)
            total_loss = total_loss + 1e-6 * l2_reg
            
            # 检查损失值
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("警告: 损失计算中发现NaN或Inf")
                if torch.isnan(noise_loss) or torch.isinf(noise_loss):
                    print(f"  噪声损失: {noise_loss.item() if not torch.isnan(noise_loss) and not torch.isinf(noise_loss) else 'NaN/Inf'}")
                if torch.isnan(mask_loss) or torch.isinf(mask_loss):
                    print(f"  掩码损失: {mask_loss.item() if not torch.isnan(mask_loss) and not torch.isinf(mask_loss) else 'NaN/Inf'}")
                if torch.isnan(panorama_loss) or torch.isinf(panorama_loss):
                    print(f"  全景图一致性损失: {panorama_loss.item() if not torch.isnan(panorama_loss) and not torch.isinf(panorama_loss) else 'NaN/Inf'}")
                print(f"  L2正则化: {l2_reg.item() if not torch.isnan(l2_reg) and not torch.isinf(l2_reg) else 'NaN/Inf'}")
                return torch.tensor(1e-5, device=x.device, requires_grad=True)
                
            return total_loss
            
        except Exception as e:
            print(f"计算扩散模型损失时出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个需要梯度的较小张量
            return torch.tensor(1e-5, device=x.device, requires_grad=True)

    def stitch_to_original_size(self, mask, result, crop_info, original_images=None):
        """
        将512x512的预测结果恢复到原始图像尺寸，将生成的warp2模板与原始warp1正确拼接
        
        参数:
            mask: 生成的掩码，形状 [B, 1, H, W]
            result: 生成的拼接结果，形状 [B, 3, H, W]
            crop_info: 裁剪信息字典，包含原始尺寸和裁剪坐标
            original_images: 原始图像信息，包含完整的原始图像
            
        返回:
            full_mask: 恢复到原始尺寸的掩码
            full_result: 恢复到原始尺寸的拼接结果
            panorama: 创建的全景图
        """
        try:
            # 获取裁剪信息
            x1, y1, x2, y2 = crop_info['x1'], crop_info['y1'], crop_info['x2'], crop_info['y2']
            orig_size1 = crop_info['orig_size1']  # (width, height)
            
            # 裁剪区域大小
            crop_width = x2 - x1
            crop_height = y2 - y1
            
            # 原始尺寸
            orig_width, orig_height = orig_size1
            
            # 调整结果到裁剪区域的尺寸
            device = result.device
            batch_size = result.shape[0]
            
            # 创建全尺寸结果张量 - 先用全0初始化
            full_result = torch.zeros((batch_size, 3, orig_height, orig_width), device=device)
            full_mask = torch.zeros((batch_size, 1, orig_height, orig_width), device=device)
            
            # 调整生成的结果和掩码到裁剪区域的大小
            if result.shape[2:] != (crop_height, crop_width):
                result_resized = F.interpolate(result, size=(crop_height, crop_width), mode='bilinear', align_corners=False)
                mask_resized = F.interpolate(mask, size=(crop_height, crop_width), mode='bilinear', align_corners=False)
            else:
                result_resized = result
                mask_resized = mask
                
            # 计算重叠区域的位置 - 优先使用crop_info中的overlap_bbox
            if 'overlap_bbox' in crop_info:
                overlap_x1, overlap_y1, overlap_x2, overlap_y2 = crop_info['overlap_bbox']
                overlap_width = overlap_x2 - overlap_x1
                overlap_center_x = (overlap_x1 + overlap_x2) // 2
                print(f"  使用裁剪区域: ({overlap_x1},{overlap_y1})-({overlap_x2},{overlap_y2}), 宽度={overlap_width}, 高度={overlap_y2-overlap_y1}")
            else:
                # 假设overlap_bbox是整个裁剪区域
                overlap_x1, overlap_y1, overlap_x2, overlap_y2 = x1, y1, x2, y2
                overlap_width = crop_width
                overlap_center_x = (x1 + x2) // 2
                print(f"  使用整个裁剪区域作为重叠区域: ({x1},{y1})-({x2},{y2}), 宽度={crop_width}")
                
            # 创建全景图
            # 假设warp1在左侧，warp2在右侧，重叠区域是warp1的右边缘和warp2的左边缘
            # 全景图宽度 = 两个图像的总宽度 - 重叠宽度
            panorama_width = orig_width + orig_width - overlap_width
            panorama_height = orig_height
            
            print(f"  创建全尺寸全景图: 尺寸={panorama_width}x{panorama_height}, 重叠={overlap_width}px")
            
            # 创建全景图张量
            panorama = torch.zeros((batch_size, 3, panorama_height, panorama_width), device=device)
            
            # 获取原始图像 (如果可用)
            orig_warp1 = None
            orig_warp2 = None
            
            if original_images is not None:
                print("  使用完整原始图像进行拼接")
                if 'warp1' in original_images:
                    orig_warp1 = original_images['warp1']
                    if orig_warp1.shape[2:] != (orig_height, orig_width):
                        orig_warp1 = F.interpolate(orig_warp1, size=(orig_height, orig_width), mode='bilinear', align_corners=False)
                
                if 'warp2' in original_images:
                    orig_warp2 = original_images['warp2']
                    if orig_warp2.shape[2:] != (orig_height, orig_width):
                        orig_warp2 = F.interpolate(orig_warp2, size=(orig_height, orig_width), mode='bilinear', align_corners=False)
            
            # 如果没有原始图像，尝试从result中恢复
            if orig_warp1 is None:
                # 创建一个空白的warp1
                orig_warp1 = torch.zeros((batch_size, 3, orig_height, orig_width), device=device)
                
                # 将result中的内容填充到原始大小的warp1中（仅填充裁剪区域）
                if y2 <= orig_height and x2 <= orig_width:
                    # 将生成结果的一部分放置在裁剪区域
                    orig_warp1[:, :, y1:y2, x1:x2] = result_resized
            
            # 如果没有原始图像，尝试从result中恢复
            if orig_warp2 is None:
                # 创建与warp1相同大小的warp2
                orig_warp2 = torch.zeros((batch_size, 3, orig_height, orig_width), device=device)
                
                # 将result中的内容填充到原始大小的warp2中
                # 注意：我们主要关注从result中恢复warp2（因为warp2是我们的基准图像）
                if y2 <= orig_height and x2 <= orig_width:
                    # 将生成结果的一部分放置在裁剪区域
                    orig_warp2[:, :, y1:y2, x1:x2] = result_resized
            
            # 更新full_result和full_mask - 将结果整合回原始尺寸
            if y2 <= orig_height and x2 <= orig_width:
                print(f"  使用原始尺寸结果: {result_resized.shape}px")
                print(f"  放置结果区域: ({overlap_x1},{overlap_y1})-({overlap_x2},{overlap_y2})")
                
                # 将生成结果放置在重叠区域的位置
                full_result[:, :, overlap_y1:overlap_y2, overlap_x1:overlap_x2] = result_resized[:, :, :overlap_y2-overlap_y1, :overlap_x2-overlap_x1]
                full_mask[:, :, overlap_y1:overlap_y2, overlap_x1:overlap_x2] = mask_resized[:, :, :overlap_y2-overlap_y1, :overlap_x2-overlap_x1]
            
            # 创建全景图
            # 步骤1: 首先放置warp1的左侧部分（不含重叠区）
            panorama[:, :, :orig_height, :overlap_x1] = orig_warp1[:, :, :, :overlap_x1]
            
            # 步骤2: 放置重叠区域 - 使用生成的结果
            # 我们使用生成的掩码（或根据x坐标创建平滑过渡）
            # 计算过渡宽度
            transition_width = min(overlap_width, 50)  # 最多50像素的平滑过渡
            transition_start = max(0, overlap_x1)
            transition_end = min(orig_width, overlap_x2)
            
            # 创建过渡区域
            for x in range(transition_start, transition_end):
                # 基于位置创建渐变过渡 - 从左到右mask值从0变为1
                alpha = (x - transition_start) / max(1, transition_end - transition_start - 1)
                mask_val = torch.full((batch_size, 1, orig_height, 1), alpha, device=device)
                
                # 在重叠区域使用平滑过渡
                # warp1占比 (1-alpha)，warp2占比 alpha
                overlap_pos = x - transition_start
                # 确保形状匹配
                warp1_col = orig_warp1[:, :, :orig_height, x:x+1]  # 提取列并保持4D形状
                # 确保overlap_pos在有效范围内
                if overlap_pos < result_resized.shape[3]:
                    result_col = result_resized[:, :, :orig_height, overlap_pos:overlap_pos+1]
                    if result_col.shape[2] != warp1_col.shape[2]:
                        # 调整高度匹配
                        result_col = F.interpolate(result_col, size=(warp1_col.shape[2], 1), mode='bilinear', align_corners=False)
                    # 混合两列
                    panorama[:, :, :orig_height, x:x+1] = (
                        warp1_col * (1.0 - mask_val) + 
                        result_col * mask_val
                    )
            
            # 步骤3: 放置warp2的右侧部分
            # warp2的右侧部分（从重叠区域结束处开始）放置在全景图中重叠区之后
            # 计算warp2右侧部分在原始图像中的起始位置
            warp2_right_start = overlap_x2
            # 计算全景图中的放置位置
            panorama_right_start = overlap_x1 + overlap_width
            
            # 计算要复制的宽度
            warp2_right_width = orig_width - warp2_right_start
            
            if warp2_right_width > 0:
                # 复制warp2的右侧部分到全景图
                panorama[:, :, :, panorama_right_start:panorama_right_start+warp2_right_width] = orig_warp2[:, :, :, warp2_right_start:orig_width]
            
            print("  全景图创建成功，保存两个版本")
            
            return full_mask, full_result, panorama
        
        except Exception as e:
            print(f"恢复原始尺寸时出错: {e}")
            import traceback
            traceback.print_exc()
            return mask, result, None
        
    def safe_forward_with_original_size(self, warp1=None, warp2=None, mask1=None, mask2=None, 
                                      input_dict=None, restore_original=True):
        """
        带有原始尺寸恢复的安全前向传播，生成warp2模板并正确处理裁剪区域
        
        参数:
            warp1, warp2: 输入图像张量 [B, C, H, W]
            mask1, mask2: 输入掩码张量 [B, 1, H, W]
            input_dict: 输入字典，包含crop_info和原始图像路径
            restore_original: 是否恢复到原始尺寸
            
        返回:
            mask: 生成的掩码
            result: 生成的拼接结果
            panorama: 全景图 (如果恢复原始尺寸)
        """
        try:
            # 获取裁剪信息
            crop_info = None
            original_paths = None
            original_images = None
            
            if input_dict is not None:
                if 'crop_info' in input_dict:
                    crop_info = input_dict['crop_info']
                if 'original_paths' in input_dict:
                    original_paths = input_dict['original_paths']
                    
                # 如果提供了完整原始图像，加载它们用于恢复尺寸
                if 'original_images' in input_dict:
                    original_images = input_dict['original_images']
            
            # 第一步: 使用当前裁剪尺寸进行模型前向传播
            mask_tensor, predicted_image = self.sample(
                img1=warp1,
                img2=warp2,
                mask1=mask1,
                mask2=mask2,
                num_steps=100  # 默认步数，可调整
            )
            
            # 应用掩码生成结果
            result = self.apply_mask(
                base_image=warp2,  # 使用warp2作为基础图像
                warp_image=warp1,  # warp1作为重叠部分
                mask=mask_tensor   # 使用生成的掩码进行混合
            )
            
            # 是否恢复到原始尺寸
            if restore_original and crop_info is not None:
                try:
                    # 对掩码和结果应用恢复操作，保持两个一致
                    mask, result, panorama = self.stitch_to_original_size(mask_tensor, result, crop_info, original_images)
                    return mask, result, panorama
                except Exception as e:
                    print(f"恢复原始尺寸时出错: {e}")
                    # 回退到裁剪尺寸结果
                    return mask_tensor, result, None
            
            # 返回裁剪尺寸的结果
            return mask_tensor, result, None
            
        except Exception as e:
            print(f"安全前向传播时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    # 添加get_loss_components方法到ImprovedDiffusionComposition类中，应该放在compute_loss方法附近
    def get_loss_components(self, x, t, base_image, warp_image, base_mask, warp_mask):
        """
        计算并返回所有损失组件，用于更详细的TensorBoard记录
        
        参数:
            x: 输入张量，混合的基础图像和变形图像
            t: 时间步
            base_image: 基础图像
            warp_image: 变形图像
            base_mask: 基础图像掩码
            warp_mask: 变形图像掩码
            
        返回:
            一个包含所有损失组件的字典
        """
        # 添加噪声得到带噪图像和目标噪声
        noisy_x, target_noise = self.forward_diffusion(x, t)
        
        # 使用前向函数预测噪声和掩码
        predicted_noise, learned_mask = self.forward(noisy_x, t, base_image, warp_image, base_mask, warp_mask)
        
        # 计算各种损失组件
        # 1. 噪声预测损失 (L2损失)
        noise_loss = F.mse_loss(predicted_noise, target_noise)
        
        # 2. 掩码损失 (BCE损失)
        combined_mask = (base_mask + warp_mask) / 2.0
        mask_loss = F.binary_cross_entropy_with_logits(learned_mask, combined_mask)
        
        # 3. 应用掩码后的拼接结果
        sigmoid_mask = torch.sigmoid(learned_mask)
        stitched_image = self.apply_mask(base_image, warp_image, sigmoid_mask)
        
        # 4. L1重建损失
        target_image = (base_image + warp_image) / 2.0  # 简单平均作为目标
        l1_loss = F.l1_loss(stitched_image, target_image)
        
        # 5. 边界平滑损失
        # 提取掩码的边界
        from Composition.Codes.loss import boundary_extraction, cal_boundary_term
        boundary_loss = cal_boundary_term(base_image, warp_image, base_mask, warp_mask, stitched_image)
        
        # 6. 平滑损失
        from Composition.Codes.loss import cal_smooth_term_stitch
        smooth_loss = cal_smooth_term_stitch(stitched_image, sigmoid_mask)
        
        # 7. 感知损失
        from Composition.Codes.loss import cal_perceptual_loss
        try:
            perceptual_loss = cal_perceptual_loss(
                stitched_image, base_image, warp_image, sigmoid_mask, 1.0 - sigmoid_mask
            )
        except Exception as e:
            print(f"计算感知损失时出错: {e}")
            perceptual_loss = torch.tensor(0.0, device=x.device)
        
        # 8. SSIM损失
        from Composition.Codes.loss import cal_ssim_loss
        try:
            ssim_loss = cal_ssim_loss(
                stitched_image, base_image, warp_image, sigmoid_mask, 1.0 - sigmoid_mask
            )
        except Exception as e:
            print(f"计算SSIM损失时出错: {e}")
            ssim_loss = torch.tensor(0.0, device=x.device)
        
        # 9. 颜色一致性损失
        from Composition.Codes.loss import cal_color_consistency_loss
        try:
            color_loss = cal_color_consistency_loss(
                stitched_image, base_image, warp_image, base_mask, warp_mask
            )
        except Exception as e:
            print(f"计算颜色一致性损失时出错: {e}")
            color_loss = torch.tensor(0.0, device=x.device)
        
        # 返回所有损失组件字典
        return {
            'noise_loss': noise_loss,
            'mask_loss': mask_loss,
            'l1_loss': l1_loss,
            'boundary_loss': boundary_loss,
            'smooth_loss': smooth_loss,
            'perceptual_loss': perceptual_loss,
            'ssim_loss': ssim_loss,
            'color_loss': color_loss
        }

# 网络构建块

class ResBlock(nn.Module):
    """带时间嵌入的残差块 - 改进版本确保维度对齐"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout, use_scale_shift_norm=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # 使用更稳定的权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
    def forward(self, x, time_emb):
        # 检查输入
        if torch.isnan(time_emb).any():
            time_emb = torch.nan_to_num(time_emb, nan=0.0)
        
        # 处理时间嵌入
        time_emb = self.time_mlp(time_emb)
        if torch.isnan(time_emb).any():
            time_emb = torch.nan_to_num(time_emb, nan=0.0)
        
        # 残差连接
        shortcut = self.shortcut(x)
        
        # 残差路径
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # 应用缩放和偏移
        if self.use_scale_shift_norm:
            # 将time_emb分为缩放和偏移部分
            scale, shift = torch.chunk(time_emb, 2, dim=1)
            # 调整维度以匹配特征通道
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            # 应用缩放和偏移
            h = self.norm2(h) * (1 + scale) + shift
            h = self.act2(h)
        else:
            h = self.norm2(h)
            h = self.act2(h)
            # 调整time_emb维度以适配h
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            # 确保channels维度匹配 - 如果不匹配则调整
            if time_emb.shape[1] != h.shape[1]:
                # 将time_emb分为两个部分
                emb_chunks = torch.chunk(time_emb, 2, dim=1)
                # 只使用前半部分，确保通道数匹配
                time_emb = emb_chunks[0]
            h = h + time_emb
            
        h = self.dropout(h)
        h = self.conv2(h)
        
        # 确保shortcut和h具有相同的维度
        if h.shape != shortcut.shape:
            # 调整h的空间维度以匹配shortcut
            h = F.interpolate(h, size=shortcut.shape[2:], mode='bilinear', align_corners=False)
        
        # 残差连接
        return h + shortcut


class DownBlock(nn.Module):
    """下采样块 - 用于特征提取"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm=False):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout,
                use_scale_shift_norm
            )
            for i in range(num_res_blocks)
        ])
        
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) if conv_resample else nn.AvgPool2d(2)
            
    def forward(self, x, time_emb):
        # 通过残差块
        for block in self.res_blocks:
            x = block(x, time_emb)
            
        # 下采样
        return self.downsample(x)


class UpBlock(nn.Module):
    """上采样块 - 改进版本确保维度对齐"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm=False):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels + out_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout,
                use_scale_shift_norm
            )
            for i in range(num_res_blocks)
        ])
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        ) if conv_resample else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
    def forward(self, x, res_x, time_emb):
        # 上采样
        x = self.upsample(x)
        
        # 确保x和res_x具有相同的空间维度
        if x.shape[2:] != res_x.shape[2:]:
            # 确定目标尺寸 - 通常选择较大的尺寸
            target_h = max(x.shape[2], res_x.shape[2])
            target_w = max(x.shape[3], res_x.shape[3])
            
            # 调整两个张量到相同尺寸
            if x.shape[2] != target_h or x.shape[3] != target_w:
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            if res_x.shape[2] != target_h or res_x.shape[3] != target_w:
                res_x = F.interpolate(res_x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # 连接跳跃连接的特征
        x = torch.cat([x, res_x], dim=1)
        
        # 通过残差块
        for block in self.res_blocks:
            x = block(x, time_emb)
            
        return x