import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import gcd
import numpy as np
from torchvision import models
import random


def calculate_groups(channels):
    """
    计算GroupNorm的合适组数，确保通道数能被组数整除
    
    Args:
        channels: 通道数
        
    Returns:
        适合的组数
    """
    # 检查常见的组数是否可用
    for groups in [32, 16, 8, 4, 2]:
        if channels % groups == 0:
            return groups
    # 如果都不行，使用1或通道数本身（如果通道数是3，使用3或1）
    return 1 if channels % 3 != 0 else 3


def build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor):
    """
    构建模型输出，支持动态尺寸的图像拼接
    
    Args:
        net: 网络模型
        warp1_tensor: 第一张扭曲图像
        warp2_tensor: 第二张扭曲图像
        mask1_tensor: 第一张图像的掩码
        mask2_tensor: 第二张图像的掩码
        
    Returns:
        包含学习掩码、拼接图像和去噪结果的字典
    """
    # 首先将所有输入调整为固定的512x512大小
    warp1_resized = F.interpolate(warp1_tensor, size=(512, 512), mode='bilinear', align_corners=False)
    mask1_resized = F.interpolate(mask1_tensor, size=(512, 512), mode='bilinear', align_corners=False)
    warp2_resized = F.interpolate(warp2_tensor, size=(512, 512), mode='bilinear', align_corners=False)
    mask2_resized = F.interpolate(mask2_tensor, size=(512, 512), mode='bilinear', align_corners=False)
    
    # 获取原始输出和扩散优化结果
    out, denoised = net(warp1_resized, warp2_resized, mask1_resized, mask2_resized)
    
    # 计算重叠区域
    overlap_region = mask1_tensor * mask2_tensor
    
    # 在重叠区域使用学习的掩码，非重叠区域保持原样
    learned_mask1 = (mask1_tensor - overlap_region) + overlap_region * out
    learned_mask2 = (mask2_tensor - overlap_region) + overlap_region * (1-out)
    
    # 计算最终拼接区域的大小
    # 获取掩码的非零区域
    mask1_nonzero = torch.nonzero(mask1_tensor.squeeze())
    mask2_nonzero = torch.nonzero(mask2_tensor.squeeze())
    
    if len(mask1_nonzero) > 0 and len(mask2_nonzero) > 0:
        # 计算包含所有非零区域的边界框
        min_y = min(mask1_nonzero[:, 0].min(), mask2_nonzero[:, 0].min())
        max_y = max(mask1_nonzero[:, 0].max(), mask2_nonzero[:, 0].max())
        min_x = min(mask1_nonzero[:, 1].min(), mask2_nonzero[:, 1].min())
        max_x = max(mask1_nonzero[:, 1].max(), mask2_nonzero[:, 1].max())
        
        # 创建足够大的输出图像
        output_height = max_y - min_y + 1
        output_width = max_x - min_x + 1
        
        # 调整denoised和掩码到目标尺寸（如果需要）
        if denoised.shape[2:] != (output_height, output_width):
            denoised = F.interpolate(denoised, size=(output_height, output_width), mode='bilinear', align_corners=False)
            learned_mask1 = F.interpolate(learned_mask1, size=(output_height, output_width), mode='bilinear', align_corners=False)
            learned_mask2 = F.interpolate(learned_mask2, size=(output_height, output_width), mode='bilinear', align_corners=False)
    
    # 使用掩码进行图像融合
    # 确保掩码和图像尺寸匹配
    if learned_mask1.shape[1] == 1:
        learned_mask1 = learned_mask1.repeat(1, denoised.shape[1], 1, 1)
    if learned_mask2.shape[1] == 1:
        learned_mask2 = learned_mask2.repeat(1, denoised.shape[1], 1, 1)
    
    # 计算最终拼接图像
    stitched_image = denoised * learned_mask1 + denoised * learned_mask2
    
    # 边界处理：确保掩码总和为1，避免过亮区域
    mask_sum = learned_mask1 + learned_mask2
    mask_sum = torch.clamp(mask_sum, min=1.0)  # 避免除零
    
    # 归一化掩码
    learned_mask1 = learned_mask1 / mask_sum
    learned_mask2 = learned_mask2 / mask_sum
    
    # 重新计算拼接图像
    stitched_image = denoised * learned_mask1 + denoised * learned_mask2
    
    # 返回结果
    out_dict = {
        'learned_mask1': learned_mask1,
        'learned_mask2': learned_mask2,
        'stitched_image': stitched_image,
        'denoised': denoised,
        'output_size': (output_height, output_width) if 'output_height' in locals() else denoised.shape[2:]
    }
    
    return out_dict


class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦位置编码，用于扩散模型的时间步编码
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AdaptiveNorm(nn.Module):
    """
    自适应归一化层，根据时间步调整归一化参数
    """
    def __init__(self, num_features, embedding_dim=128):
        super(AdaptiveNorm, self).__init__()
        groups = calculate_groups(num_features)
        self.norm = nn.GroupNorm(groups, num_features)
        self.ada_scale = nn.Linear(embedding_dim, num_features)
        self.ada_bias = nn.Linear(embedding_dim, num_features)
        
        # 使用更好的初始化
        nn.init.zeros_(self.ada_scale.weight)
        nn.init.zeros_(self.ada_scale.bias)
        nn.init.zeros_(self.ada_bias.weight)
        nn.init.zeros_(self.ada_bias.bias)
        
    def forward(self, x, emb):
        # 应用归一化
        x = self.norm(x)
        
        # 计算自适应比例和偏置
        scale = self.ada_scale(emb).unsqueeze(-1).unsqueeze(-1)
        bias = self.ada_bias(emb).unsqueeze(-1).unsqueeze(-1)
        
        # 应用比例和偏置
        return x * (1 + scale) + bias


class AttentionBlock(nn.Module):
    """局部自注意力块"""
    def __init__(self, channels, heads=4):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.scale = (channels // heads) ** -0.5  # 添加缺少的缩放因子
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.q_proj = nn.Conv2d(channels, channels, 1)  # 添加备用Q投影
        self.k_proj = nn.Conv2d(channels, channels, 1)  # 添加备用K投影
        self.v_proj = nn.Conv2d(channels, channels, 1)  # 添加备用V投影
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        """使用多头自注意力处理特征图"""
        b, c, h, w = x.shape
        
        # 确认输入通道数与期望通道数匹配，否则调整
        if c != self.channels:
            print(f"警告: 输入通道数 {c} 与注意力块预期通道数 {self.channels} 不匹配")
            if c < self.channels:
                # 填充通道
                pad = torch.zeros(b, self.channels - c, h, w, device=x.device)
                x = torch.cat([x, pad], dim=1)
            else:
                # 裁剪通道
                x = x[:, :self.channels, :, :]
            c = self.channels
        
        # 标准化输入
        x_norm = self.norm(x)
        
        # 生成查询、键、值向量
        qkv = self.qkv(x_norm)
        
        # 确保通道数正确分割
        if qkv.size(1) % 3 != 0:
            print(f"警告: QKV通道数 {qkv.size(1)} 不能被3整除，正在调整")
            # 使用备用投影层
            q = self.q_proj(x_norm)
            k = self.k_proj(x_norm)
            v = self.v_proj(x_norm)
            
            # 重塑为多头形式
            q = q.reshape(b, self.heads, c // self.heads, h * w)
            k = k.reshape(b, self.heads, c // self.heads, h * w)
            v = v.reshape(b, self.heads, c // self.heads, h * w)
        else:
            # 标准QKV拆分和重塑
            q, k, v = qkv.reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
            q = q.permute(0, 1, 3, 2)  # [b, heads, h*w, c/heads]
            k = k.permute(0, 1, 2, 3)  # [b, heads, c/heads, h*w]
            v = v.permute(0, 1, 3, 2)  # [b, heads, h*w, c/heads]
        
        # 检查维度并进行必要的转置
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            print(f"错误: 注意力向量维度不正确: q={q.shape}, k={k.shape}, v={v.shape}")
            # 尝试修复维度
            if q.dim() == 3:
                q = q.unsqueeze(1)
            if k.dim() == 3:
                k = k.unsqueeze(1)
            if v.dim() == 3:
                v = v.unsqueeze(1)
        
        # 计算注意力分数前确保维度匹配
        # q应为[b, heads, h*w, c/heads]，k应为[b, heads, c/heads, h*w]
        if k.shape[-2] != q.shape[-1] or k.shape[-1] != q.shape[-2]:
            print(f"维度不匹配: q={q.shape}, k={k.shape}")
            # 尝试转置键以使维度匹配
            k = k.transpose(-1, -2)
            print(f"  -> 转置后: k={k.shape}")
        
        # 使用缩放点积注意力
        attention = torch.matmul(q, k) * self.scale
        
        # 对注意力权重应用softmax
        attention = F.softmax(attention, dim=-1)
        
        # 检查注意力和值的维度匹配
        if attention.shape[-1] != v.shape[-2]:
            print(f"注意力权重与值维度不匹配: attention={attention.shape}, v={v.shape}")
            # 尝试调整值的维度
            v = v.transpose(-1, -2)
            print(f"  -> 转置后: v={v.shape}")
        
        # 应用注意力权重获取加权值
        try:
            out = torch.matmul(attention, v)  # [b, heads, h*w, c/heads]
        except RuntimeError as e:
            print(f"注意力计算错误: {e}")
            print(f"attention: {attention.shape}, v: {v.shape}")
            # 尝试强制重塑为匹配维度
            out = x  # 出错时退回到残差连接
            return self.proj(out)
        
        # 重塑回原始空间维度
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        
        # 应用最终投影
        return self.proj(out)


class GlobalAttention(nn.Module):
    """全局交叉注意力，用于处理整体图像上下文"""
    def __init__(self, channels, heads=8):
        super(GlobalAttention, self).__init__()
        self.channels = channels
        self.num_heads = heads
        head_dim = channels // heads
        
        self.norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        
        self.out_proj = nn.Linear(channels, channels)
        self.scale = head_dim ** -0.5
        
    def forward(self, x):
        # 获取输入维度
        b, c, h, w = x.shape
        
        # 保存原始输入用于残差连接
        residual = x
        
        # 变换形状以进行全局注意力
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [b, h*w, c]
        
        # 应用层归一化
        x = self.norm(x)
        
        # 获取查询、键和值
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # 重塑为多头形式
        q = q.reshape(b, h * w, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)  # [b, num_heads, h*w, c//num_heads]
        k = k.reshape(b, h * w, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)  # [b, num_heads, h*w, c//num_heads]
        v = v.reshape(b, h * w, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)  # [b, num_heads, h*w, c//num_heads]
        
        try:
            # 转置 k 以准备矩阵乘法
            k = k.permute(0, 1, 3, 2)  # [b, num_heads, c//num_heads, h*w]
            
            # 计算注意力分数
            attn = torch.matmul(q, k) * self.scale  # [b, num_heads, h*w, h*w]
            
            # 应用softmax以获得注意力权重
        attn = F.softmax(attn, dim=-1)
        
            # 应用注意力权重于值向量
            out = torch.matmul(attn, v)  # [b, num_heads, h*w, c//num_heads]
            
        except RuntimeError as e:
            print(f"全局注意力计算错误: {e}")
            print(f"q形状: {q.shape}, k形状: {k.shape}, v形状: {v.shape}")
            
            # 尝试修复维度不匹配
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                # 打印更详细的调试信息
                print(f"处理维度不匹配: {str(e)}")
                print(f"q形状: {q.shape}, k转置前形状: {k.shape}")
                
                # 调整 k 的维度
                if k.shape[-1] != q.shape[-1]:
                    k = k.transpose(-1, -2)
                    print(f"k转置后形状: {k.shape}")
                
                # 重试注意力计算
                attn = torch.matmul(q, k) * self.scale
                attn = F.softmax(attn, dim=-1)
                
                # 确保 v 的维度与 attn 匹配
                if v.shape[-2] != attn.shape[-1]:
                    v = v.transpose(-1, -2)
                    print(f"v调整后形状: {v.shape}")
                
                out = torch.matmul(attn, v)
            else:
                # 其他错误，返回残差连接
                print(f"遇到无法修复的错误: {e}")
                return residual
        
        # 重塑回原始形式
        out = out.permute(0, 2, 1, 3).reshape(b, h * w, c)
        
        # 应用输出投影
        out = self.out_proj(out)
        
        # 重塑回通道优先形式
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        
        # 添加残差连接
        return out + residual


class ImprovedDownBlock(nn.Module):
    """
    改进的下采样块，包含残差连接和时间步条件
    """
    def __init__(self, inchannels, outchannels, dilation, embedding_dim=128, pool=True):
        super(ImprovedDownBlock, self).__init__()
        
        # 计算合适的组数 (确保outchannels能被groups整除)
        groups = calculate_groups(outchannels)
        
        # 定义池化层（如果需要）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(groups, outchannels)
        self.act1 = nn.SiLU()
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, outchannels)
        self.act2 = nn.SiLU()
        
        # 时间步条件
        self.time_emb = nn.Sequential(
            nn.Linear(embedding_dim, outchannels),
            nn.SiLU()
        )
        
        # 残差连接
        if inchannels != outchannels or pool:
            self.residual = nn.Sequential(
                nn.Conv2d(inchannels, outchannels, 1, stride=1 if not pool else 2),
                nn.GroupNorm(groups, outchannels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x, emb=None):
        # 残差连接
        residual = self.residual(x)
        
        # 前向传播
        x = self.pool(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        # 添加时间嵌入（如果提供）
        if emb is not None:
            time_emb = self.time_emb(emb).unsqueeze(-1).unsqueeze(-1)
            x = x + time_emb
            
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        # 返回结果加上残差连接
        return x + residual


class ImprovedUpBlock(nn.Module):
    """改进的上采样块，包含更多的attention和条件机制"""
    
    def __init__(self, inchannels, outchannels, dilation, embedding_dim=128, use_attention=False):
        super(ImprovedUpBlock, self).__init__()
        
        # 计算合适的组数 (确保通道数能被组数整除)
        in_half_channels = inchannels // 2
        groups_in_half = calculate_groups(in_half_channels)
        groups_out = calculate_groups(outchannels)
        
        # 上采样卷积层 - 在上采样后处理特征
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_half_channels, in_half_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups_in_half, in_half_channels),
            nn.SiLU()
        )

        # 主要卷积块
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(groups_out, outchannels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups_out, outchannels)
        self.act2 = nn.SiLU()
        
        # 时间步条件
        self.time_emb = nn.Sequential(
            nn.Linear(embedding_dim, outchannels),
            nn.SiLU()
        )
        
        # 注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(outchannels)
        
        # 残差连接
        self.residual = nn.Conv2d(inchannels, outchannels, 1)

        # 为通道适配器创建字典缓存
        self.channel_adapters = {}

    def forward(self, x1, x2, emb=None):
        # 首先检查尺寸是否需要调整
        if x1.shape[2:] != x2.shape[2:]:
            # 确保目标尺寸是偶数，避免插值舍入问题
            target_h = x2.shape[2]
            target_w = x2.shape[3]
            # 确保尺寸是偶数
            target_h = ((target_h + 1) // 2) * 2
            target_w = ((target_w + 1) // 2) * 2
            
            # 将x1上采样到与x2匹配的尺寸
            x1 = F.interpolate(x1, size=(target_h, target_w), 
                              mode='bilinear', align_corners=True)
            
            # 确保x2也是目标尺寸
            if x2.shape[2:] != (target_h, target_w):
                x2 = F.interpolate(x2, size=(target_h, target_w), 
                                  mode='bilinear', align_corners=True)
            
            # print(f"空间尺寸调整: x1={x1.shape}, x2={x2.shape}")
        
        # 确保通道数一致性，如果存在维度不匹配，可以调整
        if x1.shape[1] != x2.shape[1]:
            # print(f"通道数不匹配: x1: {x1.shape[1]}, x2: {x2.shape[1]}")
            # 创建唯一键来标识这种通道适配情况
            adapter_key = f"{x1.shape[1]}_{x2.shape[1]}"
            
            # 检查是否已经有缓存的适配器
            if adapter_key not in self.channel_adapters:
                # print(f"创建新的通道适配器: {adapter_key}")
                # 创建新的适配器并缓存
                self.channel_adapters[adapter_key] = nn.Conv2d(
                    x1.shape[1], x2.shape[1], kernel_size=1
                ).to(x1.device)
            
            # 使用缓存的适配器
            channel_adapter = self.channel_adapters[adapter_key]
            x1 = channel_adapter(x1)
            # print(f"调整后的通道数: x1: {x1.shape[1]}, x2: {x2.shape[1]}")
        
        # 应用上采样
        x1_half = x1
        x1_up = self.upsample(x1_half)
        
        # 检查上采样后的尺寸
        if x1_up.shape[2:] != x2.shape[2:]:
            # 如果上采样后尺寸不匹配，重新调整
            x1_up = F.interpolate(x1_up, size=x2.shape[2:], 
                                 mode='bilinear', align_corners=True)
        
        # 然后应用卷积
        x1_up = self.up_conv(x1_up)
        
        # 最终尺寸检查，确保特征连接前尺寸一致
        if x1_up.shape[2:] != x2.shape[2:]:
            # 使用x2的尺寸作为标准
            x1_up = F.interpolate(x1_up, size=x2.shape[2:], 
                                 mode='bilinear', align_corners=True)
        
        # 标准化通道数，确保连接前的通道数一致性
        if hasattr(self, 'channel_normalization'):
            x1_up = self.channel_normalization(x1_up)
            x2 = self.channel_normalization(x2)
            
        # 连接特征
        x = torch.cat([x1_up, x2], dim=1)
        
        # 记录连接后的维度
        # print(f"UpBlock连接维度: x={x1_up.shape}, res_x={x2.shape} -> 连接后: {x.shape}")
        
        # 应用主卷积块
        out = self.conv1(x)
        out = self.norm1(out)
        
        # 应用时间步条件，如果提供
        if emb is not None:
            # 确保时间嵌入维度正确
            time_emb = self.time_emb(emb)
            
            # 检查并调整时间嵌入的通道数
            if time_emb.shape[1] != out.shape[1]:
                # 调整时间嵌入通道数
                if time_emb.shape[1] > out.shape[1]:
                    time_emb = time_emb[:, :out.shape[1]]
            else:
                    padding = torch.zeros(time_emb.shape[0], out.shape[1] - time_emb.shape[1],
                                        device=time_emb.device)
                    time_emb = torch.cat([time_emb, padding], dim=1)
            
            # 增加空间维度并应用
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            out = out + time_emb
            
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        # 残差连接 - 需要调整输入通道维度以匹配输出
        residual = self.residual(x)
        
        # 确保维度一致
        if residual.shape != out.shape:
            # 如果残差尺寸不一致，调整尺寸
            if residual.shape[2:] != out.shape[2:]:
                residual = F.interpolate(residual, size=out.shape[2:], 
                                       mode='bilinear', align_corners=True)
            
            # 如果通道数不一致，调整通道
            if residual.shape[1] != out.shape[1]:
                if not hasattr(self, 'channel_adapter_residual') or \
                   self.channel_adapter_residual.in_channels != residual.shape[1] or \
                   self.channel_adapter_residual.out_channels != out.shape[1]:
                    self.channel_adapter_residual = nn.Conv2d(
                        residual.shape[1], out.shape[1], kernel_size=1
                    ).to(residual.device)
                
                residual = self.channel_adapter_residual(residual)
        
        # 残差连接
        out = out + residual
        out = self.act2(out)
        
        # 如果使用注意力
        if self.use_attention:
            out = self.attention(out)
        
        return out


class SelfAttention(nn.Module):
    """
    自注意力模块，用于增强特征提取能力
    """
    def __init__(self, channels, heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = heads
        
        # 确保channels能被heads整除，避免后续维度问题
        if channels % heads != 0:
            old_heads = heads
            heads = math.gcd(channels, heads)  # 使用最大公约数
            print(f"警告: 通道数 {channels} 不能被 {old_heads} 整除，调整头数为 {heads}")
        
        self.head_dim = channels // heads
        self.num_heads = heads
        
        # 计算合适的组数 (确保channels能被groups整除)
        groups = calculate_groups(channels)
        
        self.norm = nn.GroupNorm(groups, channels)
        
        # 在qkv投影中添加检查，确保通道数是3的倍数
        qkv_channels = channels * 3
        self.qkv = nn.Conv2d(channels, qkv_channels, 1)
        
        # 确保scale值不为0，避免数值问题
        self.scale = 1.0 / math.sqrt(max(self.head_dim, 1))
        
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # 初始化为接近恒等映射
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        
        # 应用层归一化
        x = self.norm(x)
        
        # 获取查询、键和值
        try:
        qkv = self.qkv(x)
            
            # 确保可以被3整除
            if qkv.shape[1] % 3 != 0:
                # 如果通道数不能被3整除，手动调整
                pad_channels = 3 - (qkv.shape[1] % 3)
                if pad_channels > 0:
                    padding = torch.zeros(b, pad_channels, h, w, device=qkv.device)
                    qkv = torch.cat([qkv, padding], dim=1)
            
            # 现在安全地分割
            chunks = qkv.chunk(3, dim=1)
            q, k, v = chunks
            
            # 确保通道数一致性
            min_channels = min(q.shape[1], k.shape[1], v.shape[1])
            if q.shape[1] != min_channels:
                q = q[:, :min_channels]
            if k.shape[1] != min_channels:
                k = k[:, :min_channels]
            if v.shape[1] != min_channels:
                v = v[:, :min_channels]
                
            # 调整 num_heads 确保可以被 min_channels 整除
            num_heads = self.num_heads
            if min_channels % num_heads != 0:
                # 找到能被 min_channels 整除的最大因子
                old_heads = num_heads
                while min_channels % num_heads != 0 and num_heads > 1:
                    num_heads -= 1
                # print(f"调整注意力头数: {old_heads} -> {num_heads}，以匹配通道数 {min_channels}")
            
            head_dim = min_channels // num_heads
            
        except Exception as e:
            print(f"QKV分割错误: {e}")
            return residual
            
        try:
            # 重塑张量用于多头注意力计算
            q = q.reshape(b, num_heads, head_dim, h * w).permute(0, 1, 3, 2)  # b, nh, hw, hd
            k = k.reshape(b, num_heads, head_dim, h * w)  # b, nh, hd, hw
            v = v.reshape(b, num_heads, head_dim, h * w).permute(0, 1, 3, 2)  # b, nh, hw, hd
            
            # 计算注意力分数
            attention = torch.matmul(q, k) * self.scale  # b, nh, hw, hw
            attention = F.softmax(attention, dim=-1)
            
            # 应用注意力权重
            out = torch.matmul(attention, v)  # b, nh, hw, hd
            
            # 重新整形
            out = out.permute(0, 1, 3, 2).reshape(b, min_channels, h, w)
            
            # 如果通道数与原始通道数不匹配，进行调整
            if out.shape[1] != c:
                if out.shape[1] < c:
                    # 填充缺少的通道
                    pad = torch.zeros(b, c - out.shape[1], h, w, device=out.device)
                    out = torch.cat([out, pad], dim=1)
                else:
                    # 裁剪多余的通道
                    out = out[:, :c]
        
        # 输出投影
        out = self.proj(out)
        
            # 添加残差连接并返回
        return out + residual
            
        except RuntimeError as e:
            print(f"注意力计算错误: {e}")
            print(f"q形状: {q.shape}, k形状: {k.shape}, v形状: {v.shape}")
            print(f"num_heads: {num_heads}, head_dim: {head_dim}")
            # 出错时返回输入
            return residual


class ResBlock(nn.Module):
    """
    改进版残差块，支持时间条件嵌入
    """

    def __init__(self, in_channels, out_channels, time_dim=None, dropout=0.0, use_scale_shift=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_scale_shift = use_scale_shift

        # 第一个卷积块
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # 第二个卷积块
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        
        # 时间条件嵌入层
        if time_dim is not None:
            if use_scale_shift:
            self.time_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_dim, out_channels * 2)
                )
            else:
                self.time_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_dim, out_channels)
                )
        else:
            self.time_mlp = None
            
        # 残差连接层 - 处理通道数不匹配
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t=None):
        # 打印输入信息 (可选)
        # print(f"ResBlock输入: {x.shape}, 权重形状: {self.norm1.weight.shape}")
        
        # 保存原始输入用于残差连接
        original_x = x
        
        # 修复通道数不匹配 - 确保第一个归一化层的输入通道匹配
        if x.size(1) != self.in_channels:
            # print(f"调整输入通道: {x.size(1)} -> {self.in_channels}")
            if x.size(1) < self.in_channels:
                # 填充通道
                pad = torch.zeros(x.size(0), self.in_channels - x.size(1), *x.size()[2:], device=x.device)
                x = torch.cat([x, pad], dim=1)
            else:
                # 裁剪通道
                x = x[:, :self.in_channels, :, :]
        
        # 第一个卷积块
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 添加时间嵌入（如果提供）
        if self.time_mlp is not None and t is not None:
            time_emb = self.time_mlp(t)
            if self.use_scale_shift:
                time_emb = time_emb.view(time_emb.shape[0], -1, 1, 1)
                
                # 确保维度匹配 - 确保时间嵌入通道数匹配h的通道数
                if time_emb.shape[1] != h.shape[1] * 2:
                    required_channels = h.shape[1] * 2
                    # print(f"调整时间嵌入: {time_emb.shape[1]} -> {required_channels}")
                    
                    if time_emb.shape[1] > required_channels:
                        # 裁剪多余的通道
                        time_emb = time_emb[:, :required_channels]
                    else:
                        # 填充缺少的通道
                        padding = torch.zeros(time_emb.shape[0], required_channels - time_emb.shape[1], 
                                            1, 1, device=time_emb.device)
                        time_emb = torch.cat([time_emb, padding], dim=1)
                
                # 分割缩放和偏移参数
                scale, shift = torch.chunk(time_emb, 2, dim=1)
                
                # 应用缩放和偏移
                h = h * (1 + scale) + shift
            else:
                # 确保时间嵌入维度匹配
                if time_emb.shape[1] != h.shape[1]:
                    required_channels = h.shape[1]
                    # print(f"调整时间嵌入: {time_emb.shape[1]} -> {required_channels}")
                    
                    if time_emb.shape[1] > required_channels:
                        # 裁剪多余的通道
                        time_emb = time_emb[:, :required_channels]
                    else:
                        # 填充缺少的通道
                        padding = torch.zeros(time_emb.shape[0], required_channels - time_emb.shape[1], 
                                            device=time_emb.device)
                        time_emb = torch.cat([time_emb, padding], dim=1)
                
                # 应用广播加法
                time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            h = h + time_emb
            
        # 第二个卷积块 - 确保h通道数与norm2权重匹配
        if h.shape[1] != self.out_channels:
            # print(f"调整中间特征通道: {h.shape[1]} -> {self.out_channels}")
            if h.shape[1] < self.out_channels:
                # 填充通道
                pad = torch.zeros(h.size(0), self.out_channels - h.size(1), *h.size()[2:], device=h.device)
                h = torch.cat([h, pad], dim=1)
            else:
                # 裁剪通道
                h = h[:, :self.out_channels, :, :]
        
        # 继续第二个卷积块处理
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # 处理残差连接 - 确保尺寸一致
        # 首先确保通道维度匹配
        if original_x.shape[1] != h.shape[1]:
            # 使用1x1卷积调整通道数
            original_x = self.shortcut(original_x)
        
        # 确保空间尺寸匹配 (宽高)
        if original_x.shape[2:] != h.shape[2:]:
            # 为确保整数尺寸，先取整数值
            target_h = h.shape[2]
            target_w = h.shape[3]
            # 确保尺寸是偶数，有助于避免舍入错误
            target_h = (target_h // 2) * 2
            target_w = (target_w // 2) * 2
            
            # 使用对齐的双线性插值
            original_x = F.interpolate(
                original_x, 
                size=(target_h, target_w),
                mode='bilinear', 
                align_corners=True
            )
            
            # 如果h的尺寸已经变了，也调整h
            if h.shape[2:] != (target_h, target_w):
                h = F.interpolate(
                    h, 
                    size=(target_h, target_w),
                    mode='bilinear', 
                    align_corners=True
                )
        
        # 最终检查确保两个张量形状完全一致
        assert original_x.shape == h.shape, f"残差连接形状不匹配: {original_x.shape} vs {h.shape}"
        
        # 将修复的残差添加到输出
        return original_x + h


class ImprovedDiffusionModel(nn.Module):
    """
    改进的扩散模型，支持多步采样和完整的扩散逆过程
    """
    def __init__(self, image_size, in_channels, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.in_channels = in_channels
        self.time_dim = time_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim*2),
            nn.GELU(),
            nn.Linear(time_dim*2, time_dim),
        )
        
        # 通道数配置 - 修改通道数以保持一致性
        ch1 = 64
        ch2 = 128
        ch3 = 256
        ch4 = 256
        ch_mid = 512
        
        # 下采样模块
        self.downs = nn.ModuleList([
            ImprovedDownBlock(in_channels, ch1, dilation=1),  # in -> 64
            ImprovedDownBlock(ch1, ch2, dilation=1),          # 64 -> 128
            ImprovedDownBlock(ch2, ch3, dilation=1),          # 128 -> 256
            ImprovedDownBlock(ch3, ch4, dilation=1),          # 256 -> 256
        ])
        
        # 中间模块包含自注意力
        self.mid = nn.ModuleList([
            ResBlock(ch4, ch_mid, time_dim),                  # 256 -> 512
            SelfAttention(ch_mid),                            # 512 attention
            ResBlock(ch_mid, ch4, time_dim),                  # 512 -> 256
        ])
        
        # 上采样模块 - 确保通道数正确匹配
        self.ups = nn.ModuleList([
            ImprovedUpBlock(ch4 + ch4, ch3, dilation=1),      # 256+256=512 -> 256
            ImprovedUpBlock(ch3 + ch3, ch2, dilation=1),      # 256+256=512 -> 128
            ImprovedUpBlock(ch2 + ch2, ch1, dilation=1),      # 128+128=256 -> 64
            ImprovedUpBlock(ch1 + ch1, ch1, dilation=1),      # 64+64=128 -> 64
        ])
        
        # 预测噪声的最终卷积层 - 确保输入通道数匹配
        self.final_conv = nn.Sequential(
            nn.Conv2d(ch1 + in_channels, in_channels * 4, kernel_size=3, padding=1),
            # 对于in_channels=3，使用num_groups=3而不是8，确保可以整除
            nn.GroupNorm(min(3, in_channels), in_channels * 4),
            nn.SiLU(),
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1)
        )
        
        # 初始化 beta 和 alpha 参数
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.num_timesteps = 1000
        
        # 预计算 diffusion 的 beta 和 alpha 参数
        self.register_buffer('betas', self._linear_beta_schedule())
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
    
    def _linear_beta_schedule(self):
        """线性 beta 调度"""
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=self.device)
    
    def _extract(self, a, t, x_shape):
        """从 alpha 或其他缓冲区中提取特定时间步的值"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def forward_diffusion(self, x, t):
        """
        对输入图像执行前向扩散过程
        参数:
            x: 输入图像
            t: 时间步
        返回:
            带噪图像和噪声目标
        """
        # 生成随机噪声，保持数据类型与输入一致
        noise = torch.randn_like(x)
        
        # 检查t是否有效索引
        t = torch.clamp(t, 0, self.num_timesteps - 1)
        
        # 获取参数并进行形状调整
        alpha_cumprod = self.sqrt_alphas_cumprod.gather(-1, t)
        alpha_cumprod = alpha_cumprod.reshape(-1, 1, 1, 1)
        
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.gather(-1, t)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.reshape(-1, 1, 1, 1)
        
        # 在混合精度训练中提高数值稳定性
        # 检测并处理可能的NaN或Inf值
        if torch.isnan(alpha_cumprod).any() or torch.isinf(alpha_cumprod).any():
            # 使用安全的替代值
            alpha_cumprod = torch.where(
                torch.isnan(alpha_cumprod) | torch.isinf(alpha_cumprod),
                torch.ones_like(alpha_cumprod) * 0.99,
                alpha_cumprod
            )
        
        if torch.isnan(sqrt_one_minus_alpha).any() or torch.isinf(sqrt_one_minus_alpha).any():
            sqrt_one_minus_alpha = torch.where(
                torch.isnan(sqrt_one_minus_alpha) | torch.isinf(sqrt_one_minus_alpha),
                torch.ones_like(sqrt_one_minus_alpha) * 0.1,
                sqrt_one_minus_alpha
            )
        
        # 在FP16中，数值范围更小，需要防止溢出
        if x.dtype == torch.float16:
            # 转换为FP32进行计算
            x_fp32 = x.to(torch.float32)
            alpha_cumprod_fp32 = alpha_cumprod.to(torch.float32)
            sqrt_one_minus_alpha_fp32 = sqrt_one_minus_alpha.to(torch.float32)
            noise_fp32 = noise.to(torch.float32)
            
            # 使用FP32精度计算
            noisy_x = (alpha_cumprod_fp32 * x_fp32 + sqrt_one_minus_alpha_fp32 * noise_fp32)
            
            # 转回原始精度
            noisy_x = noisy_x.to(x.dtype)
        else:
            # 标准计算
            noisy_x = alpha_cumprod * x + sqrt_one_minus_alpha * noise
        
        # 确保结果不包含NaN或Inf
        if torch.isnan(noisy_x).any() or torch.isinf(noisy_x).any():
            # 替换无效值
            noisy_x = torch.where(
                torch.isnan(noisy_x) | torch.isinf(noisy_x),
                x,  # 使用原始图像作为备选
                noisy_x
            )
        
        # 在返回前剪裁到有效范围
        noisy_x = torch.clamp(noisy_x, -10.0, 10.0)
        
        return noisy_x, noise
    
    def forward(self, x, timesteps):
        """预测噪声"""
        # 时间嵌入
        t = self.time_mlp(timesteps)
        
        # 保存原始输入用于最终的skip连接
        x_orig = x
        
        # 初始特征
        x_skip = []
        h = x
        
        # 下采样并存储跳跃连接
        for down in self.downs:
            h = down(h, t)  # 使用时间嵌入
            x_skip.append(h)  # 存储每层的输出用于跳跃连接
        
        # 中间处理
        for layer in self.mid:
            if isinstance(layer, SelfAttention):
                h = layer(h)
            else:
                h = layer(h, t)
        
        # 上采样并应用跳跃连接
        x_skip = list(reversed(x_skip))
        for idx, up in enumerate(self.ups):
            # 使用skip连接的特征和上一层的特征连接
            h = up(h, x_skip[idx], t)
        
        # 确保原始输入的尺寸与当前特征图匹配
        if x_orig.shape[2:] != h.shape[2:]:
            x_orig = F.interpolate(x_orig, size=h.shape[2:], mode='bilinear', align_corners=True)
        
        # 最终输出层
        result = torch.cat([h, x_orig], dim=1)  # 添加与原始输入的跳跃连接
        return self.final_conv(result)
    
    def sample(self, x, num_steps=100):
        """从噪声采样生成图像"""
        b, *_ = x.shape
        device = x.device
        
        # 创建降噪步骤表
        if num_steps < self.num_timesteps:
            timesteps = torch.linspace(0, self.num_timesteps-1, num_steps, dtype=torch.long, device=device)
        else:
            timesteps = torch.arange(0, self.num_timesteps, dtype=torch.long, device=device)
        
        # 逆向扩散过程
        for i in reversed(range(0, len(timesteps))):
            t = torch.full((b,), timesteps[i], device=device, dtype=torch.long)
            
            # 获取当前时间步对应的系数
            alpha_t = self._extract(self.alphas, t, x.shape)
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
            
            # 如果不是最后一步
            if i > 0:
                beta_t = self._extract(self.betas, t, x.shape)
                # 获取前一个时间步的累积alpha
                alpha_cumprod_prev_t = self._extract(
                    torch.cat([self.alphas_cumprod[0:1], self.alphas_cumprod[:-1]]), 
                    t, x.shape
                )
                
                # 计算扩散系数
                variance = beta_t * (1. - alpha_cumprod_prev_t) / (1. - alpha_cumprod_t)
                noise = torch.randn_like(x) * torch.sqrt(variance)
            else:
                noise = 0.
            
            # 预测噪声
            predicted_noise = self.forward(x, t)
            
            # 应用去噪步骤
            x = 1 / torch.sqrt(alpha_t) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            ) + noise
        
        return x


class FeatureFusion(nn.Module):
    """
    特征融合模块，用于融合不同特征层次
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU()
        )
        
    def forward(self, x1, x2):
        # 确保特征图尺寸一致
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        
        # 特征融合
        fusion = torch.cat([x1, x2], dim=1)
        return self.conv(fusion)


class DownBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout, num_res_blocks=1, 
                 conv_resample=True, use_scale_shift_norm=False):
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
        
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1) if conv_resample else nn.AvgPool2d(2)
    
    def forward(self, x, time_emb):
        """前向传播"""
        # 通过残差块
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        
        # 下采样
        return self.downsample(x)


class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout, num_res_blocks=1, 
                 conv_resample=True, use_scale_shift_norm=False):
        super().__init__()
        
        # 调整通道数的卷积层 - 处理skip连接和不匹配的通道数
        self.channel_mapper = nn.Conv2d(in_channels*2, in_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # 添加通道适配层，确保输入输出安全匹配
        self.input_adapter = nn.Sequential(
            nn.GroupNorm(calculate_groups(in_channels), in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        
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
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        ) if conv_resample else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x, res_x, time_emb):
        """前向传播，带跳跃连接"""
        # 首先通过输入适配器确保通道维度安全
        x = self.input_adapter(x)
        
        # 处理跳跃连接
        if res_x is not None:
            # 打印尺寸，便于调试
            print(f"UpBlock连接维度: x={x.shape}, res_x={res_x.shape}")
            
            # 确保特征图尺寸匹配
            if x.shape[2:] != res_x.shape[2:]:
                x = F.interpolate(x, size=res_x.shape[2:], mode='bilinear', align_corners=False)
                print(f"插值后: x={x.shape}, res_x={res_x.shape}")
            
            # 检查通道数是否匹配
            if x.shape[1] == res_x.shape[1]:
                # 通道数相同，直接cat连接
                x = torch.cat([x, res_x], dim=1)
            else:
                print(f"通道不匹配，需要调整: x={x.shape[1]}, res_x={res_x.shape[1]}")
                
                # 创建一个动态通道适配器
                res_x_adapted = None
                
                if hasattr(self, 'channel_adapter_res') and self.channel_adapter_res.weight.shape[1] == res_x.shape[1]:
                    # 使用已有的适配器
                    res_x_adapted = self.channel_adapter_res(res_x)
                else:
                    # 创建新的适配器
                    self.channel_adapter_res = nn.Conv2d(res_x.shape[1], x.shape[1], kernel_size=1).to(x.device)
                    res_x_adapted = self.channel_adapter_res(res_x)
                
                # 现在可以连接
                x = torch.cat([x, res_x_adapted], dim=1)
                print(f"连接后形状: {x.shape}")
        
        # 应用通道映射器，将concat后的特征转换为所需的通道数
        x = self.channel_mapper(x)
        
        # 确保通道数匹配ResBlock的预期输入
        first_resblock = self.res_blocks[0]
        if x.shape[1] != first_resblock.in_channels:
            print(f"调整通道以匹配ResBlock: {x.shape[1]} -> {first_resblock.in_channels}")
            if x.shape[1] < first_resblock.in_channels:
                # 填充通道
                pad = torch.zeros(x.size(0), first_resblock.in_channels - x.size(1), *x.size()[2:], device=x.device)
                x = torch.cat([x, pad], dim=1)
            else:
                # 裁剪通道
                x = x[:, :first_resblock.in_channels]
        
        # 通过残差块
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        
        # 上采样
        return self.upsample(x)


class EnhancedDiffusionComposition(nn.Module):
    """
    增强的扩散模型用于图像拼接，以warp2为固定基准图像，生成最终形变蒙版
    实现分层特征提取和两阶段（局部/全局）处理
    """
    def __init__(self, num_timesteps=1000, beta_schedule='linear', 
                 image_size=256, base_channels=64, attention_resolutions=[16, 8],
                 dropout=0.0, channel_mult=(1, 2, 4, 8), conv_resample=True,
                 num_res_blocks=2, heads=4, use_scale_shift_norm=True):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.image_size = image_size
        self.heads = heads
        
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
        self.global_attn = GlobalAttention(base_channels*8, heads=heads)
        
        # 上采样路径 - 带有跳跃连接
        self.up1 = UpBlock(base_channels*8, base_channels*4, time_embed_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm)
        self.up2 = UpBlock(base_channels*4, base_channels*2, time_embed_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm)
        self.up3 = UpBlock(base_channels*2, base_channels, time_embed_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm)
        self.up4 = UpBlock(base_channels, base_channels, time_embed_dim, dropout, num_res_blocks, conv_resample, use_scale_shift_norm)
        
        # 添加最终处理层
        self.final_norm = nn.GroupNorm(calculate_groups(base_channels), base_channels)
        self.final_act = nn.SiLU()
        
        # 输出层 - 生成噪声预测
        self.final_conv = nn.Sequential(
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
        half_dim = self.time_embed[0].in_features // 2
        # 将整数转换为张量
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
        # 生成随机噪声，保持数据类型与输入一致
        noise = torch.randn_like(x)
        
        # 检查t是否有效索引
        t = torch.clamp(t, 0, self.num_timesteps - 1)
        
        # 获取参数并进行形状调整
        alpha_cumprod = self.sqrt_alphas_cumprod.gather(-1, t)
        alpha_cumprod = alpha_cumprod.reshape(-1, 1, 1, 1)
        
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.gather(-1, t)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.reshape(-1, 1, 1, 1)
        
        # 在混合精度训练中提高数值稳定性
        # 检测并处理可能的NaN或Inf值
        if torch.isnan(alpha_cumprod).any() or torch.isinf(alpha_cumprod).any():
            # 使用安全的替代值
            alpha_cumprod = torch.where(
                torch.isnan(alpha_cumprod) | torch.isinf(alpha_cumprod),
                torch.ones_like(alpha_cumprod) * 0.99,
                alpha_cumprod
            )
        
        if torch.isnan(sqrt_one_minus_alpha).any() or torch.isinf(sqrt_one_minus_alpha).any():
            sqrt_one_minus_alpha = torch.where(
                torch.isnan(sqrt_one_minus_alpha) | torch.isinf(sqrt_one_minus_alpha),
                torch.ones_like(sqrt_one_minus_alpha) * 0.1,
                sqrt_one_minus_alpha
            )
        
        # 在FP16中，数值范围更小，需要防止溢出
        if x.dtype == torch.float16:
            # 转换为FP32进行计算
            x_fp32 = x.to(torch.float32)
            alpha_cumprod_fp32 = alpha_cumprod.to(torch.float32)
            sqrt_one_minus_alpha_fp32 = sqrt_one_minus_alpha.to(torch.float32)
            noise_fp32 = noise.to(torch.float32)
            
            # 使用FP32精度计算
            noisy_x = (alpha_cumprod_fp32 * x_fp32 + sqrt_one_minus_alpha_fp32 * noise_fp32)
            
            # 转回原始精度
            noisy_x = noisy_x.to(x.dtype)
        else:
            # 标准计算
            noisy_x = alpha_cumprod * x + sqrt_one_minus_alpha * noise
        
        # 确保结果不包含NaN或Inf
        if torch.isnan(noisy_x).any() or torch.isinf(noisy_x).any():
            # 替换无效值
            noisy_x = torch.where(
                torch.isnan(noisy_x) | torch.isinf(noisy_x),
                x,  # 使用原始图像作为备选
                noisy_x
            )
        
        # 在返回前剪裁到有效范围
        noisy_x = torch.clamp(noisy_x, -10.0, 10.0)
        
        return noisy_x, noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """从噪声预测原始图像"""
        # 确保t是有效索引
        t = torch.clamp(t, 0, self.num_timesteps - 1)
        
        # 获取参数
        sqrt_recip_alphas = self.sqrt_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        sqrt_recipm1_alphas = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        
        # 防止NaN或Inf
        if torch.isnan(sqrt_recip_alphas).any() or torch.isinf(sqrt_recip_alphas).any():
            sqrt_recip_alphas = torch.where(
                torch.isnan(sqrt_recip_alphas) | torch.isinf(sqrt_recip_alphas),
                torch.ones_like(sqrt_recip_alphas) * 0.99,
                sqrt_recip_alphas
            )
        
        # 处理FP16精度
        if x_t.dtype == torch.float16:
            # 使用FP32计算
            x_t_fp32 = x_t.to(torch.float32)
            noise_fp32 = noise.to(torch.float32)
            sqrt_recip_alphas_fp32 = sqrt_recip_alphas.to(torch.float32)
            sqrt_recipm1_alphas_fp32 = sqrt_recipm1_alphas.to(torch.float32)
            
            x_0 = (x_t_fp32 - sqrt_recipm1_alphas_fp32 * noise_fp32) / sqrt_recip_alphas_fp32
            
            # 恢复原始精度
            x_0 = x_0.to(x_t.dtype)
        else:
            # 标准计算
            x_0 = (x_t - sqrt_recipm1_alphas * noise) / sqrt_recip_alphas
        
        # 确保结果不包含NaN或Inf
        if torch.isnan(x_0).any() or torch.isinf(x_0).any():
            # 使用x_t作为备选，防止无效值
            x_0 = torch.where(
                torch.isnan(x_0) | torch.isinf(x_0),
                x_t,
                x_0
            )
        
        # 保证预测值在合理范围内
        x_0 = torch.clamp(x_0, -1.0, 1.0)
        
        return x_0
    
    def q_posterior(self, x_0, x_t, t):
        """计算后验分布参数"""
        # 确保t是有效索引
        t = torch.clamp(t, 0, self.num_timesteps - 1)
        
        # 获取参数
        alpha_cumprod = self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        alpha_cumprod_prev = self.alphas_cumprod_prev.gather(-1, t).reshape(-1, 1, 1, 1)
        beta = self.betas.gather(-1, t).reshape(-1, 1, 1, 1)
        
        # 防止NaN或Inf
        if torch.isnan(alpha_cumprod).any() or torch.isinf(alpha_cumprod).any():
            alpha_cumprod = torch.where(
                torch.isnan(alpha_cumprod) | torch.isinf(alpha_cumprod),
                torch.ones_like(alpha_cumprod) * 0.99,
                alpha_cumprod
            )
        
        if torch.isnan(alpha_cumprod_prev).any() or torch.isinf(alpha_cumprod_prev).any():
            alpha_cumprod_prev = torch.where(
                torch.isnan(alpha_cumprod_prev) | torch.isinf(alpha_cumprod_prev),
                torch.ones_like(alpha_cumprod_prev) * 0.99,
                alpha_cumprod_prev
            )
        
        # 处理FP16精度
        if x_0.dtype == torch.float16:
            # 使用FP32计算
            x_0_fp32 = x_0.to(torch.float32)
            x_t_fp32 = x_t.to(torch.float32)
            alpha_cumprod_fp32 = alpha_cumprod.to(torch.float32)
            alpha_cumprod_prev_fp32 = alpha_cumprod_prev.to(torch.float32)
            beta_fp32 = beta.to(torch.float32)
            
            # 计算后验均值
            posterior_mean = (
                alpha_cumprod_prev_fp32 * x_0_fp32 + 
                (1 - alpha_cumprod_prev_fp32) * x_t_fp32
            )
            
            # 计算后验方差
            posterior_variance = (
                (1 - alpha_cumprod_prev_fp32) /
                (1 - alpha_cumprod_fp32) * beta_fp32
            )
            
            # 计算log方差
            posterior_log_variance = torch.log(posterior_variance.clamp(min=1e-20))
            
            # 恢复原始精度
            posterior_mean = posterior_mean.to(x_0.dtype)
            posterior_variance = posterior_variance.to(x_0.dtype)
            posterior_log_variance = posterior_log_variance.to(x_0.dtype)
        else:
            # 标准计算
            # 计算后验均值
            posterior_mean = (
                alpha_cumprod_prev * x_0 + 
                (1 - alpha_cumprod_prev) * x_t
            )
            
            # 计算后验方差
            posterior_variance = (
                (1 - alpha_cumprod_prev) /
                (1 - alpha_cumprod) * beta
            )
            
            # 计算log方差
            posterior_log_variance = torch.log(posterior_variance.clamp(min=1e-20))
        
        # 确保结果不包含NaN或Inf
        if torch.isnan(posterior_mean).any() or torch.isinf(posterior_mean).any():
            posterior_mean = torch.where(
                torch.isnan(posterior_mean) | torch.isinf(posterior_mean),
                x_t,  # 使用x_t作为备选
                posterior_mean
            )
        
        # 保证结果在合理范围内
        posterior_mean = torch.clamp(posterior_mean, -10.0, 10.0)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(self, x, t, img1, img2, mask1, mask2):
        """
        计算p分布的均值和方差
        使用条件输入指导去噪过程
        """
        # 确保所有输入都是有效的张量
        assert torch.is_tensor(x), "x必须是张量"
        assert torch.is_tensor(t), "t必须是张量"
        assert torch.is_tensor(img1), "img1必须是张量"
        assert torch.is_tensor(img2), "img2必须是张量"
        assert torch.is_tensor(mask1), "mask1必须是张量"
        assert torch.is_tensor(mask2), "mask2必须是张量"
        
        # 在尺寸调整前保存原始尺寸
        original_size = x.shape[2:]
        
        # 将img1和mask1调整为固定的512x512大小
        img1_resized = F.interpolate(img1, size=(512, 512), mode='bilinear', align_corners=False)
        mask1_resized = F.interpolate(mask1, size=(512, 512), mode='bilinear', align_corners=False)
        
        # 同样将img2和mask2调整为相同的512x512大小
        img2_resized = F.interpolate(img2, size=(512, 512), mode='bilinear', align_corners=False)
        mask2_resized = F.interpolate(mask2, size=(512, 512), mode='bilinear', align_corners=False)
        
        # 将x调整为512x512
        x_resized = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        
        # 编码条件信息 (img1, img2, mask1, mask2)
        x_input = torch.cat([img1_resized, img2_resized, mask1_resized, mask2_resized], dim=1)
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
        
        # 最终预测
        h = self.final_norm(h)
        h = self.final_act(h)
        
        # 预测噪声
        predicted_noise = self.final_conv(h)
        
        # 获取预测的x_0
        x_recon = self.predict_start_from_noise(x_resized, t, predicted_noise)
        
        # 裁剪x_0到合法范围
        x_recon.clamp_(-1., 1.)
        
        # 计算后验参数
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x_resized, t)
        
        # 将结果调整回原始尺寸
        if model_mean.shape[2:] != original_size:
            model_mean = F.interpolate(model_mean, size=original_size, mode='bilinear', align_corners=False)
            predicted_noise = F.interpolate(predicted_noise, size=original_size, mode='bilinear', align_corners=False)
        
        return model_mean, posterior_variance, posterior_log_variance, predicted_noise
    
    def p_sample(self, x, t, img1, img2, mask1, mask2, guidance_scale=1.0):
        """
        使用扩散模型的单步采样
        t是当前的时间步
        """
        # 首先将输入转换为灰度图像
        img1_gray = rgb_to_grayscale(img1)
        img2_gray = rgb_to_grayscale(img2)
        
        # 获取p分布的参数
        model_mean, _, model_log_variance, predicted_noise = self.p_mean_variance(x, t, img1_gray, img2_gray, mask1, mask2)
        
        # 应用增强引导
        if guidance_scale > 1.0:
            # 无条件去噪
            unconditional_mean, _, _, _ = self.p_mean_variance(x, t, torch.zeros_like(img1_gray), torch.zeros_like(img2_gray), torch.zeros_like(mask1), torch.zeros_like(mask2))
            # 应用引导比例
            model_mean = unconditional_mean + guidance_scale * (model_mean - unconditional_mean)
        
        # 添加噪声
        noise = torch.randn_like(x)
        # 仅在t>0时应用噪声
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        # 用模型预测的方差参数对采样进行控制
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    def forward(self, x, t, img1, img2, mask1, mask2):
        """
        模型前向传播 - 训练时使用
        预测添加到图像中的噪声，并返回学习到的掩码
        """
        # 保存原始尺寸，用于最终输出调整
        original_size = x.shape[2:]
        
        # 标准化输入分辨率 - 确保始终是模型期望的分辨率
        expected_size = (self.image_size, self.image_size)
        if original_size != expected_size:
            print(f"调整输入从 {original_size} 到模型标准分辨率 {expected_size}")
            x_resized = F.interpolate(x, size=expected_size, mode='bilinear', align_corners=False)
            img1_resized = F.interpolate(img1, size=expected_size, mode='bilinear', align_corners=False)
            img2_resized = F.interpolate(img2, size=expected_size, mode='bilinear', align_corners=False)
            mask1_resized = F.interpolate(mask1, size=expected_size, mode='bilinear', align_corners=False)
            mask2_resized = F.interpolate(mask2, size=expected_size, mode='bilinear', align_corners=False)
        else:
            x_resized = x
            img1_resized = img1
            img2_resized = img2
            mask1_resized = mask1
            mask2_resized = mask2
        
        # 确保输入值范围正确 - 应该在[-1, 1]之间
        if x_resized.min() < -1.1 or x_resized.max() > 1.1:
            print(f"警告: 输入值范围异常，最小值={x_resized.min():.2f}，最大值={x_resized.max():.2f}，执行归一化")
            x_resized = torch.clamp(x_resized, -1.0, 1.0)
            img1_resized = torch.clamp(img1_resized, -1.0, 1.0)
            img2_resized = torch.clamp(img2_resized, -1.0, 1.0)
        
        # 确保掩码值范围在[0, 1]之间
        mask1_resized = torch.clamp(mask1_resized, 0.0, 1.0)
        mask2_resized = torch.clamp(mask2_resized, 0.0, 1.0)
        
        # 检查批次维度是否正确
        if len(x_resized.shape) != 4:
            raise ValueError(f"输入应为4D张量 [批次,通道,高度,宽度]，但得到了 {x_resized.shape}")
        
        # 确保时间步是正确的形状
        if isinstance(t, int) or isinstance(t, float) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            t = torch.tensor([t], device=x_resized.device)
        
        # 处理条件输入，确保所有张量尺寸一致
        x_cond = torch.cat([img1_resized, img2_resized, mask1_resized, mask2_resized], dim=1)
        x_cond = self.channel_adapter(x_cond)
        
        # 获取时间编码
        t_emb = self.time_mlp(t)
        
        # 分层特征提取和处理
        d1 = self.down1(x_cond, t_emb)
        d2 = self.down2(d1, t_emb)
        d3 = self.down3(d2, t_emb)
        d4 = self.down4(d3, t_emb)
        
        # 中间处理
        h = self.mid_block1(d4, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # 全局注意力增强
        h = self.global_attn(h)
        
        # 上采样路径 - 使用下采样路径的跳跃连接
        h = self.up1(h, d4, t_emb)
        h = self.up2(h, d3, t_emb)
        h = self.up3(h, d2, t_emb)
        h = self.up4(h, d1, t_emb)
        
        # 最终处理
        h = self.final_norm(h)
        h = self.final_act(h)
        
        # 预测噪声
        predicted_noise = self.final_conv(h)
        
        # 生成掩码
        learned_mask = self.mask_branch(h)
        
        # 将结果调整回原始尺寸
        if original_size != expected_size:
            predicted_noise = F.interpolate(predicted_noise, size=original_size, mode='bilinear', align_corners=False)
            learned_mask = F.interpolate(learned_mask, size=original_size, mode='bilinear', align_corners=False)
        
        # 确保返回的是噪声预测和学习到的掩码
        return predicted_noise, learned_mask
    
    def sample(self, img1, img2, mask1, mask2, num_steps=100, guidance_scale=1.0):
        """
        完整扩散采样过程
        """
        # 将输入转换为灰度图像
        img1_gray = rgb_to_grayscale(img1)
        img2_gray = rgb_to_grayscale(img2)
        
        # 初始化随机噪声
        device = img1.device
        b = img1.shape[0]
        
        # 开始采样
        # 使用RGB或者只有3个通道的图像
        x = torch.randn(b, 3, img1.shape[2], img1.shape[3], device=device)
        
        # 保存每个时间步的结果
        intermediate_outputs = []
        
        # 反向扩散采样过程
        timesteps = torch.linspace(0, self.num_timesteps - 1, num_steps, device=device).long().flip(0)
        
        for i, t in enumerate(timesteps):
            # 广播时间步到批次大小
            t_batch = torch.ones(b, device=device).long() * t
            
            # 应用去噪模型进行采样
            with torch.no_grad():
                x = self.p_sample(x, t_batch, img1_gray, img2_gray, mask1, mask2, guidance_scale)
                
            # 保存中间结果
            if i % (num_steps // 10) == 0 or i == num_steps - 1:
                intermediate_outputs.append(x.clone())
        
        # 生成掩码
        x_input = torch.cat([img1_gray, img2_gray, mask1, mask2], dim=1)
        x_input = self.channel_adapter(x_input)
        
        # 时间编码设为0，表示无条件生成
        t_emb = self.time_mlp(torch.zeros(b, device=device).long())
        
        # 编码器路径
        d1 = self.down1(x_input, t_emb)
        d2 = self.down2(d1, t_emb)
        d3 = self.down3(d2, t_emb)
        d4 = self.down4(d3, t_emb)
        
        # 中间块和注意力
        h = self.mid_block1(d4, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        h = self.global_attn(h)
        
        # 解码器路径
        h = self.up1(h, d4, t_emb)
        h = self.up2(h, d3, t_emb)
        h = self.up3(h, d2, t_emb)
        h = self.up4(h, d1, t_emb)
        
        # 生成掩码
        h = self.final_norm(h)
        h = self.final_act(h)
        learned_mask = self.mask_branch(h)
        
        # 使用学习到的掩码和原始彩色图像（而非灰度图像）进行拼接
        learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, learned_mask)
        stitched_image = compose_images(img1, img2, learned_mask1, learned_mask2)
        
        # 返回最终采样、学习到的掩码和拼接结果
        return learned_mask, x, stitched_image
    
    def compute_loss(self, x, t, img1, img2, mask1, mask2):
        """
        计算扩散模型的损失 - 预测噪声和目标噪声之间的MSE损失
        参数:
            x: 初始噪声或输入
            t: 时间步
            img1, img2: 输入图像
            mask1, mask2: 输入掩码
        返回:
            diffusion_loss: 扩散模型的MSE损失
        """
        # 添加前向扩散过程以获取带噪图像和目标噪声
        noisy_x, target_noise = self.forward_diffusion(x, t)
        
        # 使用模型预测噪声
        predicted_noise, _ = self.forward(noisy_x, t, img1, img2, mask1, mask2)
        
        # 计算MSE损失
        diffusion_loss = F.mse_loss(predicted_noise, target_noise)
        
        return diffusion_loss


def generate_learned_masks(mask1, mask2, out):
    """
    生成学习到的掩码，带有强大的维度检查和错误处理
        
        Args:
        mask1: 第一个输入掩码
        mask2: 第二个输入掩码
        out: 模型预测的掩码/融合系数
            
        Returns:
        learned_mask1, learned_mask2: 生成的两个掩码
    """
    try:
        # 检查并确保所有输入都是有效的张量
        assert torch.is_tensor(mask1), "mask1必须是张量"
        assert torch.is_tensor(mask2), "mask2必须是张量"
        assert torch.is_tensor(out), "out必须是张量"
        
        # 使用mask1的尺寸作为标准
        target_size = mask1.shape[2:]
        
        # 确保target_size是有效的，具有两个维度
        if len(target_size) != 2 or target_size[0] < 1 or target_size[1] < 1:
            # 如果mask1尺寸无效，使用mask2的尺寸
            target_size = mask2.shape[2:]
            # 如果mask2也无效，使用out的尺寸
            if len(target_size) != 2 or target_size[0] < 1 or target_size[1] < 1:
                target_size = out.shape[2:]
                
        # 确保标准尺寸是偶数
        target_h = (target_size[0] // 2) * 2
        target_w = (target_size[1] // 2) * 2
        target_size = (target_h, target_w)
        
        # 将mask1调整为标准尺寸
        if mask1.shape[2:] != target_size:
            mask1 = F.interpolate(mask1, size=target_size, mode='bilinear', align_corners=False)
        
        # 将mask2调整为标准尺寸
        if mask2.shape[2:] != target_size:
            mask2 = F.interpolate(mask2, size=target_size, mode='bilinear', align_corners=False)
        
        # 将out调整到标准尺寸
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        
        # 确保输出是正确的通道数 (1)
        if out.shape[1] != 1:
            # 如果多通道，取平均值
            if out.shape[1] > 1:
                out = out.mean(dim=1, keepdim=True)
                
        # 确保mask1和mask2是单通道
        if mask1.shape[1] != 1:
            mask1 = mask1.mean(dim=1, keepdim=True) if mask1.shape[1] > 1 else mask1
        if mask2.shape[1] != 1:
            mask2 = mask2.mean(dim=1, keepdim=True) if mask2.shape[1] > 1 else mask2
        
        # 确保值域在[0,1]之间，使用安全的方式
        mask1 = torch.clamp(mask1, 0.0, 1.0)
        mask2 = torch.clamp(mask2, 0.0, 1.0)
        
        # 使用sigmoid确保输出在[0,1]之间
        out = torch.sigmoid(out)
        
        # 应用掩码生成公式，使用安全的操作防止数值问题
        overlap = mask1 * mask2
        non_overlap1 = mask1 - overlap
        non_overlap2 = mask2 - overlap
        
        # 创建学习的掩码
        learned_mask1 = non_overlap1 + overlap * out
        learned_mask2 = non_overlap2 + overlap * (1-out)
        
        # 确保生成的掩码之和接近1，防止亮度异常
        sum_masks = learned_mask1 + learned_mask2
        
        # 避免除以零，使用安全的除法
        eps = 1e-6
        sum_masks = torch.clamp(sum_masks, min=eps)
        
        # 标准化掩码
        learned_mask1 = learned_mask1 / sum_masks
        learned_mask2 = learned_mask2 / sum_masks
        
        # 检查并处理NaN或Inf
        if torch.isnan(learned_mask1).any() or torch.isinf(learned_mask1).any() or \
           torch.isnan(learned_mask2).any() or torch.isinf(learned_mask2).any():
            print("警告: 生成的掩码包含NaN或Inf，使用安全的替代值")
            learned_mask1 = torch.where(
                torch.isnan(learned_mask1) | torch.isinf(learned_mask1),
                mask1,  # 使用原始mask1作为替代
                learned_mask1
            )
            learned_mask2 = torch.where(
                torch.isnan(learned_mask2) | torch.isinf(learned_mask2),
                mask2,  # 使用原始mask2作为替代
                learned_mask2
            )
            
        # 再次确保范围在[0,1]之间
        learned_mask1 = torch.clamp(learned_mask1, 0.0, 1.0)
        learned_mask2 = torch.clamp(learned_mask2, 0.0, 1.0)
        
        return learned_mask1, learned_mask2
        
    except Exception as e:
        print(f"生成学习掩码时出错: {e}")
        print(f"输入形状: mask1={mask1.shape}, mask2={mask2.shape}, out={out.shape}")
        
        # 返回原始掩码作为备选
        try:
            # 尝试使用原始掩码，但确保它们是单通道并在[0,1]范围内
            if mask1.shape[1] != 1 and mask1.shape[1] > 1:
                mask1 = mask1.mean(dim=1, keepdim=True)
            if mask2.shape[1] != 1 and mask2.shape[1] > 1:
                mask2 = mask2.mean(dim=1, keepdim=True)
                
            mask1 = torch.clamp(mask1, 0.0, 1.0)
            mask2 = torch.clamp(mask2, 0.0, 1.0)
            return mask1, mask2
        except Exception:
            # 如果仍然出错，创建安全的掩码
            batch_size = out.shape[0] if hasattr(out, 'shape') else 1
            device = out.device if hasattr(out, 'device') else \
                    (mask1.device if hasattr(mask1, 'device') else 'cpu')
                    
            size = (64, 64) if not hasattr(out, 'shape') or len(out.shape) < 3 else out.shape[2:]
            safe_mask1 = torch.ones((batch_size, 1, size[0], size[1]), device=device) * 0.5
            safe_mask2 = torch.ones((batch_size, 1, size[0], size[1]), device=device) * 0.5
            return safe_mask1, safe_mask2


def compose_images(warp1, warp2, mask1, mask2, network_output):
    """将两张图像根据掩码合成为一张，带有安全检查和错误处理
        
        Args:
        warp1, warp2: 两张要组合的图像
        mask1, mask2: 输入的掩码
        network_output: 网络生成的融合系数/掩码
            
        Returns:
        合成后的图像、学习的掩码1、学习的掩码2
    """
    try:
        # 确保所有输入都在相同的设备上
        device = warp1.device
        
        # 获取学习到的掩码
        learned_mask1, learned_mask2 = generate_learned_masks(mask1, mask2, network_output)
        
        # 确保掩码通道数为1，如果需要
        if learned_mask1.shape[1] > 1 and warp1.shape[1] != learned_mask1.shape[1]:
            learned_mask1 = learned_mask1[:, 0:1]
        if learned_mask2.shape[1] > 1 and warp2.shape[1] != learned_mask2.shape[1]:
            learned_mask2 = learned_mask2[:, 0:1]
            
        # 如果图像通道数大于1，但掩码只有1个通道，则复制掩码
        if warp1.shape[1] > 1 and learned_mask1.shape[1] == 1:
            learned_mask1 = learned_mask1.repeat(1, warp1.shape[1], 1, 1)
        if warp2.shape[1] > 1 and learned_mask2.shape[1] == 1:
            learned_mask2 = learned_mask2.repeat(1, warp2.shape[1], 1, 1)
        
        # 确保空间尺寸一致
        if warp1.shape[2:] != learned_mask1.shape[2:]:
            learned_mask1 = F.interpolate(learned_mask1, size=warp1.shape[2:], mode='bilinear', align_corners=False)
        if warp2.shape[2:] != learned_mask2.shape[2:]:
            learned_mask2 = F.interpolate(learned_mask2, size=warp2.shape[2:], mode='bilinear', align_corners=False)
        
        # 计算合成图像
        alpha1 = learned_mask1
        alpha2 = learned_mask2
        
        # 确保alpha1 + alpha2 接近于1
        sum_alpha = alpha1 + alpha2
        alpha1 = alpha1 / (sum_alpha + 1e-7)
        alpha2 = alpha2 / (sum_alpha + 1e-7)
        
        # 安全检查掩码数值范围
        alpha1 = torch.clamp(alpha1, 0, 1)
        alpha2 = torch.clamp(alpha2, 0, 1)
        
        # 合成最终图像
        composed_img = warp1 * alpha1 + warp2 * alpha2
        
        return composed_img, learned_mask1, learned_mask2
        
    except Exception as e:
        print(f"图像合成过程中出错: {str(e)}")
        # 尝试恢复并返回简单合成图像
        try:
            # 强制设置掩码为0.5各占一半
            h, w = warp1.shape[2], warp1.shape[3]
            learned_mask1 = torch.ones((warp1.shape[0], 1, h, w), device=device) * 0.5
            learned_mask2 = torch.ones((warp1.shape[0], 1, h, w), device=device) * 0.5
            
            if warp1.shape[1] > 1:
                learned_mask1 = learned_mask1.repeat(1, warp1.shape[1], 1, 1)
                learned_mask2 = learned_mask2.repeat(1, warp1.shape[1], 1, 1)
                
            # 简单平均
            composed_img = (warp1 + warp2) / 2.0
            
            return composed_img, learned_mask1, learned_mask2
        except:
            # 如果还是失败，返回第一个图像
            return warp1, torch.ones_like(warp1), torch.zeros_like(warp1)


class MaskLoss(nn.Module):
    """掩码损失函数，针对AMP训练安全
    
    使用BCEWithLogitsLoss代替普通BCELoss，更适合自动混合精度训练
    """
    def __init__(self, reduction='mean'):
        super(MaskLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
        
    def forward(self, pred_mask, target_mask):
        """计算掩码损失
        
        Args:
            pred_mask: 预测掩码，可以是带或不带sigmoid的logits
            target_mask: 目标掩码，值范围为[0,1]
            
        Returns:
            计算的损失值
        """
        try:
            # 确保尺寸一致
            if pred_mask.shape[2:] != target_mask.shape[2:]:
                target_mask = F.interpolate(target_mask, size=pred_mask.shape[2:], mode='bilinear', align_corners=False)
                
            # 确保通道数匹配
            if pred_mask.shape[1] != target_mask.shape[1]:
                if pred_mask.shape[1] == 1 and target_mask.shape[1] > 1:
                    # 如果预测是单通道但目标是多通道，取目标的平均
                    target_mask = target_mask.mean(dim=1, keepdim=True)
                elif pred_mask.shape[1] > 1 and target_mask.shape[1] == 1:
                    # 如果预测是多通道但目标是单通道，复制目标到多通道
                    target_mask = target_mask.repeat(1, pred_mask.shape[1], 1, 1)
                    
            # 确保数值范围正确
            target_mask = torch.clamp(target_mask, 0, 1)
            
            # 计算损失
            loss = self.loss_fn(pred_mask, target_mask)
            
            # 检查异常值
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"掩码损失出现NaN或Inf: {loss.item()}")
                return torch.tensor(0.0, device=pred_mask.device)
                
            return loss
            
        except Exception as e:
            print(f"计算掩码损失时出错: {e}")
            print(f"预测掩码: {pred_mask.shape}, 理想掩码: {target_mask.shape}")
            return torch.tensor(0.0, device=pred_mask.device)


def cal_perceptual_loss(stitched_image, warp1, warp2, learned_mask1, learned_mask2):
    # 添加 eps 避免除以零
    eps = 1e-6
    loss = (stitched_image - warp1).abs().sum() / (learned_mask1.sum() + eps)
    loss += (stitched_image - warp2).abs().sum() / (learned_mask2.sum() + eps)
    return loss


def calculate_gradient_consistency_loss(stitched_image, img1, img2, mask):
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
    # ... 其他逻辑 ...


def ensure_on_device(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor) and tensor.device != device:
        return tensor.to(device)
    return tensor


# 添加转换彩色到灰度的函数
def rgb_to_grayscale(image_tensor):
    """
    将彩色图片转换为灰白图片，带有安全检查
    
    Args:
        image_tensor: 形状为[B, C, H, W]的图像张量
        
    Returns:
        灰白图像张量，保持原始形状但内容为灰度
    """
    try:
        # 检查是否为彩色图片（通道数大于1）
        if image_tensor.shape[1] > 1:
            # 使用标准RGB到灰度的转换公式
            # 如果只有3个通道或更少，按RGB处理
            if image_tensor.shape[1] <= 3:
                weights = torch.tensor([0.299, 0.587, 0.114], 
                                      device=image_tensor.device).view(1, 3, 1, 1)
                # 如果只有两个通道，添加零通道
                if image_tensor.shape[1] == 2:
                    zero_channel = torch.zeros_like(image_tensor[:, :1])
                    image_tensor = torch.cat([image_tensor, zero_channel], dim=1)
                # 确保通道维度是3
                rgb = image_tensor[:, :3]
                # 计算加权和
                gray = (rgb * weights).sum(dim=1, keepdim=True)
                else:
                # 如果通道数超过3，取前3个通道作为RGB
                rgb = image_tensor[:, :3]
                weights = torch.tensor([0.299, 0.587, 0.114], 
                                     device=image_tensor.device).view(1, 3, 1, 1)
                gray = (rgb * weights).sum(dim=1, keepdim=True)
                
            # 将灰度值重复到所需的通道数
            if image_tensor.shape[1] > 1:
                grayscale_image = gray.repeat(1, image_tensor.shape[1], 1, 1)
            else:
                grayscale_image = gray
                
            return grayscale_image
        else:
            # 已经是单通道，直接返回
            return image_tensor
    except Exception as e:
        print(f"灰度转换出错: {e}")
        # 返回原始图像作为备选
        return image_tensor


class DimensionValidator(nn.Module):
    """
    用于在网络中验证张量维度的模块
    可以在关键位置插入以确保维度一致性
    """
    def __init__(self, name, expected_shape=None, min_shape=None, check_batch=False):
        """
        初始化维度验证器
        
        Args:
            name: 验证点名称，用于错误信息
            expected_shape: 期望的Shape，不包括批次维度
            min_shape: 最小Shape要求，不包括批次维度
            check_batch: 是否也检查批次维度
        """
        super().__init__()
        self.name = name
        self.expected_shape = expected_shape
        self.min_shape = min_shape
        self.check_batch = check_batch
        self.last_shape = None
        
    def forward(self, x):
        """
        验证输入张量的维度
        
        Args:
            x: 输入张量
            
        Returns:
            原始张量（不做修改）
        """
        try:
            # 记录当前形状
            self.last_shape = x.shape
            
            # 检查批次维度
            if self.check_batch and self.expected_shape is not None and len(self.expected_shape) > 0:
                expected_batch = self.expected_shape[0]
                if x.shape[0] != expected_batch:
                    print(f"维度验证 '{self.name}': 批次维度不匹配 - 预期 {expected_batch}，实际 {x.shape[0]}")
            
            # 检查通道和空间维度
            if self.expected_shape is not None:
                shape_to_check = x.shape[1:] if not self.check_batch else x.shape
                expected = self.expected_shape if not self.check_batch else self.expected_shape
                
                if shape_to_check != expected:
                    print(f"维度验证 '{self.name}': 维度不匹配 - 预期 {expected}，实际 {shape_to_check}")
            
            # 检查最小维度要求
            if self.min_shape is not None:
                shape_to_check = x.shape[1:] if not self.check_batch else x.shape
                min_shape = self.min_shape if not self.check_batch else self.min_shape
                
                for i, (actual, min_val) in enumerate(zip(shape_to_check, min_shape)):
                    if actual < min_val:
                        print(f"维度验证 '{self.name}': 维度 {i} 小于最小要求 - 最小 {min_val}，实际 {actual}")
        except Exception as e:
            print(f"维度验证出错: {e}")
            
        # 不修改数据，直接返回
        return x


# 推荐在网络中关键的维度转换点添加此验证器，例如在残差连接前、特征融合前等
# 示例用法:
# self.dim_check_1 = DimensionValidator("Encoder-Block1-Output", expected_shape=(64, 128, 128))
# x = self.conv1(x)
# x = self.dim_check_1(x)  # 放在需要验证的位置


