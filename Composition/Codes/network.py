import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import gcd


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
    构建模型输出
    
    Args:
        net: 网络模型
        warp1_tensor: 第一张扭曲图像
        warp2_tensor: 第二张扭曲图像
        mask1_tensor: 第一张图像的掩码
        mask2_tensor: 第二张图像的掩码
        
    Returns:
        包含学习掩码、拼接图像和去噪结果的字典
    """
    # 获取原始输出和扩散优化结果
    out, denoised = net(warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
    
    # 使用扩散优化结果进行融合
    # 在重叠区域根据学习的掩码进行混合，非重叠区域保持原样
    overlap_region = mask1_tensor * mask2_tensor
    learned_mask1 = (mask1_tensor - overlap_region) + overlap_region * out
    learned_mask2 = (mask2_tensor - overlap_region) + overlap_region * (1-out)
    
    # 使用扩散优化结果生成最终拼接图像
    # 将范围从[-1,1]转换回处理
    stitched_image = denoised * learned_mask1 + denoised * learned_mask2
    
    # 边界处理：确保掩码总和为1，避免过亮区域
    mask_sum = learned_mask1 + learned_mask2
    mask_sum = torch.clamp(mask_sum, min=1.0)  # 避免除零
    
    # 归一化掩码
    learned_mask1 = learned_mask1 / mask_sum
    learned_mask2 = learned_mask2 / mask_sum
    
    # 重新计算拼接图像
    stitched_image = denoised * learned_mask1 + denoised * learned_mask2
    
    out_dict = {
        'learned_mask1': learned_mask1,
        'learned_mask2': learned_mask2,
        'stitched_image': stitched_image,
        'denoised': denoised
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
    """
    自注意力模块，用于增强特征提取能力
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
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
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 重塑张量用于注意力计算
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # b, hw, c
        k = k.reshape(b, c, h * w)  # b, c, hw
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # b, hw, c
        
        # 计算注意力
        scale = 1 / math.sqrt(c)
        attn = torch.bmm(q, k) * scale  # b, hw, hw
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.bmm(attn, v)  # b, hw, c
        out = out.permute(0, 2, 1).reshape(b, c, h, w)  # b, c, h, w
        
        # 输出投影
        out = self.proj(out)
        
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
    """
    改进的上采样块，包含注意力机制和时间步条件
    """
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

    def forward(self, x1, x2, emb=None):
        # 首先检查尺寸是否需要调整
        if x1.shape[2:] != x2.shape[2:]:
            # 将x1上采样到与x2匹配的尺寸
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        
        # 确保通道数一致性，如果存在维度不匹配，可以调整
        if x1.shape[1] != x2.shape[1] * 2:
            # 调整x1的通道数
            x1_channels = x1.shape[1]
            x2_channels = x2.shape[1]
            if x1_channels > x2_channels:
                # 如果x1通道数更多，可以裁剪或投影
                x1 = x1[:, :x2_channels, :, :]
            else:
                # 否则可能需要填充或复制
                padding = torch.zeros(x1.shape[0], x2_channels - x1_channels, *x1.shape[2:], device=x1.device)
                x1 = torch.cat([x1, padding], dim=1)
        
        # 特征融合
        x = torch.cat([x2, x1], dim=1)
        
        # 残差连接
        residual = self.residual(x)
        
        # 主要卷积
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
        
        # 添加注意力（如果启用）
        if self.use_attention:
            x = self.attention(x)
        
        # 残差连接
        return x + residual


class SelfAttention(nn.Module):
    """
    自注意力模块，用于增强特征提取能力
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # 计算合适的组数 (确保channels能被groups整除)
        groups = calculate_groups(channels)
        
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
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
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 重塑张量用于注意力计算
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # b, hw, c
        k = k.reshape(b, c, h * w)  # b, c, hw
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # b, hw, c
        
        # 计算注意力
        scale = 1 / math.sqrt(c)
        attn = torch.bmm(q, k) * scale  # b, hw, hw
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.bmm(attn, v)  # b, hw, c
        out = out.permute(0, 2, 1).reshape(b, c, h, w)  # b, c, h, w
        
        # 输出投影
        out = self.proj(out)
        
        return out + residual


class ResBlock(nn.Module):
    """
    残差块，包含时间步条件
    """
    def __init__(self, in_channels, out_channels, time_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 计算合适的组数 (确保out_channels能被groups整除)
        groups = calculate_groups(out_channels)
        
        # 主要卷积路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        
        # 时间嵌入条件
        self.time_dim = time_dim
        if time_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_dim, out_channels),
                nn.SiLU()
            )
        
        # 残差连接，处理通道数不匹配情况
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(groups, out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t=None):
        # 保存残差连接输入
        residual = self.shortcut(x)
        
        # 主要卷积路径
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # 添加时间嵌入（如果提供）
        if t is not None and self.time_dim is not None:
            time_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
            h = h + time_emb
            
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        
        # 添加残差连接
        return h + residual


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
        """向前扩散过程"""
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = self._extract(self.alphas_cumprod.sqrt(), t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract((1. - self.alphas_cumprod).sqrt(), t, x.shape)
        
        # 在时间步t的噪声图像
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
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


class ImprovedDiffusionComposition(nn.Module):
    """
    改进的扩散图像融合模型
    """
    def __init__(self, image_channels=3, embedding_dim=128, device="cuda", diffusion_params=None):
        super(ImprovedDiffusionComposition, self).__init__()
        self.device = device
        self.image_channels = image_channels
        self.embedding_dim = embedding_dim
        
        # 扩散模型 - 确保时间维度匹配
        self.diffusion = ImprovedDiffusionModel(
            image_size=256, 
            in_channels=image_channels, 
            time_dim=embedding_dim,  # 使用相同的embedding_dim
            device=device
        ).to(device)  # 显式移动到device
        
        # 添加通道适配器 - 将连接的图像通道(2*image_channels)转换为模型期望的通道数(image_channels)
        self.channel_adapter = nn.Conv2d(2*image_channels, image_channels, kernel_size=1, padding=0)
        
        # 扩散参数设置
        if diffusion_params is not None:
            self.beta_start = diffusion_params.get('beta_start', 1e-4)
            self.beta_end = diffusion_params.get('beta_end', 0.02)
            self.num_timesteps = diffusion_params.get('num_timesteps', 1000)
        else:
            self.beta_start = 1e-4
            self.beta_end = 0.02
            self.num_timesteps = 1000
        
        # 预计算扩散参数
        self.register_buffer('betas', self._linear_beta_schedule())
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # 掩码生成网络 - 使用UNet架构
        self.mask_generator = nn.Sequential(
            nn.Conv2d(image_channels*2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # 显式移动所有模块到指定设备
        self.to(device)
        
        # 打印初始化信息以便调试
        print(f"ImprovedDiffusionComposition initialized on {device}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Image channels: {image_channels}")
        print(f"Diffusion timesteps: {self.num_timesteps}")
    
    def _linear_beta_schedule(self):
        """线性 beta 调度"""
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=self.device)
    
    def _extract(self, a, t, x_shape):
        """从 alpha 或其他缓冲区中提取特定时间步的值"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def forward_diffusion(self, x, t):
        """向前扩散过程，添加噪声到输入图像"""
        x = x.to(self.device)
        t = t.to(self.device)
        
        noise = torch.randn_like(x, device=self.device)
        sqrt_alphas_cumprod_t = self._extract(self.alphas_cumprod.sqrt(), t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract((1. - self.alphas_cumprod).sqrt(), t, x.shape)
        
        # 在时间步t的噪声图像
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def compute_loss(self, x, y, t=None):
        """计算扩散模型损失"""
        x = x.to(self.device)
        y = y.to(self.device)
        
        batch_size = x.shape[0]
        
        # 如果没有提供时间步，则随机采样
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        else:
            t = t.to(self.device)
        
        # 对两个输入图像添加噪声
        noisy_x, noise_x = self.forward_diffusion(x, t)
        noisy_y, noise_y = self.forward_diffusion(y, t)
        
        # 拼接两个噪声图像作为输入
        noisy_input = torch.cat([noisy_x, noisy_y], dim=1)
        
        # 预测噪声
        predicted_noise = self.diffusion(noisy_input, t)
        
        # 计算MSE损失
        loss = F.mse_loss(predicted_noise, noise_x)  # 这里我们主要关注对第一张图像的噪声预测
        
        return loss
    
    def sample(self, x, y, mask1=None, mask2=None, num_steps=100, guidance_scale=1.0, use_ddim=True, timesteps=None):
        """从噪声采样生成合成图像，支持分类引导和DDIM采样
        
        Args:
            x: 第一张输入图像
            y: 第二张输入图像
            mask1: 第一张图像的掩码（可选）
            mask2: 第二张图像的掩码（可选）
            num_steps: 采样步数
            guidance_scale: 分类引导强度，1.0表示无引导
            use_ddim: 是否使用DDIM采样（更快、质量更好）
            timesteps: 采样时间步数（向后兼容）
            
        Returns:
            learned_mask1: 学习到的掩码1
            denoised: 去噪后的图像
            stitched_image: 合成的图像
        """
        x = x.to(self.device)
        y = y.to(self.device)
        if mask1 is not None:
            mask1 = mask1.to(self.device)
        if mask2 is not None:
            mask2 = mask2.to(self.device)
        
        batch_size = x.shape[0]
        device = self.device
        
        # 使用 timesteps 参数（如果提供）或默认 num_steps
        steps_count = timesteps if timesteps is not None else num_steps
        
        # 创建时间步
        if use_ddim:
            # DDIM采样使用均匀分布的时间步
            steps = torch.linspace(0, self.num_timesteps - 1, steps_count, dtype=torch.long, device=device)
            alpha_cumprod = self.alphas_cumprod
        else:
            # DDPM采样
            steps = torch.linspace(0, self.num_timesteps - 1, steps_count, dtype=torch.long, device=device)
        
        # 初始化随机噪声作为起点
        noise = torch.randn_like(x, device=device)
        denoised = noise
        
        # 逆向扩散过程
        for i in reversed(range(0, len(steps))):
            t = torch.full((batch_size,), steps[i], device=device, dtype=torch.long)
            
            # 获取当前时间步参数
            alpha_t = self._extract(self.alphas, t, x.shape)
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
            
            # 预测噪声 - 添加分类引导
            # 使用通道适配器处理输入
            concat_input = torch.cat([denoised, y], dim=1)
            adapted_input = self.channel_adapter(concat_input)
            predicted_noise = self.diffusion(adapted_input, t)
            
            if guidance_scale > 1.0:
                # 计算无条件预测
                # 使用通道适配器处理无条件输入
                unconditional_input = torch.cat([denoised, torch.zeros_like(y, device=device)], dim=1)
                adapted_unconditional = self.channel_adapter(unconditional_input)
                unconditional_noise = self.diffusion(adapted_unconditional, t)
                
                # 应用分类引导
                predicted_noise = unconditional_noise + guidance_scale * (predicted_noise - unconditional_noise)
            
            # 确保预测噪声和去噪图像具有相同的尺寸
            if predicted_noise.shape != denoised.shape:
                print(f"尺寸不匹配: denoised={denoised.shape}, predicted_noise={predicted_noise.shape}")
                # 将预测的噪声调整为与denoised相同的尺寸
                predicted_noise = F.interpolate(
                    predicted_noise, 
                    size=denoised.shape[2:], 
                    mode='bilinear', 
                    align_corners=True
                )
            
            # 计算去噪步骤
            if use_ddim and i > 0:
                # DDIM采样
                alpha_cumprod_s = self._extract(self.alphas_cumprod, torch.tensor([steps[i-1]], device=device), x.shape)
                
                # 预测x_0
                pred_x0 = (denoised - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                
                # 预测下一步
                denoised = torch.sqrt(alpha_cumprod_s) * pred_x0 + torch.sqrt(1 - alpha_cumprod_s) * predicted_noise
            else:
                # DDPM采样
                if i > 0:
                    beta_t = self._extract(self.betas, t, x.shape)
                    alpha_cumprod_prev_t = self._extract(
                        torch.cat([self.alphas_cumprod[0:1], self.alphas_cumprod[:-1]]), 
                        t, x.shape
                    )
                    
                    # 计算方差
                    variance = beta_t * (1. - alpha_cumprod_prev_t) / (1. - alpha_cumprod_t)
                    noise_factor = torch.randn_like(x, device=device) * torch.sqrt(variance)
                else:
                    noise_factor = 0.
                
                # 更新去噪图像
                denoised = 1 / torch.sqrt(alpha_t) * (
                    denoised - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
                ) + noise_factor
        
        # 计算学习到的混合掩码
        x = x.to(device)
        y = y.to(device)
        concatenated = torch.cat([x, y], dim=1)
        mask = self.mask_generator(concatenated)
        learned_mask1 = mask[:, 0:1, :, :]
        
        # 使用掩码合成最终图像
        if mask1 is not None and mask2 is not None:
            overlap_region = mask1 * mask2
            # 在重叠区域使用学习的掩码，非重叠区域保持原样
            learned_mask1 = (mask1 - overlap_region) + overlap_region * learned_mask1
            
            # 使用掩码合成最终图像
            stitched_image = x * learned_mask1 + y * (1 - learned_mask1)
        else:
            # 如果没有提供掩码，则直接使用学习的掩码
            stitched_image = x * learned_mask1 + y * (1 - learned_mask1)
        
        # 返回格式与修改后的build_model函数一致
        return learned_mask1, denoised, stitched_image

    def forward(self, warp1, warp2, mask1=None, mask2=None):
        """模型前向传播，生成拼接图像和掩码
        
        Args:
            warp1: 第一张输入图像
            warp2: 第二张输入图像
            mask1: 第一张图像的掩码（可选）
            mask2: 第二张图像的掩码（可选）
            
        Returns:
            out: 掩码的软分割 (值域 [0,1])
            denoised: 去噪后的图像
        """
        # 确保所有输入在正确的设备上，使用non_blocking加速传输
        warp1 = warp1.to(self.device, non_blocking=True)
        warp2 = warp2.to(self.device, non_blocking=True)
        if mask1 is not None:
            mask1 = mask1.to(self.device, non_blocking=True)
        if mask2 is not None:
            mask2 = mask2.to(self.device, non_blocking=True)
            
        batch_size = warp1.shape[0]
        
        # 使用torch.cuda.amp.autocast提高性能
        with torch.cuda.amp.autocast(enabled=True):
            # 1. 随机选择时间步，作为训练样本
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
            
            # 2. 对图像添加噪声
            noisy_warp1, _ = self.forward_diffusion(warp1, t)
            
            # 3. 确保两个输入图像尺寸一致
            if noisy_warp1.shape[2:] != warp2.shape[2:]:
                warp2 = F.interpolate(warp2, size=noisy_warp1.shape[2:], mode='bilinear', align_corners=True)
            
            # 4. 拼接输入并通过通道适配器
            x_input = torch.cat([noisy_warp1, warp2], dim=1)
            x_input = self.channel_adapter(x_input)
            
            # 5. 预测噪声并去噪
            try:
                predicted_noise = self.diffusion(x_input, t)
                
                # 6. 确保预测的噪声与输入形状匹配
                if predicted_noise.shape != noisy_warp1.shape:
                    predicted_noise = F.interpolate(predicted_noise, 
                                               size=noisy_warp1.shape[2:], 
                                               mode='bilinear', 
                                               align_corners=True)
                
                # 6. 计算去噪图像
                alpha_t = self._extract(self.alphas, t, warp1.shape)
                alpha_cumprod_t = self._extract(self.alphas_cumprod, t, warp1.shape)
                
                # 使用预测的噪声恢复无噪声图像
                denoised = (noisy_warp1 - torch.sqrt(1. - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                
            except RuntimeError as e:
                if "Sizes of tensors must match" in str(e):
                    # 如果发生尺寸不匹配错误，打印更多诊断信息
                    print(f"尺寸不匹配错误: {e}")
                    print(f"x_input shape: {x_input.shape}")
                    print(f"noisy_warp1 shape: {noisy_warp1.shape}")
                    print(f"warp2 shape: {warp2.shape}")
                    
                    # 重设所有张量的大小到固定尺寸
                    target_size = (256, 256)  # 或其他合适的尺寸
                    x_input_resized = F.interpolate(x_input, size=target_size, mode='bilinear', align_corners=True)
                    noisy_warp1_resized = F.interpolate(noisy_warp1, size=target_size, mode='bilinear', align_corners=True)
                    
                    # 重新尝试预测
                    predicted_noise = self.diffusion(x_input_resized, t)
                    
                    # 确保噪声与调整后的输入形状匹配
                    if predicted_noise.shape[2:] != noisy_warp1_resized.shape[2:]:
                        predicted_noise = F.interpolate(predicted_noise, 
                                                  size=noisy_warp1_resized.shape[2:], 
                                                  mode='bilinear', 
                                                  align_corners=True)
                    
                    # 计算去噪图像（使用调整大小后的版本）
                    alpha_t = self._extract(self.alphas, t, noisy_warp1_resized.shape)
                    alpha_cumprod_t = self._extract(self.alphas_cumprod, t, noisy_warp1_resized.shape)
                    
                    denoised = (noisy_warp1_resized - torch.sqrt(1. - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                    
                    # 最后将去噪图像恢复到原始大小
                    denoised = F.interpolate(denoised, size=warp1.shape[2:], mode='bilinear', align_corners=True)
                else:
                    raise e
        
        # 7. 确保生成拼接掩码的输入尺寸一致
        if warp1.shape[2:] != warp2.shape[2:]:
            warp2 = F.interpolate(warp2, size=warp1.shape[2:], mode='bilinear', align_corners=True)
        
        combined_input = torch.cat([warp1, warp2], dim=1)
        out = self.mask_generator(combined_input)
        
        # 优化内存使用，清理不再需要的中间变量
        torch.cuda.empty_cache()
        
        return out, denoised


