import os
import sys
import torch
import torch.nn as nn
import argparse
from PIL import Image
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import cv2
from collections import defaultdict

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
composition_path = os.path.join(project_root, "Composition")
composition_codes_path = os.path.join(composition_path, "Codes")
sys.path.insert(0, composition_path)
sys.path.insert(0, composition_codes_path)

# 导入自定义可视化工具
from draw.utils import FeatureVisualizer

# 尝试导入原始模型代码
try:
    from Composition.Codes.network import ImprovedDiffusionComposition, build_model
    COMPOSITION_MODEL_AVAILABLE = True
    print("成功导入原始Composition模型")
except ImportError as e:
    # 尝试导入，使用相对路径
    try:
        # 使用完整路径导入
        sys.path.insert(0, os.path.join(composition_codes_path, "utils"))
        from Composition.Codes.network import ImprovedDiffusionComposition, build_model
        COMPOSITION_MODEL_AVAILABLE = True
        print("成功导入Composition模型（相对路径）")
    except ImportError as e2:
        print(f"无法导入Composition模型: {e2}")
        COMPOSITION_MODEL_AVAILABLE = False
        traceback.print_exc()
        print("请确保模块可用")
        

# 自定义函数保存可视化表格数据
def save_metrics_to_csv(metrics_dict, filename, output_dir):
    """保存度量数据到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # 创建DataFrame
    df = pd.DataFrame(metrics_dict)
    
    # 保存为CSV
    df.to_csv(filepath, index=False)
    print(f"保存指标数据到: {filepath}")
    return filepath


# 创建可视化图表并保存
def plot_and_save_metrics(metrics_dict, title, filename, output_dir):
    """创建可视化图表并保存"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    for key, values in metrics_dict.items():
        if isinstance(values, list) and len(values) > 1:
            plt.plot(values, label=key)
    
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plt.savefig(filepath)
    plt.close()
    print(f"保存图表到: {filepath}")
    return filepath


class CompositionVisualizer(nn.Module):
    """用于可视化composition过程的包装器类"""
    
    def __init__(self, model, output_dir='draw/output/composition', target_size=None):
        """
        初始化composition可视化包装器
        
        参数:
            model: 原始的ImprovedDiffusionComposition模型
            output_dir: 输出目录
            target_size: 目标图像尺寸 (高度, 宽度)，例如 (256, 256)
        """
        super(CompositionVisualizer, self).__init__()
        self.model = model
        self.visualizer = FeatureVisualizer(save_dir=output_dir)
        self.target_size = target_size  # 保存目标尺寸
        
        # 为每个模块创建钩子
        self.hooks = []
        self.activations = {}
        
        # 钩住扩散采样过程的中间步骤，这需要修改sample方法或创建一个包装方法
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向钩子以捕获中间特征"""
        
        # 定义钩子函数
        def hook_fn(name):
            def fn(module, input, output):
                self.activations[name] = output
            return fn
        
        # 钩住掩码生成器
        if hasattr(self.model, 'mask_generator'):
            hook = self.model.mask_generator.register_forward_hook(hook_fn('mask_generator'))
            self.hooks.append(hook)
        
        # 钩住通道适配器
        if hasattr(self.model, 'channel_adapter'):
            hook = self.model.channel_adapter.register_forward_hook(hook_fn('channel_adapter'))
            self.hooks.append(hook)
        
        # 钩住扩散模型的特定部分
        if hasattr(self.model, 'diffusion'):
            diffusion_model = self.model.diffusion
            
            # 钩住前向扩散过程
            if hasattr(diffusion_model, 'forward'):
                def diffusion_forward_hook(module, input, output):
                    self.activations['diffusion_forward'] = output
                
                diffusion_model.register_forward_hook(diffusion_forward_hook)
    
    def visualize_sample_process(self, warp1, warp2, mask1=None, mask2=None, num_steps=10):
        """
        可视化采样过程
        
        参数:
            warp1: 第一个warp图像
            warp2: 第二个warp图像
            mask1: 第一个掩码（可选）
            mask2: 第二个掩码（可选）
            num_steps: 可视化中间步骤的数量
            
        返回:
            采样结果和中间结果
        """
        # 定义清理内存的函数
        def cleanup():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # 清除上一次的激活
        self.activations = {}
        
        # 创建采样步骤输出目录
        sample_dir = os.path.join(self.visualizer.save_dir, 'sampling_steps')
        os.makedirs(sample_dir, exist_ok=True)
        
        # 创建指标输出目录
        metrics_dir = os.path.join(self.visualizer.save_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # 保存原始采样方法
        original_sample = self.model.sample
        
        # 确保掩码与模型兼容
        mask1, mask2 = self.prepare_masks(mask1, mask2)
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 创建临时存储中间结果的列表
        intermediate_results = []
        selected_timesteps = []
        
        # 创建指标存储字典
        metrics = {
            'timestep': [],
            'noise_level': [],
            'psnr': [],
            'noise_pred_error': [],
            'mask_entropy': []
        }
        
        # 检测是否处于低内存模式（减少步数）
        if num_steps > 5 and getattr(self.model, 'low_memory', False):
            num_steps = 5
            print(f"在低内存模式下，减少可视化步骤数至{num_steps}")
        
        def custom_sample(x, y, mask1=None, mask2=None, num_steps=100, guidance_scale=1.0, use_ddim=True, timesteps=None):
            """临时替换的sample方法，记录中间结果"""
            nonlocal intermediate_results, selected_timesteps, metrics
            
            # 确保x和y具有相同的尺寸
            if x.shape[2:] != y.shape[2:]:
                print(f"警告: x和y的尺寸不匹配: {x.shape} vs {y.shape}")
                # 将y调整为与x相同的尺寸
                y = torch.nn.functional.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
                print(f"已将y调整为: {y.shape}")
            
            # 确保掩码是兼容的尺寸
            if mask1 is not None:
                if mask1.shape[2:] != x.shape[2:]:
                    print(f"警告: mask1与x的尺寸不匹配: {mask1.shape} vs {x.shape}")
                    mask1 = torch.nn.functional.interpolate(mask1, size=x.shape[2:], mode='nearest')
                    print(f"已将mask1调整为: {mask1.shape}")
                if mask1.shape[1] != 1:
                    mask1 = mask1.mean(dim=1, keepdim=True)
                    print(f"将mask1转换为单通道: {mask1.shape}")
            
            if mask2 is not None:
                if mask2.shape[2:] != y.shape[2:]:
                    print(f"警告: mask2与y的尺寸不匹配: {mask2.shape} vs {y.shape}")
                    mask2 = torch.nn.functional.interpolate(mask2, size=y.shape[2:], mode='nearest')
                    print(f"已将mask2调整为: {mask2.shape}")
                if mask2.shape[1] != 1:
                    mask2 = mask2.mean(dim=1, keepdim=True)
                    print(f"将mask2转换为单通道: {mask2.shape}")
            
            # 获取必要参数
            device = x.device
            batch_size = x.shape[0]
            target_shape = x.shape[2:] # 保存目标空间尺寸 H, W
            
            # 初始化变量
            mask1_t = mask1.to(device) if mask1 is not None else torch.ones_like(x[:, :1])
            mask2_t = mask2.to(device) if mask2 is not None else torch.ones_like(y[:, :1])
            overlap = mask1_t * mask2_t
            
            # 使用 timesteps 参数（如果提供）或默认 num_steps
            if timesteps is None:
                timesteps = self.model.num_timesteps
                
            # 使用线性步长
            t = torch.linspace(0, timesteps - 1, num_steps, device=device).long()
            
            # 获取 betas 和派生值
            beta_start, beta_end = self.model.beta_start, self.model.beta_end
            betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
            
            # 为记录中间结果准备
            selected_timesteps = []
            intermediate_results = [] # 确保每次调用都重置
            
            # 初始化
            img = torch.randn_like(x, device=device)
            
            # 保存初始随机噪声作为第一个中间结果
            intermediate_results.append(img.detach().cpu())
            selected_timesteps.append(0)  # 时间步 0 表示初始随机噪声
            
            # 记录初始指标
            metrics['timestep'].append(0)
            metrics['noise_level'].append(1.0)  # 初始噪声级别为100%
            metrics['psnr'].append(0.0)  # 初始PSNR为0
            metrics['noise_pred_error'].append(0.0)
            metrics['mask_entropy'].append(0.0)
            
            # 上一步预测的噪声（用于计算预测误差）
            last_predicted_noise = None
            
            # 计算要记录的步骤
            vis_indices = np.linspace(0, len(t) - 1, min(num_steps, len(t)), dtype=int)

            # 逆向扩散过程 (去噪)
            for i in range(len(t) - 1, -1, -1): # Iterate backwards from T-1 to 0
                time_step = t[i]
                time_tensor = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
                
                # 从模型获取预测的噪声
                try:
                    # 清理内存
                    cleanup()
                    
                    # 准备模型输入 (img和y拼接)
                    concat_input = torch.cat([img, y], dim=1)
                    
                    # 使用通道适配器处理
                    if hasattr(self.model, 'channel_adapter'):
                         adapted_input = self.model.channel_adapter(concat_input)
                    else:
                         # Fallback if adapter not found directly on model
                         temp_adapter = nn.Conv2d(img.shape[1] + y.shape[1], img.shape[1], kernel_size=1).to(device)
                         adapted_input = temp_adapter(concat_input)

                    # 通过U-Net预测噪声
                    predicted_noise = self.model.diffusion(adapted_input, time_tensor)

                    # --- 关键修改：确保predicted_noise尺寸与img匹配 ---
                    if predicted_noise.shape[2:] != target_shape:
                        print(f"警告: 步骤 {time_step}, predicted_noise 尺寸 {predicted_noise.shape[2:]} 与目标 {target_shape} 不匹配。正在调整...")
                        predicted_noise = torch.nn.functional.interpolate(
                            predicted_noise, size=target_shape, mode='bilinear', align_corners=False
                        )
                        print(f"调整后的 predicted_noise 尺寸: {predicted_noise.shape[2:]}")
                    # --- 结束修改 ---

                    # 计算去噪步骤 (DDIM or DDPM logic needed here)
                    # Simplified DDPM step for now:
                    alpha_t = alphas[time_step]
                    alpha_t_cumprod = alphas_cumprod[time_step]
                    beta_t = betas[time_step]
                    sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
                    sqrt_alpha_t_cumprod_prev = alphas_cumprod[time_step - 1] if time_step > 0 else torch.tensor(1.0, device=device)

                    # 计算 x_0 预测
                    x_0_pred = sqrt_recip_alpha_t * (img - beta_t / sqrt_one_minus_alphas_cumprod[time_step] * predicted_noise)

                    # 计算 x_{t-1}
                    mean_pred = (sqrt_alpha_t_cumprod_prev * beta_t / (1.0 - alpha_t_cumprod)) * x_0_pred + \
                                (torch.sqrt(alpha_t) * (1.0 - alphas_cumprod[time_step-1]) / (1.0 - alpha_t_cumprod)) * img

                    if time_step > 0:
                        noise = torch.randn_like(img)
                        variance = ((1.0 - alphas_cumprod[time_step-1]) / (1.0 - alpha_t_cumprod)) * beta_t
                        img = mean_pred + torch.sqrt(variance) * noise
                    else:
                        img = mean_pred # Final step

                except Exception as e:
                    print(f"去噪步骤 {time_step} 出错: {e}")
                    traceback.print_exc() # Print full traceback for debugging
                    # If error, maybe keep previous img? Or break? For now, continue.
                    pass
                
                # 记录中间结果 (adjust index logic if iterating backwards)
                current_step_idx = len(t) - 1 - i
                if current_step_idx in vis_indices:
                    intermediate_results.append(img.detach().cpu())
                    selected_timesteps.append(time_step.item())
                    
                    # 更新指标 (example PSNR calculation)
                    metrics['timestep'].append(time_step.item())
                    metrics['noise_level'].append(sqrt_alphas_cumprod[time_step].item()) # Noise level based on alpha_cumprod
                    try:
                        # Clamp for valid PSNR calculation
                        img_clamped = torch.clamp(img, 0, 1)
                        x_clamped = torch.clamp(x, 0, 1)
                        mse = torch.mean((img_clamped.cpu() - x_clamped.cpu()) ** 2).item()
                        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
                        metrics['psnr'].append(psnr)
                    except Exception as e_metric:
                        print(f"计算 PSNR 时出错: {e_metric}")
                        metrics['psnr'].append(0.0)
                    # Add other metrics if needed
                    metrics['noise_pred_error'].append(0.0) # Placeholder
                    metrics['mask_entropy'].append(0.0) # Placeholder
                        
            # 最终去噪结果
            denoised = img
            
            # 生成掩码
            with torch.no_grad():
                # 将warp图像和掩码合并
                concat = torch.cat([denoised, y], dim=1)
                # 使用通道适配器
                if hasattr(self.model, 'channel_adapter'):
                     adapted_input = self.model.channel_adapter(concat)
                else:
                     temp_adapter = nn.Conv2d(denoised.shape[1] + y.shape[1], denoised.shape[1], kernel_size=1).to(device)
                     adapted_input = temp_adapter(concat)

                # 使用掩码生成器 (确保输入是模型期望的)
                # Often mask generator takes concatenated *original* warps or denoised result
                # Assuming it takes denoised + warp2 here based on previous logic
                concat_for_mask = torch.cat([denoised, y], dim=1)
                out = self.model.mask_generator(concat_for_mask)
                
                # 确保out只有一个通道
                if out.shape[1] > 1:
                    print(f"警告: mask_generator 输出 {out.shape[1]} 通道，取第一个通道。")
                    out = out[:, 0:1] # Take first channel
                
                # 计算最终的拼接掩码
                learned_mask1 = (mask1_t - overlap) + overlap * out
                learned_mask2 = (mask2_t - overlap) + overlap * (1-out)

                # Clamp masks to [0, 1]
                learned_mask1 = torch.clamp(learned_mask1, 0, 1)
                learned_mask2 = torch.clamp(learned_mask2, 0, 1)
                
                # 确保掩码通道数与图像通道数匹配 for stitching
                # Use denoised (result of diffusion) and y (warp2) for stitching
                target_channels = denoised.shape[1]
                if learned_mask1.shape[1] != target_channels:
                    learned_mask1_expanded = learned_mask1.repeat(1, target_channels, 1, 1)
                else:
                     learned_mask1_expanded = learned_mask1

                if learned_mask2.shape[1] != y.shape[1]: # Use warp2's channels for mask2
                     learned_mask2_expanded = learned_mask2.repeat(1, y.shape[1], 1, 1)
                else:
                     learned_mask2_expanded = learned_mask2

                stitched_image = denoised * learned_mask1_expanded + y * learned_mask2_expanded # Stitch denoised and warp2
            
            return learned_mask1, denoised, stitched_image
        
        # 记录当前模型的低内存模式设置（如果有的话）
        low_memory = getattr(self.model, 'low_memory', False)
        if hasattr(self.model, 'low_memory'):
            # 暂时关闭低内存模式以便可视化
            setattr(self.model, 'low_memory', False)
        
        try:
            # 设置模型采样方法为自定义函数
            self.model.sample = custom_sample
            
            # 执行采样过程
            print(f"开始采样过程可视化，步数: {num_steps}...")
            learned_mask1, denoised, stitched_image = custom_sample(
                warp1, warp2, mask1, mask2,
                num_steps=100 # Using 100 steps for the inner loop
            )
            
            # 可视化中间步骤 (reverse order if needed due to loop direction)
            intermediate_results.reverse()
            selected_timesteps.reverse()
            for idx, (result, timestep) in enumerate(zip(intermediate_results, selected_timesteps)):
                # 将结果放回设备用于可视化
                result_device = result.to(warp1.device)
                self.visualizer.visualize_tensor(
                    torch.clamp(result_device, 0, 1), # Clamp before visualizing
                    f'sampling_steps/sampling_step_{idx}_t{timestep}'
                )
                # 用完立即删除，减少内存使用
                del result_device
                cleanup()
            
            # 可视化最终结果 (Clamp before visualizing)
            self.visualizer.visualize_tensor(torch.clamp(denoised, 0, 1), 'final_denoised')
            self.visualizer.visualize_masks(learned_mask1, 'final_learned_mask') # Masks should be [0,1]
            self.visualizer.visualize_tensor(torch.clamp(stitched_image, 0, 1), 'final_stitched_image')
            
            # 可视化原始输入与最终输出的比较 (Clamp before visualizing)
            self.visualizer.side_by_side_comparison(
                torch.clamp(warp1, 0, 1), torch.clamp(stitched_image, 0, 1), 'input_vs_output',
                titles=['Input Image 1', 'Stitched Output']
            )
            
            # 可视化中间特征 (use captured activations if any)
            try:
                current_activations = self.activations.copy() # Get activations captured during the custom_sample run
                for name, activation in current_activations.items():
                    if isinstance(activation, torch.Tensor):
                        self.visualizer.visualize_feature_maps(activation, f'sample_{name}') # Prefix with sample_
                        # 用完立即删除，减少内存使用
                        del self.activations[name] # Clean up from main dict
                        cleanup()
            except Exception as e:
                print(f"可视化采样过程中的中间特征时出错: {e}")
                traceback.print_exc()
            
            # 保存度量指标到CSV
            try:
                # Ensure all metric arrays have the same length
                metrics_df = pd.DataFrame(metrics) # Convert directly
                csv_path = save_metrics_to_csv(metrics_df, 'sampling_metrics.csv', metrics_dir) # Pass df

                # 创建并保存度量图表
                plot_and_save_metrics(
                     metrics_df.set_index('timestep')[['psnr', 'noise_level']].to_dict('list'), # Plot PSNR/Noise Level
                     'Sampling Process Metrics',
                     'sampling_metrics.png',
                     metrics_dir
                 )

                print(f"成功可视化了 {len(intermediate_results)} 个采样步骤")
                print(f"指标数据已保存到: {csv_path}")
            except Exception as e:
                print(f"保存度量数据时出错: {e}")
                traceback.print_exc()

        except Exception as e:
            print(f"采样过程中发生错误: {e}")
            traceback.print_exc()
            # 返回空结果
            learned_mask1 = None
            denoised = None
            stitched_image = None
            intermediate_results = []
        finally:
            # 清理临时数据
            for result in intermediate_results:
                del result
            intermediate_results = []
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 恢复低内存模式设置
            if 'low_memory' in locals() and low_memory != getattr(self.model, 'low_memory', False):
                setattr(self.model, 'low_memory', low_memory)
                
            # 恢复原始采样方法
            self.model.sample = original_sample
            # Reset metrics and activations for next call
            self.activations = {}
        
        return learned_mask1, denoised, stitched_image, intermediate_results
    
    def remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def prepare_masks(self, mask1, mask2):
        """
        处理掩码，确保它们是单通道的并且与模型兼容
        
        参数:
            mask1: 第一个掩码
            mask2: 第二个掩码
            
        返回:
            处理后的掩码对
        """
        if mask1 is not None:
            # 记录原始形状以便调试
            original_shape = mask1.shape
            
            # 确保掩码是4D张量 [B,C,H,W]
            if mask1.dim() == 3:
                mask1 = mask1.unsqueeze(0)
                print(f"将mask1从3D扩展为4D: {original_shape} -> {mask1.shape}")
            
            # 如果掩码是多通道图像，取第一个通道或平均值
            if mask1.shape[1] > 1:
                mask1_single = mask1[:, 0:1, :, :]
                print(f"将mask1从{original_shape}转换为单通道{mask1_single.shape}")
                mask1 = mask1_single
                
            # 确保掩码尺寸与图像尺寸匹配
            if hasattr(self, 'target_size'):
                target_h, target_w = self.target_size
                current_h, current_w = mask1.shape[2], mask1.shape[3]
                
                if current_h != target_h or current_w != target_w:
                    print(f"调整mask1尺寸从 {current_h}x{current_w} 到 {target_h}x{target_w}")
                    mask1 = torch.nn.functional.interpolate(
                        mask1, size=(target_h, target_w), mode='nearest'
                    )
        
        if mask2 is not None:
            # 记录原始形状以便调试
            original_shape = mask2.shape
            
            # 确保掩码是4D张量 [B,C,H,W]
            if mask2.dim() == 3:
                mask2 = mask2.unsqueeze(0)
                print(f"将mask2从3D扩展为4D: {original_shape} -> {mask2.shape}")
            
            # 如果掩码是多通道图像，取第一个通道或平均值
            if mask2.shape[1] > 1:
                mask2_single = mask2[:, 0:1, :, :]
                print(f"将mask2从{original_shape}转换为单通道{mask2_single.shape}")
                mask2 = mask2_single
                
            # 确保掩码尺寸与图像尺寸匹配
            if hasattr(self, 'target_size'):
                target_h, target_w = self.target_size
                current_h, current_w = mask2.shape[2], mask2.shape[3]
                
                if current_h != target_h or current_w != target_w:
                    print(f"调整mask2尺寸从 {current_h}x{current_w} 到 {target_h}x{target_w}")
                    mask2 = torch.nn.functional.interpolate(
                        mask2, size=(target_h, target_w), mode='nearest'
                    )
        
        return mask1, mask2


    def visualize_full_composition(self, warp1, warp2, mask1=None, mask2=None):
        """
        可视化完整的合成过程
        
        参数:
            warp1: 第一个warp图像
            warp2: 第二个warp图像
            mask1: 第一个掩码（可选）
            mask2: 第二个掩码（可选）
        """
        # 清除上一次的激活
        self.activations = {}
        
        # 首先可视化输入 (Clamping added)
        self.visualizer.visualize_tensor(torch.clamp(warp1, 0, 1), 'input_warp1')
        self.visualizer.visualize_tensor(torch.clamp(warp2, 0, 1), 'input_warp2')
        
        # 处理掩码并可视化
        if mask1 is not None:
            # 确保掩码是 [0, 1] 范围
            mask1 = torch.clamp(mask1, 0, 1)
            self.visualizer.visualize_masks(mask1, 'input_mask1')
                
        if mask2 is not None:
            # 确保掩码是 [0, 1] 范围
            mask2 = torch.clamp(mask2, 0, 1)
            self.visualizer.visualize_masks(mask2, 'input_mask2')
        
        # 确保掩码与模型兼容 (prepare_masks already handles channel/size)
        mask1, mask2 = self.prepare_masks(mask1, mask2)
        
        # 创建中间结果目录
        intermediate_dir = os.path.join(self.visualizer.save_dir, 'intermediate')
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # 1. 使用原始forward方法可视化一次前向传播
        forward_learned_mask = None
        forward_denoised = None
        try:
            with torch.no_grad():
                out, denoised = self.model(warp1, warp2, mask1, mask2)
                forward_learned_mask = out
                forward_denoised = denoised
        except Exception as e:
            print(f"模型前向传播失败: {e}")
            traceback.print_exc()

        # 可视化前向传播结果 (Handle mask channels and clamp)
        if forward_learned_mask is not None:
            # --- 关键修改：处理多通道掩码 ---
            if forward_learned_mask.shape[1] > 1:
                print(f"警告: forward_learned_mask 有 {forward_learned_mask.shape[1]} 个通道，取第一个通道进行可视化。")
                mask_to_visualize = forward_learned_mask[:, 0:1, :, :]
            else:
                mask_to_visualize = forward_learned_mask
            # --- 结束修改 ---
            self.visualizer.visualize_masks(torch.clamp(mask_to_visualize, 0, 1), 'forward_learned_mask')
        
        if forward_denoised is not None:
            self.visualizer.visualize_tensor(torch.clamp(forward_denoised, 0, 1), 'forward_denoised')
        
        # 可视化前向传播捕获的激活
        forward_activations = self.activations.copy() # Copy activations from forward pass
        self.activations = {} # Clear for sampling pass
        for name, activation in forward_activations.items():
            if isinstance(activation, torch.Tensor):
                # Visualize features (potentially clamp if they represent image-like data)
                # Assuming feature maps don't need clamping here, but could add if needed
                self.visualizer.visualize_feature_maps(activation, f'forward_{name}')
        
        # 2. 可视化完整的采样过程(慢但更详细)
        # Sampling visualization already handles clamping internally now
        learned_mask1, sample_denoised, stitched_image, _ = self.visualize_sample_process(
            warp1, warp2, mask1, mask2, num_steps=10
        )
        
        # 3. 使用build_model函数可视化最终结果
        build_model_dict = None
        try:
            # 使用安全的build_model函数
            build_model_dict = self.safe_build_model(self.model, warp1, warp2, mask1, mask2)
            
            # 可视化build_model的结果 (Add clamping)
            if build_model_dict and 'error' not in build_model_dict:
                self.visualizer.visualize_masks(torch.clamp(build_model_dict['learned_mask1'], 0, 1), 'build_model_learned_mask1')
                self.visualizer.visualize_masks(torch.clamp(build_model_dict['learned_mask2'], 0, 1), 'build_model_learned_mask2')
                self.visualizer.visualize_tensor(torch.clamp(build_model_dict['stitched_image'], 0, 1), 'build_model_stitched_image')
                self.visualizer.visualize_tensor(torch.clamp(build_model_dict['denoised'], 0, 1), 'build_model_denoised')
                
                # 最终合成图像与原始输入的比较 (Add clamping)
                self.visualizer.side_by_side_comparison(
                    torch.clamp(warp1, 0, 1), torch.clamp(build_model_dict['stitched_image'], 0, 1), 'warp1_vs_stitched',
                    titles=['Original Warp1', 'Stitched Image']
                )
                
                self.visualizer.side_by_side_comparison(
                    torch.clamp(warp2, 0, 1), torch.clamp(build_model_dict['stitched_image'], 0, 1), 'warp2_vs_stitched',
                    titles=['Original Warp2', 'Stitched Image']
                )
        except Exception as e:
            print(f"调用 safe_build_model 或可视化其结果时出错: {e}")
            traceback.print_exc()
            
        return build_model_dict # Return results from build_model if successful

    def safe_build_model(self, model, warp1, warp2, mask1, mask2):
        """
        安全地调用build_model函数，确保掩码维度正确
        
        参数:
            model: 扩散合成模型
            warp1: 第一个warp图像
            warp2: 第二个warp图像
            mask1: 第一个掩码（可选）
            mask2: 第二个掩码（可选）
            
        返回:
            构建模型的结果
        """
        from Composition.Codes.network import build_model
        
        # 确保warp1和warp2具有相同的尺寸
        if warp1.shape[2:] != warp2.shape[2:]:
            print(f"警告: warp1和warp2的尺寸不匹配: {warp1.shape} vs {warp2.shape}")
            # 将warp2调整为与warp1相同的尺寸
            warp2 = torch.nn.functional.interpolate(warp2, size=warp1.shape[2:], mode='bilinear', align_corners=False)
            print(f"已将warp2调整为: {warp2.shape}")
        
        # 确保掩码是单通道的
        mask1, mask2 = self.prepare_masks(mask1, mask2)
        
        # 确保掩码与图像尺寸匹配
        if mask1 is not None and mask1.shape[2:] != warp1.shape[2:]:
            print(f"警告: mask1与warp1的尺寸不匹配: {mask1.shape} vs {warp1.shape}")
            mask1 = torch.nn.functional.interpolate(mask1, size=warp1.shape[2:], mode='nearest')
            print(f"已将mask1调整为: {mask1.shape}")
        
        if mask2 is not None and mask2.shape[2:] != warp2.shape[2:]:
            print(f"警告: mask2与warp2的尺寸不匹配: {mask2.shape} vs {warp2.shape}")
            mask2 = torch.nn.functional.interpolate(mask2, size=warp2.shape[2:], mode='nearest')
            print(f"已将mask2调整为: {mask2.shape}")
        
        try:
            # 从network.py导入并调用build_model函数
            out_dict = build_model(model, warp1, warp2, mask1, mask2)
            # Ensure all output tensors are returned (or handle missing keys gracefully)
            required_keys = ['denoised', 'learned_mask1', 'learned_mask2', 'stitched_image']
            for key in required_keys:
                 if key not in out_dict:
                      print(f"警告: build_model 输出缺少键 '{key}'")
                      out_dict[key] = None # Assign None if missing
            return out_dict
        except RuntimeError as e:
            if "size of tensor" in str(e):
                print(f"掩码维度不匹配，尝试修复...")
                # 打印调试信息
                print(f"warp1形状: {warp1.shape}, warp2形状: {warp2.shape}")
                print(f"mask1形状: {mask1.shape if mask1 is not None else 'None'}")
                print(f"mask2形状: {mask2.shape if mask2 is not None else 'None'}")
                
                # 尝试手动生成融合掩码和拼接图像
                with torch.no_grad():
                    # 前向传播获取特征
                    out, denoised = model(warp1, warp2, mask1, mask2)
                    
                    # 确保掩码是兼容的尺寸
                    if mask1 is None:
                        mask1 = torch.ones_like(warp1[:, :1])
                    if mask2 is None:
                        mask2 = torch.ones_like(warp2[:, :1])
                    
                    # 规范化掩码形状
                    if mask1.shape[1] != 1:
                        mask1 = mask1.mean(dim=1, keepdim=True)
                    if mask2.shape[1] != 1:
                        mask2 = mask2.mean(dim=1, keepdim=True)
                    
                    # 从掩码生成学习到的掩码
                    overlap = mask1 * mask2
                    
                    # 检查out的形状，确保它与掩码兼容
                    if out.shape[1] != 1:
                        print(f"警告: out形状为 {out.shape}，将转换为单通道")
                        if out.shape[1] == 2:  # 特殊情况，取第一个通道
                            out = out[:, 0:1]
                        else:
                            out = out.mean(dim=1, keepdim=True)
                        print(f"调整后的out形状: {out.shape}")
                    
                    learned_mask1 = (mask1 - overlap) + overlap * out
                    learned_mask2 = (mask2 - overlap) + overlap * (1-out)
                    
                    # 打印调试信息
                    print(f"denoised形状: {denoised.shape}")
                    print(f"learned_mask1形状: {learned_mask1.shape}")
                    print(f"learned_mask2形状: {learned_mask2.shape}")
                    
                    # 确保掩码是单通道，然后扩展以匹配图像通道数
                    if learned_mask1.shape[1] > 1:
                        learned_mask1 = learned_mask1[:, :1]
                    if learned_mask2.shape[1] > 1:
                        learned_mask2 = learned_mask2[:, :1]
                    
                    # 扩展掩码通道以匹配图像通道数
                    learned_mask1_expanded = learned_mask1.repeat(1, denoised.shape[1], 1, 1)
                    learned_mask2_expanded = learned_mask2.repeat(1, warp2.shape[1], 1, 1)
                    
                    # 确认维度匹配
                    print(f"扩展后的learned_mask1形状: {learned_mask1_expanded.shape}")
                    print(f"扩展后的learned_mask2形状: {learned_mask2_expanded.shape}")
                    
                    # 使用denoised和warp2进行拼接
                    stitched_image = denoised * learned_mask1_expanded + warp2 * learned_mask2_expanded
                    
                    manual_result = {
                        'denoised': denoised, 
                        'learned_mask1': learned_mask1, 
                        'learned_mask2': learned_mask2, 
                        'stitched_image': stitched_image
                    }
                    # Check for None values
                    for key, val in manual_result.items():
                         if val is None:
                              print(f"警告: 手动生成结果缺少 '{key}'")
                    return manual_result
            raise


def load_images(image_paths, device, target_size=(256, 256), is_mask=False):
    """
    加载并预处理图像
    
    参数:
        image_paths: 图像路径列表
        device: 计算设备
        target_size: 目标尺寸
        is_mask: 是否加载掩码（如果是，则转换为单通道）
    
    返回:
        批量图像张量
    """
    images = []
    successful_paths = []
    
    print(f"正在加载{len(image_paths)}张图像...")
    
    for path in image_paths:
        try:
            if not os.path.exists(path):
                print(f"警告: 文件不存在 - {path}")
                continue
                
            img = Image.open(path).convert('RGB')
            original_size = img.size
            img = img.resize(target_size, Image.BICUBIC)
            img_np = np.array(img) / 255.0  # 归一化到[0,1]
            
            # 如果是掩码，转换为单通道（取平均值或第一个通道）
            if is_mask:
                img_np = np.mean(img_np, axis=2, keepdims=True)
                img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()  # [1, H, W]
            else:
                img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()  # [3, H, W]
                
            images.append(img_tensor)
            successful_paths.append(path)
            
            print(f"成功加载图像: {path} (原始尺寸: {original_size}, 目标尺寸: {target_size}, 通道数: {img_tensor.shape[0]})")
        except Exception as e:
            print(f"无法加载图像 {path}: {e}")
    
    if not images:
        raise ValueError(f"无法加载任何图像。请检查文件路径是否正确。")
    
    batch = torch.stack(images, dim=0).to(device)  # [B, C, H, W]
    print(f"创建图像批次: {batch.shape}")
    
    return batch


def main():
    parser = argparse.ArgumentParser(description='Visualize composition process')
    parser.add_argument('--warp1', type=str, nargs='+', required=True, help='Path to warp1 images')
    parser.add_argument('--warp2', type=str, nargs='+', required=True, help='Path to warp2 images')
    parser.add_argument('--mask1', type=str, nargs='+', default=None, help='Path to mask1 images')
    parser.add_argument('--mask2', type=str, nargs='+', default=None, help='Path to mask2 images')
    parser.add_argument('--output_dir', type=str, default='draw/output/composition', help='Output directory')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--mode', choices=['full', 'forward', 'sample'], default='full', 
                      help='Visualization mode: full process, forward pass only, or sampling only')
    parser.add_argument('--save_tables', action='store_true', help='Save additional tabular data for analysis')
    parser.add_argument('--vis_steps', type=int, default=10, help='Number of visualization steps for sampling process')
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256], 
                       help='Target image size (width, height)')
    parser.add_argument('--guidance_scale', type=float, default=1.0, 
                      help='Guidance scale for conditional sampling (1.0 means no extra guidance)')
    parser.add_argument('--custom_masks', action='store_true', 
                      help='Indicates that the provided masks are custom and should be treated as single-channel')
    parser.add_argument('--force_cpu', action='store_true',
                     help='Force using CPU even if CUDA is available (helpful for memory issues)')
    parser.add_argument('--low_memory', action='store_true',
                     help='Enable aggressive memory optimization for low memory environments')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建中间结果子目录
    os.makedirs(os.path.join(args.output_dir, 'sampling_steps'), exist_ok=True)
    
    # 创建表格输出目录
    tables_dir = os.path.join(args.output_dir, 'tables')
    if args.save_tables:
        os.makedirs(tables_dir, exist_ok=True)
        
    # 记录当前配置
    if args.save_tables:
        with open(os.path.join(tables_dir, 'config.txt'), 'w') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
    
    # 检查模型可用性
    if not COMPOSITION_MODEL_AVAILABLE:
        print("错误: Composition模型不可用。请检查导入路径和依赖。")
        return
    
    # 决定使用的设备
    if args.force_cpu:
        device = torch.device('cpu')
        print("强制使用CPU模式")
    else:
        device = torch.device(args.device)
    
    # 低内存模式优化
    if args.low_memory:
        print("启用低内存模式优化")
        torch.cuda.empty_cache()  # 清理缓存
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        # 减少采样步骤
        if args.vis_steps > 5:
            args.vis_steps = 5
            print(f"在低内存模式下，可视化步骤数已减少至{args.vis_steps}")
    
    try:
        # 在低内存模式下，减小批量大小和图像尺寸
        if args.low_memory and args.target_size[0] > 128:
            print(f"在低内存模式下，将目标尺寸从{args.target_size}降低到(128, 128)")
            args.target_size = [128, 128]
            
        model = ImprovedDiffusionComposition(image_channels=3, device=device)
        print(f"成功创建ImprovedDiffusionComposition模型，使用设备: {device}")
        
        # 如果提供了模型路径，则加载权重
        if args.model_path and os.path.exists(args.model_path):
            print(f"加载模型权重: {args.model_path}")
            try:
                checkpoint = torch.load(args.model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
                print("模型权重加载成功")
            except Exception as e:
                print(f"加载模型权重时出错: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"创建模型时出错: {e}")
        traceback.print_exc()
        return
    
    # 创建可视化包装器
    visualizer = CompositionVisualizer(model, output_dir=args.output_dir, target_size=tuple(args.target_size))
    
    try:
        # 清理缓存
        if torch.cuda.is_available() and not args.force_cpu:
            torch.cuda.empty_cache()
            
        # 加载图像
        print(f"加载warp1图像: {args.warp1}")
        warp1_batch = load_images(args.warp1, device, target_size=tuple(args.target_size))
        
        print(f"加载warp2图像: {args.warp2}")
        warp2_batch = load_images(args.warp2, device, target_size=tuple(args.target_size))
        
        # 清理缓存
        if torch.cuda.is_available() and not args.force_cpu:
            torch.cuda.empty_cache()
            
        # 加载掩码（如果提供）
        mask1_batch = None
        mask2_batch = None
        if args.mask1:
            print(f"加载mask1图像: {args.mask1}")
            # 根据custom_masks参数决定是否将掩码作为单通道处理
            mask1_batch = load_images(args.mask1, device, target_size=tuple(args.target_size), 
                                    is_mask=args.custom_masks)
        if args.mask2:
            print(f"加载mask2图像: {args.mask2}")
            # 根据custom_masks参数决定是否将掩码作为单通道处理
            mask2_batch = load_images(args.mask2, device, target_size=tuple(args.target_size), 
                                    is_mask=args.custom_masks)
        
        # 低内存模式下更积极地清理缓存
        if args.low_memory and torch.cuda.is_available() and not args.force_cpu:
            torch.cuda.empty_cache()
            
        # 保存图像基本信息
        if args.save_tables:
            image_info = {
                'batch_size': warp1_batch.shape[0],
                'channels': warp1_batch.shape[1],
                'height': warp1_batch.shape[2],
                'width': warp1_batch.shape[3],
                'device': str(device),
                'has_mask1': mask1_batch is not None,
                'has_mask2': mask2_batch is not None
            }
            save_metrics_to_csv({'key': list(image_info.keys()), 'value': list(image_info.values())}, 
                               'image_info.csv', tables_dir)
        
        # 执行可视化
        if args.mode == 'full':
            if args.low_memory:
                print("低内存模式下跳过完整可视化，仅执行前向传播可视化...")
                with torch.no_grad():
                    try:
                        # 使用CPU执行关键计算
                        temp_model = model
                        if not args.force_cpu and device.type == 'cuda':
                            temp_model = model.cpu()
                            warp1_cpu = warp1_batch.cpu()
                            warp2_cpu = warp2_batch.cpu()
                            mask1_cpu = mask1_batch.cpu() if mask1_batch is not None else None
                            mask2_cpu = mask2_batch.cpu() if mask2_batch is not None else None
                            out, denoised = temp_model(warp1_cpu, warp2_cpu, mask1_cpu, mask2_cpu)
                            out = out.to(device)
                            denoised = denoised.to(device)
                        else:
                            out, denoised = temp_model(warp1_batch, warp2_batch, mask1_batch, mask2_batch)
                        
                        visualizer.visualizer.visualize_masks(out, 'forward_learned_mask')
                        visualizer.visualizer.visualize_tensor(denoised, 'forward_denoised')
                        
                        # 清理缓存
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    except Exception as e:
                        print(f"前向传播时出错: {e}")
                        traceback.print_exc()
            else:
                print("执行完整可视化过程...")
                out_dict = visualizer.visualize_full_composition(warp1_batch, warp2_batch, mask1_batch, mask2_batch)
                
                # 保存模型输出的结构信息
                if args.save_tables and out_dict:
                    out_info = {
                        'key': list(out_dict.keys()),
                        'type': [type(v).__name__ for v in out_dict.values()],
                        'shape': [str(v.shape) if isinstance(v, torch.Tensor) else 'N/A' for v in out_dict.values()]
                    }
                    save_metrics_to_csv(out_info, 'output_structure.csv', tables_dir)
                
        elif args.mode == 'forward':
            print("仅执行前向传播可视化...")
            with torch.no_grad():
                try:
                    # 对于低内存情况，尝试在CPU上运行
                    if args.low_memory and not args.force_cpu and device.type == 'cuda':
                        temp_model = model.cpu()
                        warp1_cpu = warp1_batch.cpu()
                        warp2_cpu = warp2_batch.cpu()
                        mask1_cpu = mask1_batch.cpu() if mask1_batch is not None else None
                        mask2_cpu = mask2_batch.cpu() if mask2_batch is not None else None
                        out, denoised = temp_model(warp1_cpu, warp2_cpu, mask1_cpu, mask2_cpu)
                        out = out.to(device)
                        denoised = denoised.to(device)
                    else:
                        out, denoised = model(warp1_batch, warp2_batch, mask1_batch, mask2_batch)
                    
                    visualizer.visualizer.visualize_masks(out, 'forward_learned_mask')
                    visualizer.visualizer.visualize_tensor(denoised, 'forward_denoised')
                    
                    # 保存前向传播结果
                    if args.save_tables:
                        forward_info = {
                            'mask_min': out.min().item(),
                            'mask_max': out.max().item(),
                            'mask_mean': out.mean().item(),
                            'denoised_min': denoised.min().item(),
                            'denoised_max': denoised.max().item(),
                            'denoised_mean': denoised.mean().item()
                        }
                        save_metrics_to_csv({'metric': list(forward_info.keys()), 'value': list(forward_info.values())}, 
                                           'forward_stats.csv', tables_dir)
                except Exception as e:
                    print(f"前向传播时出错: {e}")
                    traceback.print_exc()
        elif args.mode == 'sample':
            print(f"执行采样过程可视化，步骤数: {args.vis_steps}...")
            # 低内存模式下更谨慎地进行采样
            if args.low_memory:
                with torch.no_grad():
                    try:
                        # 生成初始随机噪声
                        noise = torch.randn_like(warp1_batch)
                        visualizer.visualizer.visualize_tensor(noise, 'initial_noise')
                        
                        # 仅保存初始和最终结果
                        out, _ = model(warp1_batch, warp2_batch, mask1_batch, mask2_batch)
                        visualizer.visualizer.visualize_masks(out, 'learned_mask')
                        visualizer.visualizer.visualize_tensor(warp1_batch, 'source_image')
                        visualizer.visualizer.visualize_tensor(warp2_batch, 'target_image')
                        
                        # 创建简单合成图像
                        stitched = warp1_batch * out + warp2_batch * (1 - out)
                        visualizer.visualizer.visualize_tensor(stitched, 'stitched_image')
                        
                        print("低内存模式下完成简化采样可视化")
                    except Exception as e:
                        print(f"采样过程中出错: {e}")
                        traceback.print_exc()
            else:
                visualizer.visualize_sample_process(
                    warp1_batch, warp2_batch, mask1_batch, mask2_batch, num_steps=args.vis_steps
                )
        
        print(f"成功可视化composition过程。输出已保存到: {args.output_dir}")
        if args.save_tables:
            print(f"附加表格数据已保存到: {tables_dir}")
            
    except Exception as e:
        print(f"可视化过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        # 清理钩子和缓存
        visualizer.remove_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("已移除所有钩子并清理缓存")


if __name__ == '__main__':
    main() 