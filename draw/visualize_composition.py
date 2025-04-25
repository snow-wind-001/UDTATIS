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
    
    def __init__(self, model, output_dir='draw/output/composition'):
        """
        初始化composition可视化包装器
        
        参数:
            model: 原始的ImprovedDiffusionComposition模型
            output_dir: 输出目录
        """
        super(CompositionVisualizer, self).__init__()
        self.model = model
        self.visualizer = FeatureVisualizer(save_dir=output_dir)
        
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
        可视化采样过程中的中间结果
        
        参数:
            warp1: 第一个warp图像
            warp2: 第二个warp图像
            mask1: 第一个掩码（可选）
            mask2: 第二个掩码（可选）
            num_steps: 可视化的采样步骤数
        
        返回:
            采样结果和中间步骤的图像
        """
        # 创建输出目录
        sample_dir = os.path.join(self.visualizer.save_dir, 'sampling_steps')
        os.makedirs(sample_dir, exist_ok=True)
        
        # 创建指标目录
        metrics_dir = os.path.join(self.visualizer.save_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存一个自定义的sample方法，记录中间步骤
        original_sample = self.model.sample
        
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
            """自定义采样函数，记录中间结果"""
            # 移动到相应设备
            device = self.model.device
            x = x.to(device)
            y = y.to(device)
            if mask1 is not None:
                mask1 = mask1.to(device)
            if mask2 is not None:
                mask2 = mask2.to(device)
            
            batch_size = x.shape[0]
            
            # 使用 timesteps 参数（如果提供）或默认 num_steps
            steps_count = timesteps if timesteps is not None else num_steps
            
            # 创建时间步
            if use_ddim:
                # DDIM采样使用均匀分布的时间步
                steps = torch.linspace(0, self.model.num_timesteps - 1, steps_count, dtype=torch.long, device=device)
                alpha_cumprod = self.model.alphas_cumprod
            else:
                # DDPM采样
                steps = torch.linspace(0, self.model.num_timesteps - 1, steps_count, dtype=torch.long, device=device)
            
            # 计算要记录的步骤
            vis_indices = np.linspace(0, len(steps) - 1, num_steps, dtype=int)
            
            # 在每次迭代后清理未使用的变量
            def cleanup():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 初始化随机噪声作为起点
            noise = torch.randn_like(x, device=device)
            denoised = noise
            
            # 记录初始噪声
            intermediate_results.append(denoised.clone().cpu())  # 保存到CPU内存
            selected_timesteps.append(steps[0].item())
            
            # 记录初始指标
            metrics['timestep'].append(steps[0].item())
            metrics['noise_level'].append(1.0)  # 初始噪声级别为100%
            metrics['psnr'].append(0.0)  # 初始PSNR为0
            metrics['noise_pred_error'].append(0.0)
            metrics['mask_entropy'].append(0.0)
            
            # 上一步预测的噪声（用于计算预测误差）
            last_predicted_noise = None
            
            # 逆向扩散过程
            for i in reversed(range(0, len(steps))):
                t = torch.full((batch_size,), steps[i], device=device, dtype=torch.long)
                
                # 获取当前时间步参数
                alpha_t = self.model._extract(self.model.alphas, t, x.shape)
                alpha_cumprod_t = self.model._extract(self.model.alphas_cumprod, t, x.shape)
                
                # 预测噪声 - 添加分类引导
                # 使用通道适配器处理输入
                try:
                    # 在计算前清理内存
                    cleanup()
                    
                    concat_input = torch.cat([denoised, y], dim=1)
                    adapted_input = self.model.channel_adapter(concat_input)
                    predicted_noise = self.model.diffusion(adapted_input, t)
                    
                    if guidance_scale > 1.0:
                        # 计算无条件预测
                        # 使用通道适配器处理无条件输入
                        unconditional_input = torch.cat([denoised, torch.zeros_like(y, device=device)], dim=1)
                        adapted_unconditional = self.model.channel_adapter(unconditional_input)
                        unconditional_noise = self.model.diffusion(adapted_unconditional, t)
                        
                        # 应用分类引导
                        predicted_noise = unconditional_noise + guidance_scale * (predicted_noise - unconditional_noise)
                        
                        # 清理无条件预测变量
                        del unconditional_input, adapted_unconditional, unconditional_noise
                        cleanup()
                        
                except Exception as e:
                    print(f"噪声预测过程中出错: {e}")
                    traceback.print_exc()
                    # 如果出错，使用随机噪声代替
                    predicted_noise = torch.randn_like(denoised, device=device)
                
                # 确保预测噪声和去噪图像具有相同的尺寸
                if predicted_noise.shape != denoised.shape:
                    predicted_noise = torch.nn.functional.interpolate(
                        predicted_noise, 
                        size=denoised.shape[2:], 
                        mode='bilinear', 
                        align_corners=True
                    )
                
                # 计算去噪步骤
                try:
                    # 再次清理内存
                    cleanup()
                    
                    if use_ddim and i > 0:
                        # DDIM采样
                        alpha_cumprod_s = self.model._extract(self.model.alphas_cumprod, torch.tensor([steps[i-1]], device=device), x.shape)
                        
                        # 预测x_0
                        pred_x0 = (denoised - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
                        
                        # 预测下一步
                        denoised = torch.sqrt(alpha_cumprod_s) * pred_x0 + torch.sqrt(1 - alpha_cumprod_s) * predicted_noise
                        
                        # 清理中间变量
                        del pred_x0, alpha_cumprod_s
                    else:
                        # DDPM采样
                        if i > 0:
                            beta_t = self.model._extract(self.model.betas, t, x.shape)
                            alpha_cumprod_prev_t = self.model._extract(
                                torch.cat([self.model.alphas_cumprod[0:1], self.model.alphas_cumprod[:-1]]), 
                                t, x.shape
                            )
                            
                            # 计算方差
                            variance = beta_t * (1. - alpha_cumprod_prev_t) / (1. - alpha_cumprod_t)
                            noise_factor = torch.randn_like(x, device=device) * torch.sqrt(variance)
                            
                            # 清理临时变量
                            del beta_t, alpha_cumprod_prev_t, variance
                        else:
                            noise_factor = 0.
                        
                        # 更新去噪图像
                        denoised = 1 / torch.sqrt(alpha_t) * (
                            denoised - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
                        ) + noise_factor
                        
                        # 如果有noise_factor变量，清理它
                        if 'noise_factor' in locals() and noise_factor != 0:
                            del noise_factor
                    
                    # 清理不再需要的变量
                    del alpha_t, alpha_cumprod_t
                    cleanup()
                    
                except Exception as e:
                    print(f"去噪步骤计算出错: {e}")
                    traceback.print_exc()
                    # 如果出错，保持前一步的结果
                    if i > 0 and len(intermediate_results) > 1:
                        denoised = intermediate_results[-1].clone().to(device)
                
                # 如果当前步骤是要记录的步骤，则保存中间结果和指标
                if i in vis_indices:
                    # 存储到CPU减少GPU内存使用
                    intermediate_results.append(denoised.clone().cpu())
                    selected_timesteps.append(steps[i].item())
                    
                    # 计算指标
                    noise_level = 1.0 - float(i) / float(len(steps) - 1)  # 噪声水平从1.0降到0.0
                    
                    # 计算PSNR（如果与原始图像比较，这里我们与x比较）
                    try:
                        with torch.no_grad():
                            mse = torch.mean((denoised.cpu() - x.cpu()) ** 2).item()
                            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
                    except Exception as e:
                        print(f"计算PSNR时出错: {e}")
                        psnr = 0.0
                    
                    # 计算噪声预测误差
                    noise_pred_error = 0.0
                    if last_predicted_noise is not None:
                        try:
                            with torch.no_grad():
                                noise_pred_error = torch.mean(torch.abs(
                                    predicted_noise.cpu() - last_predicted_noise.cpu())).item()
                        except Exception as e:
                            print(f"计算噪声预测误差时出错: {e}")
                    
                    # 计算掩码熵（如果有掩码生成器）
                    mask_entropy = 0.0
                    try:
                        if hasattr(self.model, 'mask_generator'):
                            with torch.no_grad():
                                concat = torch.cat([denoised.cpu(), y.cpu()], dim=1)
                                # 将计算移到CPU上
                                mask_gen = self.model.mask_generator.cpu()
                                mask = mask_gen(concat)
                                # 恢复到GPU
                                if torch.cuda.is_available():
                                    self.model.mask_generator.cuda()
                                
                                # 熵 = -sum(p*log(p))
                                eps = 1e-10  # 避免log(0)
                                mask_prob = torch.clamp(mask, eps, 1.0 - eps)
                                entropy = -mask_prob * torch.log(mask_prob) - (1.0 - mask_prob) * torch.log(1.0 - mask_prob)
                                mask_entropy = torch.mean(entropy).item()
                    except Exception as e:
                        print(f"掩码熵计算出错: {e}")
                    
                    # 保存指标
                    metrics['timestep'].append(steps[i].item())
                    metrics['noise_level'].append(noise_level)
                    metrics['psnr'].append(psnr)
                    metrics['noise_pred_error'].append(noise_pred_error)
                    metrics['mask_entropy'].append(mask_entropy)
                    
                    # 更新上一步预测的噪声
                    last_predicted_noise = predicted_noise.clone().cpu()
                    
                    # 清理内存
                    cleanup()
                    
                # 如果不需要保存的步骤，释放之前的预测噪声
                elif last_predicted_noise is not None:
                    del last_predicted_noise
                    last_predicted_noise = predicted_noise.clone().cpu()
                
                # 清理当前步骤的噪声预测
                del predicted_noise
                cleanup()
            
            # 计算学习到的混合掩码
            try:
                concatenated = torch.cat([x, y], dim=1)
                mask = self.model.mask_generator(concatenated)
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
            except Exception as e:
                print(f"掩码计算或图像合成出错: {e}")
                traceback.print_exc()
                # 如果出错，创建简单的混合
                learned_mask1 = torch.ones_like(x[:, 0:1, :, :]) * 0.5
                stitched_image = (x + y) / 2.0
            
            # 最终清理
            cleanup()
            
            # 返回学习到的掩码和去噪后的图像
            return learned_mask1, denoised, stitched_image
        
        # 替换采样方法
        self.model.sample = custom_sample
        
        # 执行采样过程
        try:
            # 记录是否处于低内存模式
            low_memory = getattr(self.model, 'low_memory', False)
            if low_memory:
                setattr(self.model, 'low_memory', True)
                
            learned_mask1, denoised, stitched_image = self.model.sample(
                warp1, warp2, mask1, mask2, num_steps=100, use_ddim=True
            )
            
            # 可视化中间步骤
            for idx, (result, timestep) in enumerate(zip(intermediate_results, selected_timesteps)):
                # 将结果放回设备用于可视化
                result_device = result.to(warp1.device)
                self.visualizer.visualize_tensor(
                    result_device, f'sampling_step_{idx}_t{timestep}', 
                    save_dir=sample_dir
                )
                # 用完立即删除，减少内存使用
                del result_device
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 可视化最终结果
            self.visualizer.visualize_tensor(denoised, 'final_denoised')
            self.visualizer.visualize_masks(learned_mask1, 'final_learned_mask')
            self.visualizer.visualize_tensor(stitched_image, 'final_stitched_image')
            
            # 可视化原始输入与最终输出的比较
            self.visualizer.side_by_side_comparison(
                warp1, stitched_image, 'input_vs_output',
                titles=['Input Image 1', 'Stitched Output']
            )
            
            # 可视化中间特征
            for name, activation in self.activations.items():
                if isinstance(activation, torch.Tensor):
                    self.visualizer.visualize_feature_maps(activation, name)
                    # 用完立即删除，减少内存使用
                    del self.activations[name]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 保存度量指标到CSV
            csv_path = save_metrics_to_csv(metrics, 'sampling_metrics.csv', metrics_dir)
            
            # 创建并保存度量图表
            plot_and_save_metrics(
                {'PSNR': metrics['psnr'], 'Noise Level': metrics['noise_level'], 
                 'Mask Entropy': metrics['mask_entropy']}, 
                'Sampling Process Metrics', 
                'sampling_metrics.png', 
                metrics_dir
            )
            
            print(f"成功可视化了 {len(intermediate_results)} 个采样步骤")
            print(f"指标数据已保存到: {csv_path}")
            
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
        
        return learned_mask1, denoised, stitched_image, intermediate_results
    
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
        
        # 首先可视化输入
        self.visualizer.visualize_tensor(warp1, 'input_warp1')
        self.visualizer.visualize_tensor(warp2, 'input_warp2')
        
        # 处理并可视化掩码图像
        if mask1 is not None:
            # 如果掩码是3通道图像，取第一个通道
            if mask1.shape[1] == 3:
                mask1_single = mask1[:, 0:1, :, :]
                print(f"将mask1从{mask1.shape}转换为单通道{mask1_single.shape}")
                self.visualizer.visualize_masks(mask1_single, 'input_mask1')
            else:
                self.visualizer.visualize_masks(mask1, 'input_mask1')
                
        if mask2 is not None:
            # 如果掩码是3通道图像，取第一个通道
            if mask2.shape[1] == 3:
                mask2_single = mask2[:, 0:1, :, :]
                print(f"将mask2从{mask2.shape}转换为单通道{mask2_single.shape}")
                self.visualizer.visualize_masks(mask2_single, 'input_mask2')
            else:
                self.visualizer.visualize_masks(mask2, 'input_mask2')
        
        # 创建中间结果目录
        intermediate_dir = os.path.join(self.visualizer.save_dir, 'intermediate')
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # 1. 使用原始forward方法可视化一次前向传播
        with torch.no_grad():
            out, denoised = self.model(warp1, warp2, mask1, mask2)
        
        # 可视化前向传播结果
        self.visualizer.visualize_masks(out, 'forward_learned_mask')
        self.visualizer.visualize_tensor(denoised, 'forward_denoised')
        
        # 可视化前向传播捕获的激活
        for name, activation in self.activations.items():
            if isinstance(activation, torch.Tensor):
                self.visualizer.visualize_feature_maps(activation, f'forward_{name}')
        
        # 2. 可视化完整的采样过程(慢但更详细)
        learned_mask1, sample_denoised, stitched_image, _ = self.visualize_sample_process(
            warp1, warp2, mask1, mask2, num_steps=10
        )
        
        # 3. 使用build_model函数可视化最终结果
        out_dict = build_model(self.model, warp1, warp2, mask1, mask2)
        
        # 可视化build_model的结果
        self.visualizer.visualize_masks(out_dict['learned_mask1'], 'build_model_learned_mask1')
        self.visualizer.visualize_masks(out_dict['learned_mask2'], 'build_model_learned_mask2')
        self.visualizer.visualize_tensor(out_dict['stitched_image'], 'build_model_stitched_image')
        self.visualizer.visualize_tensor(out_dict['denoised'], 'build_model_denoised')
        
        # 最终合成图像与原始输入的比较
        self.visualizer.side_by_side_comparison(
            warp1, out_dict['stitched_image'], 'warp1_vs_stitched',
            titles=['Original Warp1', 'Stitched Image']
        )
        
        self.visualizer.side_by_side_comparison(
            warp2, out_dict['stitched_image'], 'warp2_vs_stitched',
            titles=['Original Warp2', 'Stitched Image']
        )
        
        return out_dict
    
    def remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


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
    visualizer = CompositionVisualizer(model, output_dir=args.output_dir)
    
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