#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
import glob
from PIL import Image
import torchvision.transforms as transforms
import argparse
import json
import cv2
import time
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append('.')

# 从main.py中导入处理函数
from main import process_mask_for_composition, load_config

def tensor_to_image(tensor):
    """将PyTorch张量转换为NumPy图像"""
    if tensor.dim() == 4:  # [B,C,H,W]
        tensor = tensor[0]  # 只取第一个样本
    
    # 移到CPU
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

def main():
    parser = argparse.ArgumentParser(description='Composition测试脚本 - 调试版')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--limit', type=int, default=3, help='测试图像数量限制')
    parser.add_argument('--model_path', type=str, help='指定模型路径，不指定则使用最新模型')
    parser.add_argument('--threshold', type=int, default=127, help='二值化阈值(0-255)')
    parser.add_argument('--interactive', action='store_true', help='启用交互模式')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 获取数据和结果路径
    data_path = config['composition']['test']['test_path']
    result_path = config['composition']['test']['result_path']
    
    # 确保结果目录存在
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
    
    # 打印目录信息
    print(f"数据路径: {data_path}")
    print(f"结果路径: {result_path}")
    print(f"子目录:")
    for dir_name, path in save_dirs.items():
        print(f"  - {dir_name}: {path}")
    print(f"合并图像目录: {merged_dir}")
    
    # 验证数据目录
    required_dirs = ['warp1', 'warp2', 'mask1', 'mask2']
    for dir_name in required_dirs:
        dir_path = os.path.join(data_path, dir_name)
        if not os.path.exists(dir_path):
            print(f"错误: 找不到目录 {dir_path}")
            return
    
    # 获取文件列表
    warp1_files = sorted(glob.glob(os.path.join(data_path, 'warp1', '*.*')))
    warp2_files = sorted(glob.glob(os.path.join(data_path, 'warp2', '*.*')))
    mask1_files = sorted(glob.glob(os.path.join(data_path, 'mask1', '*.*')))
    mask2_files = sorted(glob.glob(os.path.join(data_path, 'mask2', '*.*')))
    
    print(f"找到文件:")
    print(f"  - warp1: {len(warp1_files)} 文件")
    print(f"  - warp2: {len(warp2_files)} 文件")
    print(f"  - mask1: {len(mask1_files)} 文件")
    print(f"  - mask2: {len(mask2_files)} 文件")
    
    # 确定处理的文件数量
    max_files = min(len(warp1_files), len(warp2_files), len(mask1_files), len(mask2_files))
    if args.limit > 0 and args.limit < max_files:
        max_files = args.limit
    
    print(f"将处理 {max_files} 个图像对")
    
    # 设置模型路径
    model_path = args.model_path
    if not model_path:
        # 尝试寻找最新的模型文件
        model_dir = config['composition']['train']['model_save_path']
        if os.path.exists(model_dir):
            checkpoint_files = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
            if checkpoint_files:
                model_path = checkpoint_files[-1]  # 使用最新的checkpoint
                print(f"使用最新模型: {model_path}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    if model_path and os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        try:
            from Composition.Codes.enhanced_network import ImprovedDiffusionComposition
            
            # 创建模型实例
            model = ImprovedDiffusionComposition(
                num_timesteps=1000,
                beta_schedule='linear',
                image_size=512,
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
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            print(f"模型加载成功，设置为评估模式")
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print(f"没有找到有效的模型，将使用mask2作为输入mask进行测试")
        model = None
    
    # 定义预处理变换
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 定义掩码预处理变换
    mask_preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # 初始化统计信息
    total_time = 0
    success_count = 0
    results = []
    
    # 处理图像
    for idx in range(max_files):
        try:
            # 获取文件名
            warp1_file = warp1_files[idx]
            warp2_file = warp2_files[idx]
            mask1_file = mask1_files[idx]
            mask2_file = mask2_files[idx]
            
            file_name = os.path.basename(warp1_file)
            name_without_ext = os.path.splitext(file_name)[0]
            
            print(f"\n处理图像对 {idx+1}/{max_files}: {name_without_ext}")
            print(f"  文件:")
            print(f"    - warp1: {warp1_file}")
            print(f"    - warp2: {warp2_file}")
            print(f"    - mask1: {mask1_file}")
            print(f"    - mask2: {mask2_file}")
            
            # 读取图像
            warp1_img = Image.open(warp1_file).convert('RGB')
            warp2_img = Image.open(warp2_file).convert('RGB')
            mask1_img = Image.open(mask1_file).convert('L')
            mask2_img = Image.open(mask2_file).convert('L')
            
            print(f"  图像尺寸:")
            print(f"    - warp1: {warp1_img.size}")
            print(f"    - warp2: {warp2_img.size}")
            print(f"    - mask1: {mask1_img.size}")
            print(f"    - mask2: {mask2_img.size}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 如果有模型，使用模型生成mask
            if model:
                # 转换为张量
                warp1_tensor = preprocess(warp1_img.resize((512, 512), Image.LANCZOS)).unsqueeze(0).to(device)
                warp2_tensor = preprocess(warp2_img.resize((512, 512), Image.LANCZOS)).unsqueeze(0).to(device)
                mask1_tensor = mask_preprocess(mask1_img.resize((512, 512), Image.LANCZOS)).unsqueeze(0).to(device)
                mask2_tensor = mask_preprocess(mask2_img.resize((512, 512), Image.LANCZOS)).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # 使用safe_forward方法处理图像对
                    output_mask, output_result = model.safe_forward(
                        warp1=warp1_tensor,
                        warp2=warp2_tensor,
                        mask1=mask1_tensor,
                        mask2=mask2_tensor
                    )
                
                # 转换输出mask为图像
                mask_img = tensor_to_image(output_mask)
                mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
            else:
                # 使用mask2作为mask
                mask_img_gray = np.array(mask2_img)
            
            # 转换原始图像为OpenCV格式以进行后处理
            warp1_img_cv = cv2.cvtColor(np.array(warp1_img), cv2.COLOR_RGB2BGR)
            warp2_img_cv = cv2.cvtColor(np.array(warp2_img), cv2.COLOR_RGB2BGR)
            mask1_img_np = np.array(mask1_img)
            
            # 保存模型生成的mask（或mask2）
            mask_save_path = os.path.join(save_dirs.get('mask', os.path.join(result_path, 'mask')), f"{name_without_ext}_mask.png")
            cv2.imwrite(mask_save_path, mask_img_gray)
            print(f"  保存mask到: {mask_save_path}")
            
            # 处理图像合成
            warp1_display, warp2_display, merged_img, mask_binary = process_mask_for_composition(
                mask_img_gray, warp1_img_cv, warp2_img_cv, mask1_img_np, args.threshold
            )
            
            # 保存结果
            merged_output_path = os.path.join(merged_dir, f"{name_without_ext}_merged.png")
            mask_binary_path = os.path.join(save_dirs.get('mask', os.path.join(result_path, 'mask')), f"{name_without_ext}_mask_binary.png")
            
            cv2.imwrite(merged_output_path, merged_img)
            cv2.imwrite(mask_binary_path, mask_binary)
            
            print(f"  保存合并结果到: {merged_output_path}")
            print(f"  保存二值化mask到: {mask_binary_path}")
            
            # 计算处理时间
            process_time = time.time() - start_time
            total_time += process_time
            
            # 更新统计信息
            success_count += 1
            results.append({
                'filename': name_without_ext,
                'process_time': process_time,
                'status': 'success'
            })
            
            print(f"  处理时间: {process_time:.2f}秒")
            
        except Exception as e:
            print(f"处理图像 {idx+1} 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'filename': name_without_ext if 'name_without_ext' in locals() else f"unknown_{idx}",
                'status': 'error',
                'error': str(e)
            })
    
    # 计算统计信息
    avg_time = total_time / max(success_count, 1)
    print(f"\n处理完成!")
    print(f"处理图像: {success_count}/{max_files}")
    print(f"平均处理时间: {avg_time:.2f}秒")
    
    # 保存处理报告
    report_path = os.path.join(result_path, "processing_report.json")
    with open(report_path, "w") as f:
        json.dump({
            'total_images': max_files,
            'success_count': success_count,
            'average_time': avg_time,
            'results': results
        }, f, indent=4)
    
    print(f"完整报告已保存至: {report_path}")

if __name__ == "__main__":
    main() 