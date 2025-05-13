#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
import glob
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import cv2
import time
import math

# 添加项目根目录到路径
sys.path.append('.')

def tensor_to_image(tensor):
    """
    将PyTorch张量转换为NumPy图像
    
    参数:
        tensor: PyTorch张量 [C,H,W] 或 [B,C,H,W]
        
    返回:
        numpy图像 [H,W,C], 值范围为0-255的uint8
    """
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
    
    # 第一部分: warp2 * mask - a保持原始分辨率
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

def main():
    parser = argparse.ArgumentParser(description='增强版Composition测试脚本 - 最终版')
    parser.add_argument('--model_path', type=str, default='Composition/model/best_model.pth', 
                        help='模型路径')
    parser.add_argument('--test_data', type=str, default='data/UDIS-D/composition_data/test',
                        help='测试数据路径')
    parser.add_argument('--output_dir', type=str, default='test_composition_results',
                        help='输出结果目录')
    parser.add_argument('--limit', type=int, default=10,
                        help='测试图像数量限制，设为-1表示处理全部图像')
    parser.add_argument('--gpu', type=int, default=0,
                        help='指定使用的GPU ID')
    parser.add_argument('--sample_steps', type=int, default=100, help='扩散采样步数')
    parser.add_argument('--image_size', type=int, default=512,
                        help='输入图像的处理分辨率')
    parser.add_argument('--threshold', type=int, default=127,
                        help='Mask二值化阈值')
    parser.add_argument('--interactive', action='store_true',
                        help='启用交互模式，允许调整阈值')
    
    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available():
        if args.gpu >= 0 and args.gpu < torch.cuda.device_count():
            device = torch.device(f'cuda:{args.gpu}')
            print(f"使用GPU: {torch.cuda.get_device_name(args.gpu)}")
        else:
            print(f"指定的GPU ID {args.gpu} 无效，使用默认GPU")
            device = torch.device('cuda')
    else:
        print("CUDA不可用，使用CPU")
        device = torch.device('cpu')
    
    # 创建输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子目录
    result_dir = os.path.join(output_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    
    # 保存测试参数
    with open(os.path.join(output_dir, "test_config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    try:
        from Composition.Codes.enhanced_network import ImprovedDiffusionComposition
        
        # 创建模型实例
        model = ImprovedDiffusionComposition(
            num_timesteps=1000,
            beta_schedule='linear',
            image_size=args.image_size,
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
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                # 检查点是训练期间保存的整个状态
                model.load_state_dict(checkpoint['model'])
            else:
                # 尝试其他可能的键
                possible_keys = ['state_dict', 'net', 'network', 'model_weights']
                for key in possible_keys:
                    if key in checkpoint:
                        model.load_state_dict(checkpoint[key])
                        break
                else:
                    # 如果没有找到任何已知键，尝试直接加载
                    try:
                        model.load_state_dict(checkpoint)
                    except Exception as load_err:
                        print(f"无法加载模型权重: {load_err}")
                        raise
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print(f"模型加载成功，设置为评估模式")
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 查找测试图像
    warp1_dir = os.path.join(args.test_data, "warp1")
    warp2_dir = os.path.join(args.test_data, "warp2")
    mask1_dir = os.path.join(args.test_data, "mask1")
    mask2_dir = os.path.join(args.test_data, "mask2")
    
    warp1_files = sorted(glob.glob(os.path.join(warp1_dir, "*.*")))
    warp2_files = sorted(glob.glob(os.path.join(warp2_dir, "*.*")))
    mask1_files = sorted(glob.glob(os.path.join(mask1_dir, "*.*")))
    mask2_files = sorted(glob.glob(os.path.join(mask2_dir, "*.*")))
    
    # 确保文件列表非空且长度一致
    if not warp1_files or not warp2_files or not mask1_files or not mask2_files:
        print(f"无法找到测试图像，请检查路径: {args.test_data}")
        return
    
    if len(warp1_files) != len(warp2_files) or len(warp1_files) != len(mask1_files) or len(warp1_files) != len(mask2_files):
        print(f"警告: 文件数量不一致 - warp1: {len(warp1_files)}, warp2: {len(warp2_files)}, mask1: {len(mask1_files)}, mask2: {len(mask2_files)}")
        min_len = min(len(warp1_files), len(warp2_files), len(mask1_files), len(mask2_files))
        warp1_files = warp1_files[:min_len]
        warp2_files = warp2_files[:min_len]
        mask1_files = mask1_files[:min_len]
        mask2_files = mask2_files[:min_len]
    
    # 限制测试图像数量
    if args.limit > 0 and args.limit < len(warp1_files):
        warp1_files = warp1_files[:args.limit]
        warp2_files = warp2_files[:args.limit]
        mask1_files = mask1_files[:args.limit]
        mask2_files = mask2_files[:args.limit]
    
    # 预处理变换
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    mask_preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    
    print(f"开始处理 {len(warp1_files)} 对图像...")
    
    # 记录处理时间和评估指标
    total_time = 0
    success_count = 0
    results = []
    
    # 如果是交互模式，显示使用说明
    if args.interactive:
        print("使用说明:")
        print("  - 按回车键 (Enter): 处理下一张照片")
        print("  - 按 '+' 键: 增加阈值")
        print("  - 按 '-' 键: 减小阈值")
        print("  - 按 's' 键: 保存当前图像")
        print("  - 按 ESC 键: 退出程序")
    
    # 默认阈值
    threshold = args.threshold
    
    # 处理每对图像
    for i, (warp1_path, warp2_path, mask1_path, mask2_path) in enumerate(tqdm(zip(warp1_files, warp2_files, mask1_files, mask2_files), total=len(warp1_files))):
        try:
            file_basename = os.path.splitext(os.path.basename(warp1_path))[0]
            print(f"\n处理图像对 {i+1}/{len(warp1_files)}: {file_basename}")
            
            # 加载图像
            warp1 = Image.open(warp1_path).convert('RGB')
            warp2 = Image.open(warp2_path).convert('RGB')
            mask1 = Image.open(mask1_path).convert('L')
            mask2 = Image.open(mask2_path).convert('L')
            
            # 记录原始尺寸
            original_size1 = warp1.size
            original_size2 = warp2.size
            original_size_mask1 = mask1.size
            print(f"  原始图像尺寸: warp1={original_size1}, warp2={original_size2}")
            
            # 创建模型输入 - 调整到模型需要的输入尺寸
            warp1_resized = warp1.resize((args.image_size, args.image_size), Image.LANCZOS)
            warp2_resized = warp2.resize((args.image_size, args.image_size), Image.LANCZOS)
            mask1_resized = mask1.resize((args.image_size, args.image_size), Image.LANCZOS)
            mask2_resized = mask2.resize((args.image_size, args.image_size), Image.LANCZOS)
            
            # 转换为张量
            warp1_tensor = preprocess(warp1_resized).unsqueeze(0).to(device)
            warp2_tensor = preprocess(warp2_resized).unsqueeze(0).to(device)
            mask1_tensor = mask_preprocess(mask1_resized).unsqueeze(0).to(device)
            mask2_tensor = mask_preprocess(mask2_resized).unsqueeze(0).to(device)
            
            # 使用模型进行处理
            start_time = time.time()
            print(f"使用 {args.sample_steps} 步扩散采样生成mask")
            
            # 使用模型处理图像对
            output_mask, output_result = model.safe_forward(
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
            warp1_img = cv2.cvtColor(np.array(warp1), cv2.COLOR_RGB2BGR)
            warp2_img = cv2.cvtColor(np.array(warp2), cv2.COLOR_RGB2BGR)
            mask1_img = np.array(mask1)
            
            # 保存网络生成的mask和结果图像
            mask_save_path = os.path.join(result_dir, f"{file_basename}_mask.png")
            cv2.imwrite(mask_save_path, mask_img_gray)
            
            # 处理当前图像
            if args.interactive:
                current_threshold = threshold
                while True:
                    # 使用当前阈值处理图像
                    warp1_display, warp2_display, merged_img, mask_binary = process_mask_for_composition(
                        mask_img_gray, warp1_img, warp2_img, mask1_img, current_threshold
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
                        return  # 完全退出程序
                    elif key == 13 or key == 10:  # 回车键
                        # 保存当前结果，然后继续到下一张图片
                        output_path = os.path.join(merged_dir, f"{file_basename}_merged_thresh{current_threshold}.png")
                        cv2.imwrite(output_path, merged_img)
                        
                        mask_output_path = os.path.join(result_dir, f"{file_basename}_mask_binary.png")
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
                        output_path = os.path.join(merged_dir, f"{file_basename}_merged_thresh{current_threshold}.png")
                        cv2.imwrite(output_path, merged_img)
                        
                        mask_output_path = os.path.join(result_dir, f"{file_basename}_mask_binary.png")
                        cv2.imwrite(mask_output_path, mask_binary)
                        
                        print(f"  已保存合并图像到: {output_path}")
                        print(f"  已保存二值化mask到: {mask_output_path}")
            else:
                # 非交互模式 - 直接处理并保存
                warp1_display, warp2_display, merged_img, mask_binary = process_mask_for_composition(
                    mask_img_gray, warp1_img, warp2_img, mask1_img, threshold
                )
                
                # 保存结果
                merged_output_path = os.path.join(merged_dir, f"{file_basename}_merged.png")
                cv2.imwrite(merged_output_path, merged_img)
                
                mask_binary_path = os.path.join(result_dir, f"{file_basename}_mask_binary.png")
                cv2.imwrite(mask_binary_path, mask_binary)
                
                print(f"  保存合并结果: {merged_output_path}")
                
                # 显示图像
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
                
                # 显示1秒后继续
                cv2.waitKey(1000)
            
            # 更新处理信息
            success_count += 1
            results.append({
                'filename': file_basename,
                'process_time': process_time,
                'threshold': threshold,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"处理图像对 {file_basename} 时出错: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'filename': file_basename,
                'status': 'error',
                'error': str(e)
            })
    
    cv2.destroyAllWindows()
    
    # 打印统计信息
    avg_time = total_time / max(success_count, 1)
    print(f"\n处理完成!")
    print(f"成功处理: {success_count}/{len(warp1_files)} 图像对")
    print(f"平均处理时间: {avg_time:.2f}秒")
    print(f"结果保存在: {output_dir}")
    
    # 保存处理报告
    with open(os.path.join(output_dir, "processing_report.json"), "w") as f:
        json.dump({
            'total_images': len(warp1_files),
            'success_count': success_count,
            'average_time': avg_time,
            'results': results
        }, f, indent=4)

if __name__ == "__main__":
    main() 