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

# 添加项目根目录到路径
sys.path.append('.')

# 从main.py中导入处理函数
from main import process_mask_for_composition

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

def main():
    parser = argparse.ArgumentParser(description='完整Composition测试脚本')
    parser.add_argument('--model_path', type=str, default='Composition/model/model_epoch_23.pth', 
                        help='模型路径')
    parser.add_argument('--test_data', type=str, default='data/UDIS-D/composition_data/test',
                        help='测试数据路径')
    parser.add_argument('--output_dir', type=str, default='Composition/full_test_results',
                        help='输出结果目录')
    parser.add_argument('--limit', type=int, default=5,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子目录
    input_dir = os.path.join(output_dir, "input")
    mask_dir = os.path.join(output_dir, "mask")
    mask_binary_dir = os.path.join(output_dir, "mask_binary")
    merged_dir = os.path.join(output_dir, "merged")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(mask_binary_dir, exist_ok=True)
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
    
    # 定义预处理变换
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 定义掩码预处理变换
    mask_preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    
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
    
    # 限制处理的文件数量
    max_files = min(len(warp1_files), len(warp2_files), len(mask1_files), len(mask2_files))
    if args.limit > 0 and args.limit < max_files:
        max_files = args.limit
    
    print(f"将处理 {max_files} 个图像对")
    
    # 初始化统计信息
    total_time = 0
    success_count = 0
    results = []
    
    # 处理图像
    with torch.no_grad():  # 禁用梯度计算
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
                
                # 读取图像
                warp1_img = Image.open(warp1_file).convert('RGB')
                warp2_img = Image.open(warp2_file).convert('RGB')
                mask1_img = Image.open(mask1_file).convert('L')
                mask2_img = Image.open(mask2_file).convert('L')
                
                # 转换为张量
                warp1_tensor = preprocess(warp1_img.resize((args.image_size, args.image_size), Image.LANCZOS)).unsqueeze(0).to(device)
                warp2_tensor = preprocess(warp2_img.resize((args.image_size, args.image_size), Image.LANCZOS)).unsqueeze(0).to(device)
                mask1_tensor = mask_preprocess(mask1_img.resize((args.image_size, args.image_size), Image.LANCZOS)).unsqueeze(0).to(device)
                mask2_tensor = mask_preprocess(mask2_img.resize((args.image_size, args.image_size), Image.LANCZOS)).unsqueeze(0).to(device)
                
                # 保存输入图像
                input_imgs = Image.new('RGB', (args.image_size*2, args.image_size))
                input_imgs.paste(warp1_img.resize((args.image_size, args.image_size), Image.LANCZOS), (0, 0))
                input_imgs.paste(warp2_img.resize((args.image_size, args.image_size), Image.LANCZOS), (args.image_size, 0))
                input_imgs.save(os.path.join(input_dir, f"{name_without_ext}_input.png"))
                
                # 使用模型生成mask
                start_time = time.time()
                output_mask, output_result = model.safe_forward(
                    warp1=warp1_tensor,
                    warp2=warp2_tensor,
                    mask1=mask1_tensor,
                    mask2=mask2_tensor
                )
                predict_time = time.time() - start_time
                
                # 转换输出mask为图像
                mask_img = tensor_to_image(output_mask)
                mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
                
                # 保存模型生成的mask
                cv2.imwrite(os.path.join(mask_dir, f"{name_without_ext}_mask.png"), mask_img_gray)
                
                # 转换原始图像为OpenCV格式以进行后处理
                warp1_img_cv = cv2.cvtColor(np.array(warp1_img), cv2.COLOR_RGB2BGR)
                warp2_img_cv = cv2.cvtColor(np.array(warp2_img), cv2.COLOR_RGB2BGR)
                mask1_img_np = np.array(mask1_img)
                
                # 处理图像合成
                process_start_time = time.time()
                _, _, merged_img, mask_binary = process_mask_for_composition(
                    mask_img_gray, warp1_img_cv, warp2_img_cv, mask1_img_np, args.threshold
                )
                process_time = time.time() - process_start_time
                
                # 计算总处理时间
                total_process_time = predict_time + process_time
                total_time += total_process_time
                
                # 保存结果
                merged_output_path = os.path.join(merged_dir, f"{name_without_ext}_merged.png")
                mask_binary_path = os.path.join(mask_binary_dir, f"{name_without_ext}_mask_binary.png")
                
                cv2.imwrite(merged_output_path, merged_img)
                cv2.imwrite(mask_binary_path, mask_binary)
                
                # 更新统计信息
                success_count += 1
                results.append({
                    'filename': name_without_ext,
                    'predict_time': predict_time,
                    'process_time': process_time,
                    'total_time': total_process_time,
                    'status': 'success'
                })
                
                print(f"  预测时间: {predict_time:.2f}秒")
                print(f"  处理时间: {process_time:.2f}秒")
                print(f"  总时间: {total_process_time:.2f}秒")
                print(f"  结果已保存到: {merged_output_path}")
                
                # 如果是交互模式，显示结果并等待用户输入
                if args.interactive:
                    # 创建输入图像并排显示
                    h1, w1 = warp1_img_cv.shape[:2]
                    h2, w2 = warp2_img_cv.shape[:2]
                    max_height = max(h1, h2)
                    
                    # 等比例缩放
                    if h1 != max_height:
                        scale = max_height / h1
                        new_width = int(w1 * scale)
                        warp1_resized = cv2.resize(warp1_img_cv, (new_width, max_height), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        warp1_resized = warp1_img_cv
                        
                    if h2 != max_height:
                        scale = max_height / h2
                        new_width = int(w2 * scale)
                        warp2_resized = cv2.resize(warp2_img_cv, (new_width, max_height), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        warp2_resized = warp2_img_cv
                    
                    # 创建水平拼接图像
                    inputs_img = np.hstack((warp1_resized, warp2_resized))
                    
                    # 显示输入和结果
                    cv2.namedWindow("Input Images", cv2.WINDOW_NORMAL)
                    cv2.imshow("Input Images", inputs_img)
                    cv2.resizeWindow("Input Images", min(1200, inputs_img.shape[1]), min(800, inputs_img.shape[0]))
                    
                    cv2.namedWindow("Merged Result", cv2.WINDOW_NORMAL)
                    cv2.imshow("Merged Result", merged_img)
                    cv2.resizeWindow("Merged Result", min(1200, merged_img.shape[1]), min(800, merged_img.shape[0]))
                    
                    print("按任意键继续到下一张图像 (按ESC退出)")
                    key = cv2.waitKey(0) & 0xFF
                    if key == 27:  # ESC
                        break
                
            except Exception as e:
                print(f"处理图像 {name_without_ext} 失败: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'filename': name_without_ext,
                    'status': 'error',
                    'error': str(e)
                })
    
    # 计算统计信息
    avg_time = total_time / max(success_count, 1)
    print(f"\n处理完成!")
    print(f"处理图像: {success_count}/{max_files}")
    print(f"平均处理时间: {avg_time:.2f}秒")
    
    # 保存处理报告
    with open(os.path.join(output_dir, "processing_report.json"), "w") as f:
        json.dump({
            'total_images': max_files,
            'success_count': success_count,
            'average_time': avg_time,
            'results': results
        }, f, indent=4)
    
    print(f"完整报告已保存至: {os.path.join(output_dir, 'processing_report.json')}")
    
    # 关闭所有窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 