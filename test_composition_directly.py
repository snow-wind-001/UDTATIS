#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import glob
import argparse
from datetime import datetime
import time

# 添加项目根目录到路径
sys.path.append('.')

# 从main.py中导入处理函数
from main import process_mask_for_composition

def main():
    parser = argparse.ArgumentParser(description='直接测试Composition功能')
    parser.add_argument('--data_dir', type=str, default='data/UDIS-D/composition_data/test',
                        help='数据目录，包含warp1, warp2, mask1, mask2子目录')
    parser.add_argument('--output_dir', type=str, default='Composition/direct_test_results',
                        help='输出结果目录')
    parser.add_argument('--num_images', type=int, default=10,
                        help='处理的图像数量')
    parser.add_argument('--threshold', type=int, default=127,
                        help='二值化阈值(0-255)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建结果子目录
    warp1_dir = os.path.join(output_dir, "warp1")
    warp2_dir = os.path.join(output_dir, "warp2")
    merged_dir = os.path.join(output_dir, "merged")
    mask_dir = os.path.join(output_dir, "mask")
    
    os.makedirs(warp1_dir, exist_ok=True)
    os.makedirs(warp2_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # 获取输入数据路径
    input_warp1_dir = os.path.join(args.data_dir, "warp1")
    input_warp2_dir = os.path.join(args.data_dir, "warp2")
    input_mask1_dir = os.path.join(args.data_dir, "mask1")
    input_mask2_dir = os.path.join(args.data_dir, "mask2")
    
    # 获取输入文件列表
    warp1_files = sorted(glob.glob(os.path.join(input_warp1_dir, "*.*")))
    warp2_files = sorted(glob.glob(os.path.join(input_warp2_dir, "*.*")))
    mask1_files = sorted(glob.glob(os.path.join(input_mask1_dir, "*.*")))
    mask2_files = sorted(glob.glob(os.path.join(input_mask2_dir, "*.*")))
    
    # 确保文件列表非空且长度一致
    if not warp1_files or not warp2_files or not mask1_files or not mask2_files:
        print(f"无法找到测试图像，请检查路径: {args.data_dir}")
        return
    
    # 确定处理的图像数量
    max_images = min(len(warp1_files), len(warp2_files), len(mask1_files), len(mask2_files), args.num_images)
    
    print(f"开始处理 {max_images} 张图像...")
    
    # 处理图像
    for i in range(max_images):
        try:
            # 获取文件名
            warp1_file = warp1_files[i]
            warp2_file = warp2_files[i]
            mask1_file = mask1_files[i]
            mask2_file = mask2_files[i]
            
            file_name = os.path.basename(warp1_file)
            name_without_ext = os.path.splitext(file_name)[0]
            
            print(f"\n处理图像 {i+1}/{max_images}: {name_without_ext}")
            
            # 读取图像
            warp1_img = cv2.imread(warp1_file)
            warp2_img = cv2.imread(warp2_file)
            mask1_img = cv2.imread(mask1_file, cv2.IMREAD_GRAYSCALE)
            mask2_img = cv2.imread(mask2_file, cv2.IMREAD_GRAYSCALE)
            
            if warp1_img is None or warp2_img is None or mask1_img is None or mask2_img is None:
                print(f"  Error: 无法加载图像 {name_without_ext}")
                continue
            
            # 使用mask2作为mask，演示目的
            # 在实际使用中，应该使用神经网络生成的mask
            mask_img = mask2_img
            
            # 处理图像
            start_time = time.time()
            warp1_display, warp2_display, merged_img, mask_binary = process_mask_for_composition(
                mask_img, warp1_img, warp2_img, mask1_img, args.threshold
            )
            process_time = time.time() - start_time
            
            # 保存结果
            warp1_output_path = os.path.join(warp1_dir, f"{name_without_ext}.png")
            warp2_output_path = os.path.join(warp2_dir, f"{name_without_ext}.png")
            merged_output_path = os.path.join(merged_dir, f"{name_without_ext}.png")
            mask_output_path = os.path.join(mask_dir, f"{name_without_ext}.png")
            
            cv2.imwrite(warp1_output_path, warp1_display)
            cv2.imwrite(warp2_output_path, warp2_display)
            cv2.imwrite(merged_output_path, merged_img)
            cv2.imwrite(mask_output_path, mask_binary)
            
            print(f"  处理时间: {process_time:.2f}秒")
            print(f"  结果已保存到: {merged_output_path}")
            
            # 显示结果（可选）
            if False:  # 设置为True可以在处理过程中显示结果
                cv2.namedWindow("Merged Result", cv2.WINDOW_NORMAL)
                cv2.imshow("Merged Result", merged_img)
                cv2.waitKey(500)  # 显示0.5秒
            
        except Exception as e:
            print(f"  Error processing image {name_without_ext}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n处理完成! 结果保存在: {output_dir}")
    
    # 关闭所有窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 