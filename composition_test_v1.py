#!/usr/bin/env python
import os
import cv2
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import sys

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
    
    # 将mask_binary从单通道转换为三通道用于可视化
    mask_3channel = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
    
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像合成测试')
    parser.add_argument('--warp1', type=str, required=True, help='第一张图像路径')
    parser.add_argument('--warp2', type=str, required=True, help='第二张图像路径')
    parser.add_argument('--mask1', type=str, required=True, help='第一张图像掩码路径')
    parser.add_argument('--mask', type=str, required=True, help='网络生成的掩码路径')
    parser.add_argument('--output', type=str, default='merged.png', help='输出合并图像路径')
    parser.add_argument('--threshold', type=int, default=127, help='掩码二值化阈值(0-255)')
    parser.add_argument('--interactive', action='store_true', help='启用交互模式')
    
    args = parser.parse_args()
    
    # 加载图像
    warp1_img = cv2.imread(args.warp1)
    warp2_img = cv2.imread(args.warp2)
    mask1_img = cv2.imread(args.mask1, cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    
    if warp1_img is None or warp2_img is None or mask1_img is None or mask_img is None:
        print("错误：无法加载图像文件")
        sys.exit(1)
    
    # 当前阈值
    current_threshold = args.threshold
    
    if args.interactive:
        print("使用说明:")
        print("  - 按回车键 (Enter): 保存并退出")
        print("  - 按 '+' 键: 增加阈值")
        print("  - 按 '-' 键: 减小阈值")
        print("  - 按 's' 键: 保存当前图像")
        print("  - 按 ESC 键: 退出不保存")
        
        while True:
            # 使用当前阈值处理图像
            warp1_display, warp2_display, merged_img, mask_binary = process_mask_for_composition(
                mask_img, warp1_img, warp2_img, mask1_img, current_threshold
            )
            
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
            
            # 在结果窗口上显示阈值信息
            info_img = merged_img.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(info_img, f"Threshold: {current_threshold}", (20, 30), font, 1, (0, 255, 255), 2)
            cv2.imshow("Merged Result", info_img)
            
            # 等待键盘输入
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC键
                cv2.destroyAllWindows()
                print("取消操作")
                return
            elif key == 13 or key == 10:  # 回车键
                # 保存当前结果并退出
                cv2.imwrite(args.output, merged_img)
                mask_output_path = os.path.splitext(args.output)[0] + "_mask.png"
                cv2.imwrite(mask_output_path, mask_binary)
                
                print(f"已保存合并图像到: {args.output}")
                print(f"已保存二值化mask到: {mask_output_path}")
                cv2.destroyAllWindows()
                break
            elif key == ord('+') or key == ord('='):  # '+'键
                # 增加阈值
                current_threshold = min(255, current_threshold + 10)
                print(f"阈值增加到: {current_threshold}")
            elif key == ord('-') or key == ord('_'):  # '-'键
                # 减小阈值
                current_threshold = max(0, current_threshold - 10)
                print(f"阈值减小到: {current_threshold}")
            elif key == ord('s'):  # 's'键
                # 保存当前结果但不退出
                cv2.imwrite(args.output, merged_img)
                mask_output_path = os.path.splitext(args.output)[0] + "_mask.png"
                cv2.imwrite(mask_output_path, mask_binary)
                
                print(f"已保存合并图像到: {args.output}")
                print(f"已保存二值化mask到: {mask_output_path}")
    else:
        # 非交互模式 - 直接处理并保存
        _, _, merged_img, mask_binary = process_mask_for_composition(
            mask_img, warp1_img, warp2_img, mask1_img, args.threshold
        )
        
        # 保存结果
        cv2.imwrite(args.output, merged_img)
        mask_output_path = os.path.splitext(args.output)[0] + "_mask.png"
        cv2.imwrite(mask_output_path, mask_binary)
        
        print(f"已保存合并图像到: {args.output}")
        print(f"已保存二值化mask到: {mask_output_path}")

if __name__ == "__main__":
    main() 