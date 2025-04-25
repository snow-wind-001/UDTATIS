# 扩散合成模型可视化工具

本目录包含用于可视化 ImprovedDiffusionComposition 模型中 warp 过程和 composition 过程的工具。这些工具不修改原始代码，而是通过钩子机制捕获中间特征和处理过程。

## 目录结构

```
draw/
├── utils.py                 # 通用可视化工具类
├── visualize_warp.py        # Warp过程可视化脚本
├── visualize_composition.py # Composition过程可视化脚本
├── run_warp_visualization.sh # Warp可视化脚本的辅助运行脚本
├── run_composition_visualization.sh # Composition可视化脚本的辅助运行脚本
└── output/                  # 输出目录(运行后自动创建)
    ├── warp/                # Warp过程输出
    └── composition/         # Composition过程输出
        └── sampling_steps/  # 采样步骤可视化
```

## 依赖安装

确保已安装以下依赖：

```bash
pip install torch torchvision matplotlib opencv-python pillow numpy pandas
```

## 使用方法

### 可视化 Warp 过程

warp过程只需要两个图像目录作为输入：

```bash
python draw/visualize_warp.py [--image1_dir <第一组图像目录>] [--image2_dir <第二组图像目录>] [--output_dir <输出目录>] [--model_path <模型权重文件>] [--device <cuda/cpu>] [--target_size <宽> <高>]
```

参数说明：
- `--image1_dir`: 第一组图像的目录，默认为 `images/image1`
- `--image2_dir`: 第二组图像的目录，默认为 `images/image2`
- `--output_dir`: 输出目录，默认为 `draw/output/warp`
- `--model_path`: 模型权重文件路径（可选）
- `--device`: 运行设备，默认为CUDA（如果可用）或CPU
- `--target_size`: 目标图像尺寸，默认为 `256 256`
- `--demo`: 使用演示模式，即使导入失败也尝试可视化

示例：

```bash
# 使用默认参数，从images/image1和images/image2目录读取图像
python draw/visualize_warp.py

# 指定自定义目录和输出位置
python draw/visualize_warp.py --image1_dir data/source --image2_dir data/target --output_dir results/warp_vis

# 使用辅助脚本运行（更简单）
bash draw/run_warp_visualization.sh
```

### 可视化 Composition 过程

```bash
python draw/visualize_composition.py --warp1 <warp1图像路径> --warp2 <warp2图像路径> [--mask1 <mask1路径>] [--mask2 <mask2路径>] [--output_dir <输出目录>] [--model_path <模型权重文件>] [--device <cuda/cpu>] [--mode <full/forward/sample>] [--vis_steps <可视化步数>]
```

参数说明：
- `--warp1`/`--warp2`: 变形后的图像路径（可以指定多个）
- `--mask1`/`--mask2`: 对应的掩码路径（可选）
- `--output_dir`: 输出目录，默认为 `draw/output/composition`
- `--model_path`: 模型权重文件路径（可选）
- `--device`: 运行设备
- `--mode`: 可视化模式（full/forward/sample）
- `--vis_steps`: 可视化的步数，默认为10
- `--target_size`: 目标图像尺寸
- `--low_memory`: 低内存模式，减少内存使用

示例：

```bash
# 基本用法
python draw/visualize_composition.py --warp1 images/warp1.png --warp2 images/warp2.png --mode full

# 使用掩码和自定义输出目录
python draw/visualize_composition.py --warp1 images/warp1.png --warp2 images/warp2.png --mask1 images/mask1.png --mask2 images/mask2.png --output_dir results/composition_vis

# 使用辅助脚本运行（支持批处理）
bash draw/run_composition_visualization.sh --warp1_dir images/warp1 --warp2_dir images/warp2 --mode sample
```

辅助脚本 `run_composition_visualization.sh` 提供了更多选项：
- `--warp1_dir`/`--warp2_dir`: 可以是目录或单个文件
- `--mask1_dir`/`--mask2_dir`: 掩码目录或文件
- `--save_tables`: 保存指标到CSV表格
- `--vis_steps`: 采样过程可视化步数
- `--custom_masks`: 使用自定义掩码而非模型生成的掩码
- `--force_cpu`: 强制使用CPU运行
- `--low_memory`: 低内存模式

## 输出说明

### Warp 过程输出

- `input_img1_*.png`: 第一个输入图像可视化
- `input_img2_*.png`: 第二个输入图像可视化
- `features_*_*.png`: 不同模块提取的特征图
- `valid_scores_masks_*.png`: 有效点掩码
- `mesh_deformation_*.png`: 变形网格可视化
- `warped_img_homo_*.png`: 单应性变换结果
- `warped_img_tps_*.png`: TPS变换结果
- `original_vs_warped_comparison_*.png`: 原始图像和变形后图像的对比

### Composition 过程输出

- `input_warp1_*.png`: 第一个输入图像
- `input_warp2_*.png`: 第二个输入图像
- `input_mask1_masks_*.png`: 第一个掩码(如有)
- `input_mask2_masks_*.png`: 第二个掩码(如有)
- `forward_learned_mask_masks_*.png`: 前向传播学习的掩码
- `forward_denoised_*.png`: 前向传播去噪结果
- `forward_*_features_*.png`: 各模块的前向特征图
- `sampling_steps/sampling_step_*_t*.png`: 采样过程中的中间步骤
- `final_*_*.png`: 采样最终结果
- `build_model_*_*.png`: 使用build_model函数的结果
- `*_comparison_*.png`: 输入与输出的并排比较

## 特征图可视化

对于每个模块输出的特征图，系统会自动将多通道特征图转换为彩色热力图显示。主要可视化模块包括：

1. **下采样层特征图**: 展示扩散模型下采样过程中的特征提取情况
2. **中间层特征图**: 展示扩散模型中间处理过程中的特征
3. **上采样层特征图**: 展示扩散模型上采样过程中的特征重建
4. **掩码生成器输出**: 展示学习到的混合掩码
5. **噪声预测结果**: 展示模型预测的噪声分布
6. **去噪结果**: 展示最终的去噪图像

## 修改与扩展

如果需要自定义可视化过程，您可以修改以下文件：

- `utils.py`: 添加新的可视化方法
- `visualize_warp.py`: 修改warp过程的钩子或可视化方法
- `visualize_composition.py`: 修改composition过程的钩子或可视化方法

## 注意事项

1. 确保输入图像具有相同的尺寸或使用 `--target_size` 参数指定尺寸
2. 对于复杂图像，采样过程可能较慢，可以通过调整 `vis_steps` 参数减少可视化的步骤数
3. 如果GPU内存有限，建议使用 `--device cpu` 或 `--force_cpu` 在CPU上运行
4. 如果系统内存有限，可使用 `--low_memory` 参数降低内存占用
5. Warp过程会自动处理目录中所有支持的图像格式(png, jpg, jpeg, bmp, tiff, webp)
6. Composition可视化支持批处理多对图像，使用辅助脚本更方便 