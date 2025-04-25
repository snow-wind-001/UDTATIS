#!/bin/bash

# 确保输出目录存在
mkdir -p draw/output/warp

# 确保图像目录存在
mkdir -p images/image1
mkdir -p images/image2

# 查找模型文件
MODEL_PATH=""
if [ -d "Warp/model" ]; then
    # 查找最新的模型文件
    LATEST_MODEL=$(find Warp/model -name "*.pth" | sort | tail -n 1)
    if [ ! -z "$LATEST_MODEL" ]; then
        MODEL_PATH="--model_path $LATEST_MODEL"
        echo "找到Warp模型: $LATEST_MODEL"
    fi
fi

# 检查图像目录是否为空
if [ -z "$(ls -A images/image1)" ] || [ -z "$(ls -A images/image2)" ]; then
    echo "警告: images/image1 或 images/image2 目录为空!"
    echo "请将图像放入对应目录后再运行此脚本。"
    exit 1
fi

# 运行warp过程可视化
echo "开始可视化warp过程..."
python draw/visualize_warp.py --image1_dir images/image1 --image2_dir images/image2 --output_dir draw/output/warp $MODEL_PATH

# 检查是否成功执行
if [ $? -eq 0 ]; then
    echo "可视化成功! 结果已保存至 draw/output/warp"
    echo "查看结果: ls -la draw/output/warp"
else
    echo "可视化过程失败，请检查错误信息。"
fi 