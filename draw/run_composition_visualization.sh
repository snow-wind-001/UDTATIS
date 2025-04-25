#!/bin/bash

# 运行composition可视化脚本
# 使用示例: ./run_composition_visualization.sh --mode full

# 默认参数
WARP1_DIR="images/warp1"
WARP2_DIR="images/warp2"
MASK1_DIR="images/mask1"
MASK2_DIR="images/mask2"
OUTPUT_DIR="draw/output/composition"
MODEL_PATH=""
MODE="full"
SAVE_TABLES=false
VIS_STEPS=10
TARGET_SIZE="256 256"
DEVICE="cuda"
CUSTOM_MASKS=false
FORCE_CPU=false
LOW_MEMORY=false

# 处理命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --warp1_dir)
      WARP1_DIR="$2"
      shift
      shift
      ;;
    --warp2_dir)
      WARP2_DIR="$2"
      shift
      shift
      ;;
    --mask1_dir)
      MASK1_DIR="$2"
      shift
      shift
      ;;
    --mask2_dir)
      MASK2_DIR="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift
      shift
      ;;
    --mode)
      MODE="$2"
      shift
      shift
      ;;
    --save_tables)
      SAVE_TABLES=true
      shift
      ;;
    --vis_steps)
      VIS_STEPS="$2"
      shift
      shift
      ;;
    --target_size)
      TARGET_SIZE="$2 $3"
      shift
      shift
      shift
      ;;
    --device)
      DEVICE="$2"
      shift
      shift
      ;;
    --custom_masks)
      CUSTOM_MASKS=true
      shift
      ;;
    --force_cpu)
      FORCE_CPU=true
      shift
      ;;
    --low_memory)
      LOW_MEMORY=true
      shift
      ;;
    *)
      echo "未知参数: $1"
      shift
      ;;
  esac
done

# 构建warp1和warp2参数
WARP1_ARGS=""
WARP2_ARGS=""
MASK1_ARGS=""
MASK2_ARGS=""

# 检查目录是否存在
if [ -d "$WARP1_DIR" ]; then
  for file in "$WARP1_DIR"/*; do
    if [[ $file == *.png || $file == *.jpg || $file == *.jpeg ]]; then
      WARP1_ARGS="$WARP1_ARGS --warp1 $file"
    fi
  done
else
  # 如果不是目录，直接使用参数
  WARP1_ARGS="--warp1 $WARP1_DIR"
fi

if [ -d "$WARP2_DIR" ]; then
  for file in "$WARP2_DIR"/*; do
    if [[ $file == *.png || $file == *.jpg || $file == *.jpeg ]]; then
      WARP2_ARGS="$WARP2_ARGS --warp2 $file"
    fi
  done
else
  # 如果不是目录，直接使用参数
  WARP2_ARGS="--warp2 $WARP2_DIR"
fi

# 处理mask1和mask2
if [ -d "$MASK1_DIR" ]; then
  for file in "$MASK1_DIR"/*; do
    if [[ $file == *.png || $file == *.jpg || $file == *.jpeg ]]; then
      MASK1_ARGS="$MASK1_ARGS --mask1 $file"
    fi
  done
elif [ "$MASK1_DIR" != "" ]; then
  # 如果不是目录但不为空，直接使用参数
  MASK1_ARGS="--mask1 $MASK1_DIR"
fi

if [ -d "$MASK2_DIR" ]; then
  for file in "$MASK2_DIR"/*; do
    if [[ $file == *.png || $file == *.jpg || $file == *.jpeg ]]; then
      MASK2_ARGS="$MASK2_ARGS --mask2 $file"
    fi
  done
elif [ "$MASK2_DIR" != "" ]; then
  # 如果不是目录但不为空，直接使用参数
  MASK2_ARGS="--mask2 $MASK2_DIR"
fi

# 添加模型路径参数
MODEL_ARGS=""
if [ "$MODEL_PATH" != "" ]; then
  MODEL_ARGS="--model_path $MODEL_PATH"
fi

# 添加保存表格参数
TABLES_ARGS=""
if [ "$SAVE_TABLES" = true ]; then
  TABLES_ARGS="--save_tables"
fi

# 添加自定义掩码参数
CUSTOM_MASKS_ARGS=""
if [ "$CUSTOM_MASKS" = true ]; then
  CUSTOM_MASKS_ARGS="--custom_masks"
fi

# 添加强制CPU模式参数
FORCE_CPU_ARGS=""
if [ "$FORCE_CPU" = true ]; then
  FORCE_CPU_ARGS="--force_cpu"
fi

# 添加低内存模式参数
LOW_MEMORY_ARGS=""
if [ "$LOW_MEMORY" = true ]; then
  LOW_MEMORY_ARGS="--low_memory"
fi

# 构建完整命令
CMD="python draw/visualize_composition.py $WARP1_ARGS $WARP2_ARGS $MASK1_ARGS $MASK2_ARGS --output_dir $OUTPUT_DIR $MODEL_ARGS --mode $MODE --vis_steps $VIS_STEPS --target_size $TARGET_SIZE --device $DEVICE $TABLES_ARGS $CUSTOM_MASKS_ARGS $FORCE_CPU_ARGS $LOW_MEMORY_ARGS"

echo "执行命令: $CMD"
echo "================================"

# 执行命令
eval $CMD

echo "================================"
echo "完成！输出已保存到: $OUTPUT_DIR" 