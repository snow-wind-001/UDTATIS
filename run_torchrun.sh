#!/bin/bash
# UDTATIS 分布式训练启动脚本 (torchrun版)
# 这个脚本使用PyTorch的torchrun命令启动分布式训练

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # 无颜色

echo -e "${BLUE}=== UDTATIS 分布式训练启动脚本 (torchrun版) ===${NC}"
echo

# 默认参数
GPUS=$(nvidia-smi --list-gpus | wc -l)
PART="composition"
BATCH_SIZE=8
EPOCHS=100
DATA_DIR="data/UDIS-D/composition_data"
GRAD_ACCUM_STEPS=1
SYNC_BN=0
USE_AMP=1
PRETRAINED=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --part)
      PART="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --grad_accum_steps)
      GRAD_ACCUM_STEPS="$2"
      shift 2
      ;;
    --sync_bn)
      SYNC_BN=1
      shift
      ;;
    --no_amp)
      USE_AMP=0
      shift
      ;;
    --pretrained)
      PRETRAINED="$2"
      shift 2
      ;;
    --help)
      echo "使用方法: $0 [选项]"
      echo
      echo "选项:"
      echo "  --gpus NUM        要使用的GPU数量 (默认: 1)"
      echo "  --part PART       要训练的模块 (composition 或 warp, 默认: composition)"
      echo "  --batch_size NUM  每个GPU的批次大小 (默认: 8)"
      echo "  --epochs NUM      训练轮数 (默认: 100)"
      echo "  --data_dir DIR    数据目录 (默认: data/UDIS-D/composition_data)"
      echo "  --grad_accum_steps NUM  梯度累积步数 (默认: 1)"
      echo "  --sync_bn         启用同步BatchNorm"
      echo "  --no_amp          禁用混合精度训练"
      echo "  --pretrained PATH 预训练模型路径"
      echo "  --help            显示此帮助信息并退出"
      echo
      echo "示例:"
      echo "  $0 --gpus 4 --part warp --batch_size 16"
      echo "  $0 --gpus 2 --part composition --grad_accum_steps 2 --sync_bn"
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      echo "使用 $0 --help 获取帮助信息"
      exit 1
      ;;
  esac
done

# 显示配置信息
echo -e "${GREEN}训练配置:${NC}"
echo "GPU数量: $GPUS"
echo "模块: $PART"
echo "批次大小: $BATCH_SIZE (每个GPU)"
echo "有效批次大小: $((BATCH_SIZE * GPUS))"
echo "训练轮数: $EPOCHS"
echo "数据目录: $DATA_DIR"
echo "梯度累积: $GRAD_ACCUM_STEPS 步"
echo "同步BatchNorm: $([ $SYNC_BN -eq 1 ] && echo "启用" || echo "禁用")"
echo "混合精度训练: $([ $USE_AMP -eq 1 ] && echo "启用" || echo "禁用")"
if [ ! -z "$PRETRAINED" ]; then
  echo "预训练模型: $PRETRAINED"
fi
echo

# 构建命令
CMD="torchrun --nproc_per_node=$GPUS main.py --mode train --part $PART --distributed --world_size $GPUS --batch_size $BATCH_SIZE --data_dir $DATA_DIR --grad_accum_steps $GRAD_ACCUM_STEPS"

# 添加可选参数
if [ $SYNC_BN -eq 1 ]; then
  CMD="$CMD --sync_bn"
fi

if [ $USE_AMP -eq 0 ]; then
  CMD="$CMD --no_amp"
fi

# 根据模块添加正确的轮数参数
if [ "$PART" == "warp" ]; then
  CMD="$CMD --warp_epochs $EPOCHS"
else
  CMD="$CMD --comp_epochs $EPOCHS"
fi

if [ ! -z "$PRETRAINED" ]; then
  CMD="$CMD --pretrained $PRETRAINED"
fi

# 显示将要执行的命令
echo -e "${BLUE}将执行命令:${NC}"
echo "$CMD"
echo

# 询问用户是否继续
read -p "是否继续? [y/N] " response
if [[ ! "$response" =~ ^[yY]$ ]]; then
  echo "已取消"
  exit 0
fi

# 执行命令
echo -e "${GREEN}开始训练...${NC}"
eval $CMD

# 检查退出状态
EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
  echo -e "${GREEN}训练完成!${NC}"
else
  echo -e "${RED}训练出错，退出状态: $EXIT_STATUS${NC}"
  echo "查看上面的错误消息以了解问题所在。"
fi

exit $EXIT_STATUS 