#!/bin/bash

# 使用环境变量中的 RANK（从0开始）
if [ -z "$RANK" ]; then
    echo "Error: RANK environment variable not set"
    exit 1
fi

# 基础起始索引和每个节点的增量
BASE_START_INDEX=0
NODE_INCREMENT=50

# 计算当前节点的起始索引
START_INDEX=$((BASE_START_INDEX + RANK * NODE_INCREMENT))
INCREMENT=10

# 定义 Xvfb 的基准编号起点
XVFB_BASE_DISPLAY=1

# 定义并行任务数量
NUM_TASKS=5

# Python 脚本路径
PYTHON_SCRIPT="python"
SCRIPT_PATH="src/evaluate3.py"
CONFIG_NAME="config_alfred_clean"

# 工作目录
WORK_DIR="./"

# 创建该节点的日志目录
LOG_DIR="${WORK_DIR}/logs_clean/node_${RANK}"
mkdir -p $LOG_DIR

echo "节点 RANK=$RANK 开始运行，起始索引：$START_INDEX"

# 启动并行任务
for i in $(seq 0 $((NUM_TASKS - 1))); do
    # 计算 eval_start_index 和 eval_end_index
    EVAL_START=$((START_INDEX + i * INCREMENT))
    EVAL_END=$((EVAL_START + INCREMENT))

    # 计算 Xvfb 显示编号
    DISPLAY_NUM=$((XVFB_BASE_DISPLAY + i))

    # 启动 Xvfb
    Xvfb :$DISPLAY_NUM &

    # 切换到工作目录并运行 Python 脚本
    cd $WORK_DIR
    $PYTHON_SCRIPT $SCRIPT_PATH --config-name=$CONFIG_NAME \
        alfred.x_display="$DISPLAY_NUM" \
        alfred.eval_start_index=$EVAL_START \
        alfred.eval_end_index=$EVAL_END \
        > "${LOG_DIR}/task_${i}.log" 2>&1 &

    echo "节点 RANK=$RANK - 启动任务 $((i + 1))：Xvfb :$DISPLAY_NUM，eval_start_index=$EVAL_START，eval_end_index=$EVAL_END"
done

# 等待所有任务完成
wait

echo "节点 RANK=$RANK 的所有任务完成！"