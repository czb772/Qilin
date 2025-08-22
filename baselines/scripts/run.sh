#!/bin/bash
# VLM搜索训练启动脚本
# 用于启动VLM模型的训练任务

# 更新代码库
git pull

# 解析输入参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG_FILE="$2"; shift ;;      # 配置文件路径
        --cuda) VALID_CUDA_DEVICES="$2"; shift ;; # 指定使用的GPU设备
        --rank) NODE_RANK="$2"; shift ;;          # 节点排名（多机训练）
        --num_machines) num_machines="$2"; shift ;; # 机器数量（多机训练）
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# 设置默认值
NODE_RANK=${NODE_RANK:-0}                        # 默认节点排名为0
num_machines=${num_machines:-1}                  # 默认机器数量为1

# 验证配置文件是否存在
if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
    echo "Error: --config is either missing or the file does not exist."
    read -p "Please enter a valid CONFIG_FILE (e.g., config/Qwen2.5-14B.yaml): " CONFIG_FILE
    while [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; do
        echo "Invalid CONFIG_FILE. Please try again."
        read -p "Enter a valid CONFIG_FILE: " CONFIG_FILE
    done
fi

# 验证CUDA设备参数
if [[ -z "$VALID_CUDA_DEVICES" ]]; then
    echo "Warning: --cuda is missing. Using default CUDA_VISIBLE_DEVICES."
    # 自动获取所有可用的GPU设备
    VALID_CUDA_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
fi

# 设置CUDA可见设备环境变量
export CUDA_VISIBLE_DEVICES=${VALID_CUDA_DEVICES}

# 设置输出目录
output_dir=output/$(basename ${CONFIG_FILE} .yaml)
echo "output_dir: ${output_dir}"
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}

# 设置GPU数量和分布式训练参数
sh scripts/set_gpu_num.sh ${CONFIG_FILE} ${num_machines}

# 生成时间戳
TIME_STAMP=$(date +'%Y-%m-%d-%H-%M-%S')

# 根据机器数量决定训练模式
if [[ ${num_machines} -eq 1 ]]; then
    echo "Starting single-node training"
    
    # 单机模式使用动态端口分配
    port=29581
    # 检查端口是否被占用的函数
    is_port_in_use() {
        lsof -i:"$1" > /dev/null 2>&1
        return $?
    }

    # 循环检查端口是否被占用
    while is_port_in_use $port; do
        echo "Port $port is in use. Trying next port..."
        port=$((port + 1))
    done
    echo "Using available port: $port"

    # 单机训练命令
    # 使用accelerate启动分布式训练
    accelerate launch \
        --main_process_port=$port \
        --config_file config/default_config.yaml \
        src/trainer.py ${CONFIG_FILE} ${TIME_STAMP} \
        | tee ${log_dir}/train.log                    
else
    # 多机训练设置
    MASTER_IP="10.152.90.106"                          # 主节点IP地址
    MASTER_PORT=60007                                  # 主节点端口
    NODE_RANK=${NODE_RANK:-0}                         # 当前节点排名

    echo "Starting multi-node training:"
    echo "Master IP: ${MASTER_IP}"
    echo "Master Port: ${MASTER_PORT}"
    echo "Node Rank: ${NODE_RANK}"
    echo "World Size: ${num_machines}"

    # 多机训练命令
    accelerate launch \
        --multi_gpu \
        --num_machines=${num_machines} \
        --machine_rank=${NODE_RANK} \
        --main_process_ip="${MASTER_IP}" \
        --main_process_port=${MASTER_PORT} \
        --config_file config/default_config.yaml \
        src/trainer.py ${CONFIG_FILE} ${TIME_STAMP} ${NODE_RANK} ${num_machines} \
        | tee ${log_dir}/train_node${NODE_RANK}.log    # 保存节点训练日志
fi

echo "=================done train=================="
