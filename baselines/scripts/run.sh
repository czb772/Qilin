git pull

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG_FILE="$2"; shift ;;
        --cuda) VALID_CUDA_DEVICES="$2"; shift ;;
        --rank) NODE_RANK="$2"; shift ;;
        --num_machines) num_machines="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
NODE_RANK=${NODE_RANK:-0}  # Default to 0 if not provided
num_machines=${num_machines:-1}  # Default to 1 if not provided

# Validate CONFIG_FILE
if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
    echo "Error: --config is either missing or the file does not exist."
    read -p "Please enter a valid CONFIG_FILE (e.g., config/Qwen2.5-14B.yaml): " CONFIG_FILE
    while [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; do
        echo "Invalid CONFIG_FILE. Please try again."
        read -p "Enter a valid CONFIG_FILE: " CONFIG_FILE
    done
fi

# Validate VALID_CUDA_DEVICES
if [[ -z "$VALID_CUDA_DEVICES" ]]; then
    echo "Warning: --cuda is missing. Using default CUDA_VISIBLE_DEVICES."
    VALID_CUDA_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
fi

export CUDA_VISIBLE_DEVICES=${VALID_CUDA_DEVICES}
output_dir=output/$(basename ${CONFIG_FILE} .yaml)
echo "output_dir: ${output_dir}"
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
sh scripts/set_gpu_num.sh ${CONFIG_FILE} ${num_machines}

TIME_STAMP=$(date +'%Y-%m-%d-%H-%M-%S')

# Determine training mode based on whether num_machines is specified
if [[ ${num_machines} -eq 1 ]]; then
    echo "Starting single-node training"
    
    # Single machine mode uses dynamic port allocation
    port=29581
    # Function to check if port is occupied
    is_port_in_use() {
        lsof -i:"$1" > /dev/null 2>&1
        return $?
    }

    # Loop to check if port is occupied
    while is_port_in_use $port; do
        echo "Port $port is in use. Trying next port..."
        port=$((port + 1))
    done
    echo "Using available port: $port"

    # Single machine training command
    accelerate launch \
        --main_process_port=$port \
        --config_file config/default_config.yaml \
        src/trainer.py ${CONFIG_FILE} ${TIME_STAMP} \
        | tee ${log_dir}/train.log
else
    # Multi-machine training setup
    MASTER_IP="10.152.90.106"
    MASTER_PORT=60007
    NODE_RANK=${NODE_RANK:-0}

    echo "Starting multi-node training:"
    echo "Master IP: ${MASTER_IP}"
    echo "Master Port: ${MASTER_PORT}"
    echo "Node Rank: ${NODE_RANK}"
    echo "World Size: ${num_machines}"

    # Multi-machine training command
    accelerate launch \
        --multi_gpu \
        --num_machines=${num_machines} \
        --machine_rank=${NODE_RANK} \
        --main_process_ip="${MASTER_IP}" \
        --main_process_port=${MASTER_PORT} \
        --config_file config/default_config.yaml \
        src/trainer.py ${CONFIG_FILE} ${TIME_STAMP} ${NODE_RANK} ${num_machines} \
        | tee ${log_dir}/train_node${NODE_RANK}.log
fi

echo "=================done train=================="
