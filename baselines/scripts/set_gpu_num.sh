CONFIG_FILE=$1
WORLD_SIZE=$2
cat config/default_config.yaml.sample > config/default_config.yaml
# num_processes=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# echo "num_processes: "$num_processes >>  config/default_config.yaml
# Get CUDA_VISIBLE_DEVICES environment variable
cuda_devices=${CUDA_VISIBLE_DEVICES:-"Not Set"}
echo "cuda_devices: $cuda_devices"
num_processes=$(echo "$cuda_devices" | awk -F',' '{print NF}')
# num_processes is current node's GPU count * WORLD_SIZE
num_processes=$((num_processes * WORLD_SIZE))
echo "num_processes: $num_processes"
echo "num_processes: $num_processes" >> config/default_config.yaml
echo "num_machines: $WORLD_SIZE"
echo "num_machines: $WORLD_SIZE" >> config/default_config.yaml

# If config filename contains dcn, set mixed_precision to no
if [[ "$CONFIG_FILE" == *"dcn"* ]]; then
    sed -i 's/mixed_precision:.*$/mixed_precision: "no"/g' config/default_config.yaml
fi
