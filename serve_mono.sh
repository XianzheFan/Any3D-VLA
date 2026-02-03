apt update
apt install -y libgl1

MODEL_NAME=grit-Concerto-mono-dinosiglip-16-128-40000
MODEL_STEP=340000

CODE_DIR=./any3d-vla
CKPT_DIR=./any3d-vla/storage/ckpt/exp
CHECKPOINT=$CKPT_DIR/$MODEL_NAME/checkpoint-$MODEL_STEP/model.safetensors
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
  NUM_GPUS=$(ls /dev | grep -E 'nvidia[0-9]+' | wc -l)
fi

export HF_ENDPOINT=https://hf-mirror.com
export GX_STORAGE_PATH=$CODE_DIR/storage
export PYTHONPATH=$PYTHONPATH:./any3d-vla
export SDL_AUDIODRIVER=dummy

cd $CODE_DIR
python3 -u -m vla_network.scripts.serve --dataset-statistics dummy --path $CHECKPOINT --port 6666 --batch-size 1
sleep inf