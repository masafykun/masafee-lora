#!/bin/bash
SWEEP=/home/user/masafee-lora/sweep
SD15=/home/user/masafee-lora/models/v1-5-pruned-emaonly.safetensors
CF=/home/user/masafee-lora/models/Counterfeit-V3.0_fp16.safetensors
cd /home/user/masafee-lora
source venv/bin/activate
cd sd-scripts
mkdir -p $SWEEP/loras $SWEEP/gen $SWEEP/grids

CONFIGS=(
"sd15_lr1_d16 $SD15 1e-4 16 8"
"sd15_lr1_d32 $SD15 1e-4 32 16"
"sd15_lr2_d32 $SD15 2e-4 32 16"
"cf_lr1_d16 $CF 1e-4 16 8"
"cf_lr1_d32 $CF 1e-4 32 16"
"cf_lr2_d32 $CF 2e-4 32 16"
)

echo "### SWEEP START $(date)"
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg
  name=$1; base=$2; lr=$3; dim=$4; alpha=$5
  if [ ! -f "$base" ]; then echo "### SKIP $name (base missing)"; continue; fi
  echo "### TRAIN $name lr=$lr dim=$dim/$alpha $(date)"
  accelerate launch --num_cpu_threads_per_process 4 train_network.py \
    --pretrained_model_name_or_path="$base" \
    --train_data_dir=/home/user/masafee-lora/train_data/img \
    --output_dir=$SWEEP/loras --output_name=$name \
    --resolution=768,768 \
    --enable_bucket --min_bucket_reso=320 --max_bucket_reso=1024 --bucket_reso_steps=64 \
    --network_module=networks.lora --network_dim=$dim --network_alpha=$alpha \
    --train_batch_size=2 --max_train_epochs=30 \
    --learning_rate=$lr --lr_scheduler=cosine --lr_warmup_steps=0 \
    --optimizer_type=AdamW8bit \
    --mixed_precision=bf16 --save_precision=fp16 --save_model_as=safetensors \
    --save_every_n_epochs=10 \
    --caption_extension=.txt --shuffle_caption --keep_tokens=1 \
    --flip_aug --cache_latents --sdpa --gradient_checkpointing \
    --seed=42 --max_data_loader_n_workers=2 || echo "### TRAIN FAILED $name"
done
echo "### TRAINING PHASE DONE $(date)"

for lora in $SWEEP/loras/*.safetensors; do
  [ -f "$lora" ] || continue
  bn=$(basename "$lora" .safetensors)
  case $bn in
    sd15*) base=$SD15;;
    cf*) base=$CF;;
    *) base=$SD15;;
  esac
  outdir=$SWEEP/gen/$bn
  mkdir -p "$outdir"
  echo "### GEN $bn $(date)"
  python gen_img.py --ckpt "$base" \
    --network_module networks.lora --network_weights "$lora" --network_mul 0.8 \
    --outdir "$outdir" --from_file $SWEEP/sweep_prompts.txt \
    --images_per_prompt 1 --sampler euler_a --fp16 --sdpa --batch_size 1 \
    || echo "### GEN FAILED $bn"
done
echo "### GEN PHASE DONE $(date)"

python $SWEEP/make_grids.py || echo "### GRID FAILED"
echo "### SWEEP ALL DONE $(date)"
