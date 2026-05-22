#!/bin/bash
SWEEP=/home/user/masafee-lora/sweep
EXT=$SWEEP/ext60
CF=/home/user/masafee-lora/models/Counterfeit-V3.0_fp16.safetensors
cd /home/user/masafee-lora
source venv/bin/activate
cd sd-scripts
mkdir -p $EXT/loras $EXT/gen $EXT/grids

echo "### EXT60 TRAIN START $(date)"
accelerate launch --num_cpu_threads_per_process 4 train_network.py \
  --pretrained_model_name_or_path="$CF" \
  --train_data_dir=/home/user/masafee-lora/train_data/img \
  --output_dir=$EXT/loras --output_name=cf60 \
  --resolution=768,768 \
  --enable_bucket --min_bucket_reso=320 --max_bucket_reso=1024 --bucket_reso_steps=64 \
  --network_module=networks.lora --network_dim=32 --network_alpha=16 \
  --train_batch_size=2 --max_train_epochs=60 \
  --learning_rate=1e-4 --lr_scheduler=cosine --lr_warmup_steps=0 \
  --optimizer_type=AdamW8bit \
  --mixed_precision=bf16 --save_precision=fp16 --save_model_as=safetensors \
  --save_every_n_epochs=10 \
  --caption_extension=.txt --shuffle_caption --keep_tokens=1 \
  --flip_aug --cache_latents --sdpa --gradient_checkpointing \
  --seed=42 --max_data_loader_n_workers=2 || echo "### TRAIN FAILED"
echo "### EXT60 TRAIN DONE $(date)"

for ck in "cf60-000030 e30" "cf60-000040 e40" "cf60-000050 e50" "cf60 e60"; do
  set -- $ck
  file=$EXT/loras/$1.safetensors
  label=$2
  if [ ! -f "$file" ]; then echo "### MISSING $1"; continue; fi
  mkdir -p $EXT/gen/$label
  echo "### GEN $label $(date)"
  python gen_img.py --ckpt "$CF" \
    --network_module networks.lora --network_weights "$file" --network_mul 0.8 \
    --outdir $EXT/gen/$label --from_file $SWEEP/sweep_prompts.txt \
    --images_per_prompt 1 --sampler euler_a --fp16 --sdpa --batch_size 1 || echo "### GEN FAILED $label"
done
echo "### EXT60 GEN DONE $(date)"
python $EXT/make_grid_ext.py || echo "### GRID FAILED"
echo "### EXT60 ALL DONE $(date)"
