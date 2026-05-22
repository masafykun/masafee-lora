#!/bin/bash
set -e
cd /home/user/masafee-lora
source venv/bin/activate
cd sd-scripts
accelerate launch --num_cpu_threads_per_process 4 train_network.py \
  --pretrained_model_name_or_path=/home/user/masafee-lora/models/v1-5-pruned-emaonly.safetensors \
  --train_data_dir=/home/user/masafee-lora/train_data/img \
  --output_dir=/home/user/masafee-lora/output \
  --output_name=masafee_v1 \
  --logging_dir=/home/user/masafee-lora/logs \
  --resolution=512,512 \
  --enable_bucket --min_bucket_reso=256 --max_bucket_reso=1024 --bucket_reso_steps=64 \
  --network_module=networks.lora \
  --network_dim=32 --network_alpha=16 \
  --train_batch_size=2 \
  --max_train_epochs=16 \
  --learning_rate=1e-4 --unet_lr=1e-4 --text_encoder_lr=5e-5 \
  --lr_scheduler=cosine --lr_warmup_steps=0 \
  --optimizer_type=AdamW8bit \
  --mixed_precision=bf16 --save_precision=fp16 \
  --save_model_as=safetensors \
  --save_every_n_epochs=4 \
  --caption_extension=.txt \
  --shuffle_caption --keep_tokens=1 \
  --flip_aug --cache_latents \
  --sdpa --gradient_checkpointing \
  --seed=42 --max_data_loader_n_workers=2 \
  --sample_every_n_epochs=4 \
  --sample_prompts=/home/user/masafee-lora/sample_prompts.txt \
  --sample_sampler=euler_a
echo "=== TRAINING DONE exit=$? ==="
