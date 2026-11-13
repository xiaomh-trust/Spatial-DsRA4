# [Spatial-DsRA v4]: Set experiment-specific values explicitly; the design gives no defaults.
: "${SPATIAL_WARMUP_STEPS:?Set Stage I optimizer steps}"
: "${SPATIAL_ABSORPTION_START:?Set Stage III starting optimizer step}"
: "${SPATIAL_ORTH_COEFF:?Set eta}"
: "${SPATIAL_BASIS_LR_SCALE:?Set a learning rate multiplier between 0 and 1}"
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_align.py \
  --spatial_warmup_steps "$SPATIAL_WARMUP_STEPS" \
  --spatial_absorption_start "$SPATIAL_ABSORPTION_START" \
  --spatial_orth_coeff "$SPATIAL_ORTH_COEFF" \
  --spatial_basis_lr_scale "$SPATIAL_BASIS_LR_SCALE" \
  --vla_path ckpts/openvla-7b \
  --vggt_path ckpts/model.pt \
  --data_root_dir data/libero/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir ckpts/training_results/ \
  --pooling_func bilinear \
  --vggt_layers_align -1 \
  --align_loss_type cosine \
  --align_loss_coeff 0.5 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --use_vlm_norm True \
  --use_vggt_pe True \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only True \
  --merge_lora_during_training False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_override "YOUR_RUN_ID"
