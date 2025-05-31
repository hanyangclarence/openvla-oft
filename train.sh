export HF_TOKEN=
export WANDB_API_KEY=28b3c634497c0dc6c16767729d4719b1012a94f2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir dataset/rl_bench_o1_dataset/2.0.0 \
  --dataset_name rlbencho1 \
  --run_root_dir logs \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set True \
  --val_freq 500 \
  --save_freq 500 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "mahlerrrr76" \
  --wandb_project "embodied_o1" \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state--film \
  --debug False