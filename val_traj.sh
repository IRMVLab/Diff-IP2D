TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=2 --master_port=12263 --use_env run_val_traj.py \
