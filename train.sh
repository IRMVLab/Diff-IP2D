TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=2 --master_port=12224 --use_env run_train.py \
