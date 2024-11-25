# Developed by Junyi Ma
# Diff-IP2D: Diffusion-Based Hand-Object Interaction Prediction on Egocentric Videos
# https://github.com/IRMVLab/Diff-IP2D
# We thank OCT (Liu et al.), Diffuseq (Gong et al.), and USST (Bao et al.) for providing the codebases.


import argparse
import os
import datetime
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
from netscripts.get_datasets import get_dataset
from netscripts.get_network import get_network_for_diffip
from netscripts.get_optimizer import get_optimizer
from netscripts import modelio
from options import netsopts, expopts
from datasets.datasetopts import DatasetArgs
from diffip2d.step_sample import create_named_schedule_sampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from netscripts.epoch_feat import TrainValLoop
from basic_utils import create_network_and_diffusion
import logging.config
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
from diffip2d.utils import dist_util, logger


def main(args):

    # Initialization
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    dist_util.setup_dist()

    datasetargs = DatasetArgs(ek_version=args.ek_version)
    num_frames_input = int(datasetargs.fps * datasetargs.t_buffer)
    num_frames_output = int(datasetargs.fps * datasetargs.t_ant)
    start_epoch = 0

    # building architecture
    model_hoi, obj_head = get_network_for_diffip(args, num_frames_input=num_frames_input,
                        num_frames_output=num_frames_output)

    model_diff_args = {
        "hidden_t_dim": args.hidden_dim,
        "hidden_dim": args.hidden_dim,
        "vocab_size": None, # deprecated in non-nlp task
        "config_name": "huggingface-config",  # deprecated in non-nlp task
        "use_plm_init": "no",
        "dropout": args.dropout,
        "diffusion_steps": args.diffusion_steps,
        "noise_schedule": args.noise_schedule,
        "learn_sigma": args.learn_sigma,
        "timestep_respacing": args.timestep_respacing,
        "predict_xstart": args.predict_xstart,
        "rescale_timesteps": args.rescale_timesteps,
        "sigma_small": args.sigma_small,
        "rescale_learned_sigmas": args.rescale_learned_sigmas,
        "use_kl": args.use_kl,
        "sf_encoder_hidden": args.sf_encoder_hidden,
        "traj_decoder_hidden1": args.traj_decoder_hidden1,
        "traj_decoder_hidden2": args.traj_decoder_hidden2,
        "motion_encoder_hidden": args.motion_encoder_hidden,
        "madt_depth": args.madt_depth,
    }
    if int(os.environ['LOCAL_RANK']) == 0:
        logging.info("diffusion setups\n================= \n%s \n=================", model_diff_args)
    sf_encoder, model_denoise, diffusion, traj_decoder, motion_encoder = create_network_and_diffusion(**model_diff_args)
    if int(os.environ['LOCAL_RANK']) == 0:
        logging.info("finish building diffusion model!")

    schedule_sampler_args = args.schedule_sampler_args
    schedule_sampler = create_named_schedule_sampler(schedule_sampler_args, diffusion)
    if int(os.environ['LOCAL_RANK']) == 0:
        logging.info("finish building schedule sampler!")

    _, dls = get_dataset(args, base_path="./")

    if args.evaluate:
        args.epochs = start_epoch + 1
        traj_val_loader = None
        optimizer=None
        scheduler=None
    else:
        train_loader = dls['train']
        traj_val_loader = dls['validation']
        print("training dataset size: {}".format(len(train_loader.dataset)))
        optimizer, scheduler = get_optimizer(args, sf_encoder=sf_encoder, model_denoise=model_denoise,traj_decoder=traj_decoder,
                                              train_loader=train_loader,model_hoi=model_hoi, motion_encoder=motion_encoder, obj_head=obj_head)
                                              
    # We follow data structure of OCT to train and test our models 
    if not args.traj_only:
        val_loader = dls['eval']
    else:
        traj_val_loader = val_loader = dls['validation']
    print("evaluation dataset size: {}".format(len(val_loader.dataset)))
    
    if args.evaluate and args.traj_only:
        loader = traj_val_loader
    elif args.evaluate and (not args.traj_only):
        loader = val_loader
    else:
        loader = train_loader

    TrainValLoop(
            epochs = args.epochs,
            loader=loader,
            evaluate=args.evaluate,
            optimizer=optimizer,
            use_schedule=args.use_schedule,
            scheduler=scheduler,
            model_hoi=model_hoi,
            obj_head=obj_head,
            sf_encoder=sf_encoder,
            model_denoise=model_denoise,
            diffusion=diffusion,
            diffusion_steps=args.diffusion_steps,
            traj_decoder=traj_decoder,
            motion_encoder=motion_encoder,
            holi_past=args.holi_past,
            fast_test=args.fast_test,
            seq_len_obs=args.seq_len_obs,
            seq_len_unobs=args.seq_len_unobs,
            feat_dim=args.hidden_dim,
            sample_times=args.sample_times,
            learnable_weight=args.learnable_weight,
            reg_loss_weight=args.reg_loss_weight,
            rec_loss_weight=args.rec_loss_weight,
            schedule_sampler=schedule_sampler,
            test_start_idx=args.test_start_idx,
            resume=args.resume,
            base_model=args.base_model,
            log_path=args.log_path,
            checkpoint_path=args.checkpoint_path,
            collection_path_traj=args.collection_path_traj,
            collection_path_aff=args.collection_path_aff,
    ).run_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOI Forecasting")
    netsopts.add_nets_opts(parser)
    netsopts.add_train_opts(parser)
    expopts.add_exp_opts(parser)
    expopts.add_path_opts(parser)
    args = parser.parse_args()

    if args.use_cuda and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        args.batch_size = args.batch_size * num_gpus
        if int(os.environ['LOCAL_RANK']) == 0:
            logging.info("use batch size: %s", args.batch_size)

    if args.traj_only: assert args.evaluate, "evaluate trajectory on validation set must set --evaluate"
    main(args)
    gpu_id = os.environ['LOCAL_RANK']
    logging.info("GPU: %s Done!", gpu_id)