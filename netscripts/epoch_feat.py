# Developed by Junyi Ma
# Diff-IP2D: Diffusion-Based Hand-Object Interaction Prediction on Egocentric Videos
# https://github.com/IRMVLab/Diff-IP2D
# We thank OCT (Liu et al.), Diffuseq (Gong et al.), and USST (Bao et al.) for providing the codebases.

import time
import numpy as np
import torch
import os
from tqdm import trange
import functools
import logging.config
import datetime
from functools import partial
from diffip2d.rounding import denoised_fn_round
from diffip2d.step_sample import LossAwareSampler, UniformSampler
from netscripts import modelio
from netscripts.epoch_utils import progress_bar as bar, AverageMeters
from evaluation.traj_eval import evaluate_traj_stochastic
from evaluation.affordance_eval import evaluate_affordance
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from diffip2d.utils import dist_util, logger

def traj_affordance_dist(hand_traj, contact_point, future_valid=None, invalid_value=9):
    batch_size = contact_point.shape[0]
    expand_size = int(hand_traj.shape[0] / batch_size)
    contact_point = contact_point.unsqueeze(dim=1).expand(-1, expand_size, 2).reshape(-1, 2)
    dist = torch.sum((hand_traj - contact_point) ** 2, dim=1).reshape(batch_size, -1)
    if future_valid is None:
        sorted_dist, sorted_idx = torch.sort(dist, dim=-1, descending=False)
        return sorted_dist[:, 0]
    else:
        dist = dist.reshape(batch_size, 2, -1)
        future_valid = future_valid > 0
        future_invalid = ~future_valid[:, :, None].expand(dist.shape)
        dist[future_invalid] = invalid_value
        sorted_dist, sorted_idx = torch.sort(dist, dim=-1, descending=False)
        selected_dist = sorted_dist[:, :, 0]
        selected_dist, selected_idx = selected_dist.min(dim=1)
        valid = torch.gather(future_valid, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
        selected_dist = selected_dist * valid
    return selected_dist

def compute_fde(pred_traj, gt_traj, valid_traj=None, reduction=True):
    pred_last = pred_traj[:, :, -1, :]
    gt_last = gt_traj[:, :, -1, :]

    valid_loc = (gt_last[:, :, 0] >= 0) & (gt_last[:, :, 1] >= 0) \
                & (gt_last[:, :, 0] < 1) & (gt_last[:, :, 1] < 1)

    error = gt_last - pred_last
    error = error * valid_loc[:, :, None]

    if torch.is_tensor(error):
        if valid_traj is None:
            valid_traj = torch.ones(pred_traj.shape[0], pred_traj.shape[1])
        error = error ** 2
        fde = torch.sqrt(error.sum(dim=2)) * valid_traj
        if reduction:
            fde = fde.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()
    else:
        if valid_traj is None:
            valid_traj = np.ones((pred_traj.shape[0], pred_traj.shape[1]), dtype=int)
        error = np.linalg.norm(error, axis=2)
        fde = error * valid_traj
        if reduction:
            fde = fde.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()

    return fde, valid_traj


def compute_wde(pred_traj, gt_traj, valid_traj=None, seq_len_unobs=4, reduction=True):
    valid_loc = (gt_traj[:, :, :, 0] >= 0) & (gt_traj[:, :, :, 1] >= 0)  \
                 & (gt_traj[:, :, :, 0] < 1) & (gt_traj[:, :, :, 1] < 1)

    error = gt_traj - pred_traj
    error = error * valid_loc[:, :, :, None]

    if torch.is_tensor(error):
        if valid_traj is None:
            valid_traj = torch.ones(pred_traj.shape[0], pred_traj.shape[1])
        error = error ** 2
        ade = torch.sqrt(error.sum(dim=3)).mean(dim=2) * valid_traj
        if reduction:
            ade = ade.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()
    else:
        if valid_traj is None:
            valid_traj = np.ones((pred_traj.shape[0], pred_traj.shape[1]), dtype=int)
        error = np.linalg.norm(error, axis=3)
        ade = 0.0
        for i in range(seq_len_unobs):
            ade += ((i+1)/seq_len_unobs)*error[:,:,i] * valid_traj
        if reduction:
            ade = ade.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()

    return ade, valid_traj

class TrainValLoop:
    def __init__(
            self,
            start_epoch = 0,
            epochs = 25,
            loader=None,
            evaluate=False,
            use_schedule=False,
            scheduler=None,
            optimizer=None,
            model_hoi=None,
            obj_head=None,
            sf_encoder=None,
            motion_encoder=None,
            model_denoise=None,
            diffusion=None,
            diffusion_steps=1000,
            traj_decoder=None,
            base_model=None,
            holi_past=True,
            seq_len_obs=10,
            seq_len_unobs=4,
            feat_dim=512,
            sample_times=10,
            learnable_weight=True,
            reg_loss_weight=0.2,
            rec_loss_weight=1.0,
            schedule_sampler=None,
            resume=None,
            fast_test=True,
            log_path=None,
            checkpoint_path=None,
            collection_path_traj=None,
            collection_path_aff=None,
            refine_const=1,
            test_start_idx=0,
    ):
        self.sf_encoder = DDP(
            sf_encoder.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.model_denoise = DDP(
                    model_denoise.to(dist_util.dev()) ,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=True,
                )

        self.traj_decoder = DDP(
                    traj_decoder.to(dist_util.dev()) ,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=True,
                )
        
        self.model_hoi = DDP(
            model_hoi.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.obj_head = DDP(
            obj_head.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.motion_encoder = DDP(
            motion_encoder.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.diffusion = diffusion
        self.start_epoch = start_epoch
        self.all_epochs = epochs
        self.evaluate = evaluate
        self.optimizer = optimizer
        self.loader = loader
        self.scheduler = scheduler
        self.schedule_sampler = schedule_sampler
        self.seq_len_obs = seq_len_obs
        self.seq_len_unobs = seq_len_unobs
        self.seq_len_total = seq_len_obs + seq_len_unobs
        self.sample_times = sample_times
        if not learnable_weight:
            self.reg_loss_weight = reg_loss_weight
            self.rec_loss_weight = rec_loss_weight
        else:
            self.reg_loss_weight = torch.nn.Parameter(torch.tensor(reg_loss_weight, device=dist_util.dev(), dtype=torch.float32))
            self.rec_loss_weight = torch.nn.Parameter(torch.tensor(rec_loss_weight, device=dist_util.dev(), dtype=torch.float32))
        self.feat_dim = feat_dim
        self.diffusion_steps = diffusion_steps
        self.refine_diffusion_steps = diffusion_steps-refine_const
        self.use_schedule = use_schedule
        self.holi_past = holi_past
        self.fast_test = fast_test
        self.test_start_idx = test_start_idx

        self.gts_affordance_dict, self.preds_affordance_dict = {}, {}
        self.ade_list = []
        self.fde_list = []

        if resume is not None:
            # pre_encoder_state_dict to sf_encoder_state_dict for our pretrained model
            self.start_epoch = modelio.load_checkpoint_by_name(self.sf_encoder, resume_path=resume[0], state_dict_name="pre_encoder_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.model_denoise, resume_path=resume[0], state_dict_name="model_denoise_state_dict", strict=False, device=dist_util.dev())
            # post_encoder_state_dict to traj_decoder_state_dict for our pretrained model
            self.start_epoch = modelio.load_checkpoint_by_name(self.traj_decoder, resume_path=resume[0], state_dict_name="post_encoder_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.model_hoi, resume_path=resume[0], state_dict_name="model_hoi_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.motion_encoder, resume_path=resume[0], state_dict_name="motion_encoder_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.obj_head, resume_path=resume[0], state_dict_name="obj_head_state_dict", strict=False, device=dist_util.dev())
            print("finish loading diffusion model from epoch {}".format(self.start_epoch))


        dist_util.sync_params(self.sf_encoder.parameters())
        dist_util.sync_params(self.model_denoise.parameters())
        dist_util.sync_params(self.traj_decoder.parameters())
        dist_util.sync_params(self.model_hoi.parameters())
        dist_util.sync_params(self.motion_encoder.parameters())
        dist_util.sync_params(self.obj_head.parameters())

        self.all_epochs, self.start_epoch = (1, 0) if self.evaluate else (self.all_epochs, self.start_epoch)

        self.logger = logging.getLogger('main')
        self.logger.setLevel(level=logging.DEBUG)
        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        if evaluate:
            handler = logging.FileHandler(log_path +'/' + f"eval_{time_str}.log")
        else:
            handler = logging.FileHandler(log_path +'/' + f"{time_str}.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.checkpoint_path = checkpoint_path
        self.collection_path_traj = collection_path_traj
        self.collection_path_aff = collection_path_aff

    def run_loop(self):
            if not self.evaluate: 
                self.model_hoi.train()
                self.sf_encoder.train()
                self.model_denoise.train()
                self.traj_decoder.train()
                self.motion_encoder.train()
                self.obj_head.train()
            else:
                self.model_hoi.eval()
                self.sf_encoder.eval()
                self.model_denoise.eval()
                self.traj_decoder.eval()
                self.motion_encoder.eval()
                self.obj_head.eval()

            for epoch in range(self.start_epoch, self.all_epochs):
                if not self.evaluate:
                    self.logger.info("Using lr {}".format(self.optimizer.param_groups[0]["lr"]))
                    self.epoch_pass(
                        phase='train',
                        epoch=epoch,
                        train=True,)
                else:
                    self.logger.info("Start to test!")
                    self.epoch_pass(
                        phase='traj',
                        epoch=epoch,
                        train=False,)

    def epoch_pass(self, epoch, phase, train=True):
        time_meters = AverageMeters()

        if train:
            self.loader.sampler.set_epoch(epoch)
            self.logger.info(f"{phase} epoch: {epoch + 1}")
            loss_meters = AverageMeters()
        else:
            self.loader.sampler.set_epoch(epoch)
            preds_traj, gts_traj, valids_traj = [], [], []
            gts_affordance_dict, preds_affordance_dict = {}, {}

        end = time.time()
        loss_all = []
        for batch_idx, sample in enumerate(self.loader):

            if train:
                self.optimizer.zero_grad()

                input = sample['feat'].float().to(dist_util.dev())
                bbox_feat = sample['bbox_feat'].float().to(dist_util.dev())
                valid_mask = sample['valid_mask'].float().to(dist_util.dev())
                future_hands = sample['future_hands'].float().to(dist_util.dev())
                contact_point = sample['contact_point'].float().to(dist_util.dev())
                future_valid = sample['future_valid'].float().to(dist_util.dev())
                time_meters.add_loss_value("data_time", time.time() - end)

                homo_transform = sample['homography_stack'].float().to(dist_util.dev())
                homo_transform = homo_transform[:, :self.seq_len_obs, ...].contiguous()
                homo_transform = homo_transform.view(homo_transform.shape[0], self.seq_len_obs, 3*3)
                motion_feat_encoded = self.motion_encoder(homo_transform)
                grl_feat = self.model_hoi(input, bbox_feat, valid_mask)

                grl_feat_past = grl_feat[:, 0:self.seq_len_obs,...]
                grl_feat_future = grl_feat[:, int(self.seq_len_obs+1):,...]
                valid_mask_past = valid_mask[:, :, 0:self.seq_len_obs]
                valid_mask_future = valid_mask[:, :, int(self.seq_len_obs+1):]
                grl_feat = torch.cat((grl_feat_past, grl_feat_future), dim=1)
                valid_mask = torch.cat((valid_mask_past, valid_mask_future), dim=-1)
                
                right_feat = torch.cat((grl_feat[:,:,0:1,:], grl_feat[:,:,1:2,:], grl_feat[:,:,3:4,:]), dim=2)
                left_feat = torch.cat((grl_feat[:,:,0:1,:], grl_feat[:,:,2:3,:], grl_feat[:,:,4:5,:]), dim=2)
                valid_mask_r = valid_mask[:,1:2,:]
                valid_mask_l = valid_mask[:,2:3,:]
                right_feat = right_feat.view(*right_feat.shape[0:2], -1)
                left_feat = left_feat.view(*right_feat.shape[0:2], -1)

                right_feat_encoded = self.sf_encoder(right_feat)
                left_feat_encoded = self.sf_encoder(left_feat)

                t, weights = self.schedule_sampler.sample(grl_feat.shape[0], dist_util.dev())

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model_denoise,
                    self.traj_decoder,
                    [right_feat_encoded, left_feat_encoded],
                    t,
                    [None,None],
                    motion_feat_encoded,
                )

                loss_feat_dict = compute_losses()
                loss_feat_level = loss_feat_dict["loss_feat_level"]
                rec_feature_r = loss_feat_dict["rec_feature_r"]
                rec_feature_l = loss_feat_dict["rec_feature_l"]

                future_feature = torch.cat((rec_feature_r[:,-1*self.seq_len_unobs:, ...], rec_feature_l[:,-1*self.seq_len_unobs:, ...]), dim=1)
                pred_future_traj = self.traj_decoder(future_feature)
                pred_future_traj_r = pred_future_traj[:, :self.seq_len_unobs, :]
                pred_future_traj_l = pred_future_traj[:, self.seq_len_unobs:, :]

                vanilla_future_traj_r = self.traj_decoder(right_feat_encoded[:,-(self.seq_len_unobs+1):,:].contiguous())
                vanilla_future_traj_l = self.traj_decoder(left_feat_encoded[:,-(self.seq_len_unobs+1):,:].contiguous())
                vanilla_r_loss = torch.sum((vanilla_future_traj_r - future_hands[:, 0, 0:, :]) ** 2, dim=-1)
                vanilla_r_loss = vanilla_r_loss.sum(-1) * self.reg_loss_weight
                vanilla_l_loss = torch.sum((vanilla_future_traj_l - future_hands[:, 1, 0:, :]) ** 2, dim=-1)
                vanilla_l_loss = vanilla_l_loss.sum(-1) * self.reg_loss_weight

                future_traj_r_loss = torch.sum((pred_future_traj_r - future_hands[:, 0, 1:, :]) ** 2, dim=-1)
                future_traj_r_loss = future_traj_r_loss.sum(-1) * self.rec_loss_weight
                future_traj_l_loss = torch.sum((pred_future_traj_l - future_hands[:, 1, 1:, :]) ** 2, dim=-1)
                future_traj_l_loss = future_traj_l_loss.sum(-1) * self.rec_loss_weight

                losses_r =  (future_traj_r_loss + vanilla_r_loss) * future_valid[:,0]
                losses_l =  (future_traj_l_loss + vanilla_l_loss) * future_valid[:,1]

                losses_two_hand = losses_r + losses_l

                future_rhand, future_lhand = future_hands[:, 0, :, :], future_hands[:, 1, :, :]
                if self.holi_past:
                    r_pred_contact, r_obj_loss, r_obj_kl_loss = self.obj_head(torch.mean(rec_feature_r[:, :, ...], dim=1, keepdim=False), contact_point, future_rhand, return_pred=True)
                    l_pred_contact, l_obj_loss, l_obj_kl_loss = self.obj_head(torch.mean(rec_feature_l[:, :, ...], dim=1, keepdim=False), contact_point, future_lhand, return_pred=True)
                else:
                    r_pred_contact, r_obj_loss, r_obj_kl_loss = self.obj_head(torch.mean(rec_feature_r[:, self.seq_len_obs-1:self.seq_len_obs, ...], dim=1, keepdim=False), contact_point, future_rhand, return_pred=True)
                    l_pred_contact, l_obj_loss, l_obj_kl_loss = self.obj_head(torch.mean(rec_feature_l[:, self.seq_len_obs-1:self.seq_len_obs, ...], dim=1, keepdim=False), contact_point, future_lhand, return_pred=True)

                obj_loss = torch.stack([r_obj_loss, l_obj_loss], dim=1)
                obj_kl_loss = torch.stack([r_obj_kl_loss, l_obj_kl_loss], dim=1)
                obj_loss[~(future_valid > 0)] = 1e9
                selected_obj_loss, selected_idx = obj_loss.min(dim=1)
                selected_valid = torch.gather(future_valid, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
                selected_obj_kl_loss = torch.gather(obj_kl_loss, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
                obj_loss = selected_obj_loss * selected_valid
                obj_kl_loss = selected_obj_kl_loss * selected_valid
                lambda_obj = 0.1
                lambda_obj_kl = 1e-3
                losses_obj = lambda_obj*obj_loss + lambda_obj_kl*obj_kl_loss
                
                losses_two_sides = losses_two_hand + losses_obj

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses_two_sides.detach()
                    )

                # loss-aware weights
                loss = (losses_two_sides * weights).mean()

                model_losses = {
                        "future_traj_r_loss":future_traj_r_loss[losses_r!=0].mean(),
                        "future_traj_l_loss":future_traj_l_loss[losses_l!=0].mean(),
                        "rec_r_loss":loss_feat_level['mse_r'][losses_r!=0].mean(),
                        "rec_l_loss":loss_feat_level['mse_l'][losses_l!=0].mean(),
                        "total_loss":loss,
                        "future_valid_r": future_valid[:,0].sum(),
                        "future_valid_l": future_valid[:,1].sum(),
                        "obj_loss": obj_loss.mean(),
                }

                loss.backward()
                self.optimizer.step()

                for key, val in model_losses.items():
                    if val is not None:
                        loss_meters.add_loss_value(key, val.detach().cpu().item())

                time_meters.add_loss_value("batch_time", time.time() - end)

                if dist_util.get_rank() == 0:
                    self.logger.info(loss_meters.average_meters["total_loss"].avg)

                    suffix = "Epoch:{epoch} " \
                            "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s " \
                            "| future_traj_r_loss: {future_traj_r_loss:.3f} " \
                            "| future_traj_l_loss: {future_traj_l_loss:.3f} " \
                            "| rec_r_loss: {rec_r_loss:.3f}" \
                            "| rec_l_loss: {rec_l_loss:.3f}" \
                            "| total_loss: {total_loss:.3f} " \
                            "| future_valid_r: {future_valid_r:.1f} "\
                            "| future_valid_l: {future_valid_l:.1f} "\
                            "| obj_loss: {obj_loss:.3f} ".format(
                                epoch=epoch, batch=batch_idx + 1, size=len(self.loader),
                                data=time_meters.average_meters["data_time"].val,
                                bt=time_meters.average_meters["batch_time"].avg,
                                future_traj_r_loss=model_losses["future_traj_r_loss"],
                                future_traj_l_loss=model_losses["future_traj_l_loss"],
                                rec_r_loss=model_losses["rec_r_loss"],
                                rec_l_loss=model_losses["rec_l_loss"],
                                total_loss=model_losses["total_loss"],
                                future_valid_r=model_losses["future_valid_r"],
                                future_valid_l=model_losses["future_valid_l"],
                                obj_loss=model_losses["obj_loss"],
                                                                    )
                    self.logger.info(suffix)
                    bar(suffix)

                end = time.time()
                if self.scheduler is not None and self.use_schedule:
                    self.scheduler.step()

            else:
                input = sample['feat'].float().to(dist_util.dev())
                bbox_feat = sample['bbox_feat'].float().to(dist_util.dev())
                valid_mask = sample['valid_mask'].float().to(dist_util.dev())
                future_valid = sample['future_valid'].float().to(dist_util.dev())
                homo_transform = sample['homography_stack'].float().to(dist_util.dev())

                time_meters.add_loss_value("data_time", time.time() - end)

                sample_fn = (
                    self.diffusion.p_sample_loop
                )

                with torch.no_grad():
                    homo_transform = homo_transform.view(homo_transform.shape[0], self.seq_len_obs, 3*3)
                    motion_feat_encoded = self.motion_encoder(homo_transform)
                    
                    grl_feat = self.model_hoi(input, bbox_feat, valid_mask)
                    bsize = grl_feat.shape[0]
                    grl_feat_past = grl_feat[:, 0:self.seq_len_obs,...]
                    grl_feat = grl_feat_past
                    
                    right_feat = torch.cat((grl_feat[:,:,0:1,:], grl_feat[:,:,1:2,:], grl_feat[:,:,3:4,:]), dim=2)
                    left_feat = torch.cat((grl_feat[:,:,0:1,:], grl_feat[:,:,2:3,:], grl_feat[:,:,4:5,:]), dim=2)

                    valid_mask_r = valid_mask[:,1:2,:]
                    valid_mask_l = valid_mask[:,2:3,:]
                    valid_mask_r = torch.cat((valid_mask_r, torch.ones((valid_mask_r.shape[0], valid_mask_r.shape[1], self.seq_len_unobs)).to(dist_util.dev())), dim=-1)
                    valid_mask_l = torch.cat((valid_mask_l, torch.ones((valid_mask_l.shape[0], valid_mask_l.shape[1], self.seq_len_unobs)).to(dist_util.dev())), dim=-1)

                    right_feat = right_feat.view(*right_feat.shape[0:2], -1)
                    left_feat = left_feat.view(*right_feat.shape[0:2], -1)
                    right_feat_encoded = self.sf_encoder(right_feat)
                    left_feat_encoded = self.sf_encoder(left_feat)

                    for sample_idx in range(self.sample_times):
                        pseudo_future = torch.zeros((bsize, self.seq_len_unobs, self.feat_dim))
                        noise_r = torch.randn_like(pseudo_future).to(dist_util.dev())
                        noise_l = torch.randn_like(pseudo_future).to(dist_util.dev())
                        x_noised_r = torch.cat((right_feat_encoded, noise_r), dim=1)
                        x_noised_l = torch.cat((left_feat_encoded, noise_l), dim=1)

                        future_mask = torch.zeros((bsize, self.seq_len_total, self.feat_dim))
                        future_mask[:,int(-1*self.seq_len_unobs):,:] = 1

                        sample_shape = (x_noised_r.shape[0], x_noised_r.shape[1], x_noised_r.shape[2])

                        model_kwargs = {}

                        if int(os.environ['LOCAL_RANK']) == 0:
                            print("========== sample id: "+str(sample_idx) + "/" +str(self.sample_times) +" denoising ==========")

                        samples_r, samples_l = sample_fn(
                            model_denoise=self.model_denoise,
                            shape=sample_shape,
                            noise=[x_noised_r[:,self.test_start_idx:,...], x_noised_l[:,self.test_start_idx:,...]],
                            motion_feat_encoded=motion_feat_encoded[:,self.test_start_idx:,...],
                            clip_denoised=False,
                            model_kwargs=model_kwargs,
                            clamp_step=0,
                            clamp_first=True,
                            mask=future_mask,
                            x_start=[right_feat_encoded, left_feat_encoded],
                            gap=self.refine_diffusion_steps if self.fast_test else 1,
                            device=dist_util.dev(),
                            valid_mask = [None,None],
                        )


                        samples_r = samples_r[-1]
                        samples_l = samples_l[-1]

                        future_feature = torch.cat((samples_r[:,int(-1*self.seq_len_unobs):, ...], samples_l[:,int(-1*self.seq_len_unobs):, ...]), dim=1)
                        pred_future_traj = self.traj_decoder(future_feature)
                        pred_future_traj_r = pred_future_traj[:, :self.seq_len_unobs, :]
                        pred_future_traj_l = pred_future_traj[:, self.seq_len_unobs:, :]

                        future_hands = sample['future_hands'][:, :, 1:, :].float().numpy()
                        future_valid = sample['future_valid'].float().numpy()

                        pred_future_traj = torch.stack((pred_future_traj_r,pred_future_traj_l), dim=1).cpu().float().numpy()

                        if not 'eval' in self.loader.dataset.partition:
                            ade, _ = compute_wde(pred_future_traj, future_hands, valid_traj=future_valid, seq_len_unobs=self.seq_len_unobs, reduction=True)
                            self.ade_list.append(ade)

                            fde, _ = compute_fde(pred_future_traj, future_hands, future_valid, reduction=True)
                            self.fde_list.append(fde)

                            self.logger.info(str(batch_idx) + "/" + str(len(self.loader)) +  " ade ours "+str(ade))
                            self.logger.info(str(batch_idx) + "/" + str(len(self.loader)) +  " fde ours "+str(fde))

                    if 'eval' in self.loader.dataset.partition:

                        observe_bbox = bbox_feat[:, :2, int(self.seq_len_obs-1), :]
                        observe_rhand, observe_lhand = observe_bbox[:, 0, :], observe_bbox[:, 1, :]
                        future_rhand = (observe_rhand[:, :2] + observe_rhand[:, 2:]) / 2
                        future_lhand = (observe_lhand[:, :2] + observe_lhand[:, 2:]) / 2
                        future_rhand = future_rhand.unsqueeze(dim=1)
                        future_lhand = future_lhand.unsqueeze(dim=1)

                        future_rhand = torch.cat((future_rhand, pred_future_traj_r), dim=1)
                        future_lhand = torch.cat((future_lhand, pred_future_traj_l), dim=1)

                        future_valid = torch.from_numpy(future_valid).to(dist_util.dev())
                        future_hands =  torch.from_numpy(future_hands).to(dist_util.dev())

                        contact_points_list = []
                        for tidx in range(self.sample_times):
                            
                            if self.holi_past:
                                memory_r = torch.mean(samples_r[:, :, ...], dim=1, keepdim=False)
                                memory_l = torch.mean(samples_l[:, :, ...], dim=1, keepdim=False)
                            else:
                                memory_r = torch.mean(samples_r[:, self.seq_len_obs-1:self.seq_len_obs, ...], dim=1, keepdim=False)
                                memory_l = torch.mean(samples_l[:, self.seq_len_obs-1:self.seq_len_obs, ...], dim=1, keepdim=False)                                    
                            r_pred_contact = self.obj_head.module.inference(memory_r, future_rhand)
                            l_pred_contact = self.obj_head.module.inference(memory_l, future_lhand)

                            pred_contact = torch.stack([r_pred_contact, l_pred_contact], dim=1)

                            if future_valid is not None and torch.all(future_valid.sum(dim=1) >= 1):
                                r_pred_contact_dist = traj_affordance_dist(future_hands.reshape(-1, 2), r_pred_contact, future_valid)
                                l_pred_contact_dist = traj_affordance_dist(future_hands.reshape(-1, 2), l_pred_contact,
                                                                        future_valid)
                                pred_contact_dist = torch.stack((r_pred_contact_dist, l_pred_contact_dist), dim=1)
                                _, selected_idx = pred_contact_dist.min(dim=1)
                                selected_idx = selected_idx.unsqueeze(dim=1).unsqueeze(dim=2).expand(pred_contact.shape[0], 1, pred_contact.shape[2])
                                contact_points = torch.gather(pred_contact, dim=1, index=selected_idx).squeeze(dim=1)
                                contact_points_list.append(contact_points)

                        contact_points = torch.stack(contact_points_list, dim=1)
                        uids = sample['uid'].numpy()
                        contact_points = contact_points.cpu().numpy()
                        for idx, uid in enumerate(uids):
                            self.gts_affordance_dict[uid] = self.loader.dataset.eval_labels[uid]['norm_contacts']
                            self.preds_affordance_dict[uid] = contact_points[idx]

                  
        if train:
            if not os.path.exists(self.checkpoint_path):
                os.mkdir(self.checkpoint_path)
            warmup_epochs = 0
            snapshot = 1
            if (epoch + 1 - warmup_epochs) % snapshot == 0 and (dist_util.get_rank() == 0):
                print("save epoch "+str(epoch+1)+" checkpoint")
                modelio.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "sf_encoder_state_dict": self.sf_encoder.state_dict(),
                    "model_denoise_state_dict": self.model_denoise.state_dict(),
                    "traj_decoder_state_dict": self.traj_decoder.state_dict(),
                    "model_hoi_state_dict": self.model_hoi.state_dict(),
                    "motion_encoder_state_dict": self.motion_encoder.state_dict(),
                    "obj_head_state_dict": self.obj_head.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                checkpoint=self.checkpoint_path,
                filename = f"checkpoint_{epoch+1}.pth.tar")
                torch.save(self.optimizer.state_dict(), "optimizer.pt")

                return loss_meters
        else:

            num_gpus = torch.cuda.device_count()
            if int(os.environ['LOCAL_RANK']) == 0:
                print("we are using "+ str(num_gpus) + " GPUs for evaluation!")
            device_id_for_save = int(os.environ['LOCAL_RANK'])
            if not os.path.exists(self.collection_path_traj):
                os.mkdir(self.collection_path_traj)
            if not os.path.exists(self.collection_path_aff):
                os.mkdir(self.collection_path_aff)

            if not 'eval' in self.loader.dataset.partition:
                current_save_path = os.path.join(self.collection_path_traj, "ours_ade"+str(device_id_for_save))
                np.save(current_save_path, np.array(self.ade_list))

                current_save_path = os.path.join(self.collection_path_traj, "ours_fde"+str(device_id_for_save))
                np.save(current_save_path, np.array(self.fde_list))

                if int(os.environ['LOCAL_RANK']) == 0:
                    while 1:
                        if len(os.listdir(self.collection_path_traj))==num_gpus*2:
                            ade_mean = []
                            fde_mean = []
                            for i in range(num_gpus):
                                ade_mean = ade_mean + np.load(os.path.join(self.collection_path_traj, "ours_ade"+str(i) + ".npy")).tolist()
                                fde_mean = fde_mean + np.load(os.path.join(self.collection_path_traj, "ours_fde"+str(i) + ".npy")).tolist()                                
                            break
                        time.sleep(2)
                    self.logger.info("ours wde ---> "+str(np.array(ade_mean).mean()))
                    self.logger.info("ours fde ---> "+str(np.array(fde_mean).mean()))

            else:
                ours_affordance_metrics = evaluate_affordance(self.preds_affordance_dict,
                                                         self.gts_affordance_dict,
                                                         n_pts=5,
                                                         gaussian_sigma=3.,
                                                         gaussian_k_ratio=3.)

                ours_affordance_metrics_str = ' '.join([f"{key}: {value}" for key, value in ours_affordance_metrics.items()])
                for key, value in ours_affordance_metrics.items():
                    current_save_path = os.path.join(self.collection_path_aff, "ours_"+key+str(device_id_for_save))
                    np.save(current_save_path, np.array(value))
                if int(os.environ['LOCAL_RANK']) == 0:
                    while 1:
                        if len(os.listdir(self.collection_path_aff))==num_gpus * 3:
                            for key, value in ours_affordance_metrics.items():
                                aff_mean =0
                                for i in range(num_gpus):
                                    aff_mean = aff_mean + np.load(os.path.join(self.collection_path_aff, "ours_"+key+str(i) + ".npy"))
                                self.logger.info("ours "+key+" ---> "+str(aff_mean/num_gpus))
                            break
                        time.sleep(2)
                    