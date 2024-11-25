# Developed by Junyi Ma
# Diff-IP2D: Diffusion-Based Hand-Object Interaction Prediction on Egocentric Videos
# https://github.com/IRMVLab/Diff-IP2D
# We thank OCT (Liu et al.), Diffuseq (Gong et al.), and USST (Bao et al.) for providing the codebases.

import torch
from networks.traj_decoder import TrajCVAE
from networks.affordance_decoder import AffordanceCVAE
from networks.transformer import ObjectTransformer,ObjectTransformerModel
from networks.model import Model

def get_network_for_diffip(args, num_frames_input=10, num_frames_output=4):
    net =  ObjectTransformerModel(src_in_features=args.src_in_features,
                            trg_in_features=args.trg_in_features,
                            num_patches=args.num_patches,
                            encoder_time_embed_type=args.encoder_time_embed_type,
                            decoder_time_embed_type=args.decoder_time_embed_type,
                            num_frames_input=num_frames_input,
                            num_frames_output=num_frames_output,
                            embed_dim=args.embed_dim, coord_dim=args.coord_dim,
                            num_heads=args.num_heads, enc_depth=args.enc_depth, dec_depth=args.dec_depth)
    
    obj_head = AffordanceCVAE(in_dim=2, hidden_dim=512, latent_dim=256, condition_dim=512)
    return net, obj_head


def get_network(args, num_frames_input=10, num_frames_output=4):
    hand_head = TrajCVAE(in_dim=2, hidden_dim=args.hidden_dim,
                         latent_dim=args.latent_dim, condition_dim=args.embed_dim,
                         coord_dim=args.coord_dim)
    obj_head = AffordanceCVAE(in_dim=2, hidden_dim=args.hidden_dim,
                              latent_dim=args.latent_dim, condition_dim=args.embed_dim)
    net = ObjectTransformer(src_in_features=args.src_in_features,
                            trg_in_features=args.trg_in_features,
                            num_patches=args.num_patches,
                            hand_head=hand_head, obj_head=obj_head,
                            encoder_time_embed_type=args.encoder_time_embed_type,
                            decoder_time_embed_type=args.decoder_time_embed_type,
                            num_frames_input=num_frames_input,
                            num_frames_output=num_frames_output,
                            embed_dim=args.embed_dim, coord_dim=args.coord_dim,
                            num_heads=args.num_heads, enc_depth=args.enc_depth, dec_depth=args.dec_depth)
    net = torch.nn.DataParallel(net)

    model = Model(net, lambda_obj=args.lambda_obj, lambda_traj=args.lambda_traj,
                  lambda_obj_kl=args.lambda_obj_kl, lambda_traj_kl=args.lambda_traj_kl)
    return model

