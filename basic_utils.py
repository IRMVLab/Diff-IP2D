# Developed by Junyi Ma
# Diff-IP2D: Diffusion-Based Hand-Object Interaction Prediction on Egocentric Videos
# https://github.com/IRMVLab/Diff-IP2D
# We thank OCT (Liu et al.), Diffuseq (Gong et al.), and USST (Bao et al.) for providing the codebases.

from diffip2d import gaussian_diffusion as gd
from diffip2d.gaussian_diffusion import HOIDiffusion, space_timesteps
from diffip2d.transformer_model import TransformerNetModel, MADT, RL_HOITransformerNetModel
from diffip2d.pre_encoder import SideFusionEncoder, MotionEncoder
from diffip2d.post_decoder import TrajDecoder

def create_network_and_diffusion(
    hidden_t_dim,
    hidden_dim,
    vocab_size,
    config_name,
    use_plm_init,
    dropout,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    sf_encoder_hidden,
    traj_decoder_hidden1,
    traj_decoder_hidden2,
    motion_encoder_hidden,
    madt_depth,
    feat_num=3,   # global hand object
    traj_dim=2,   # 2D traj on egocentric video 
    homo_dim=3,   # homography matrix
    **kwargs,
):

    # we will support more input params for different structures
    sf_encoder =  SideFusionEncoder(input_dims=feat_num * hidden_dim, output_dims=hidden_dim, encoder_hidden_dims=sf_encoder_hidden)
    traj_decoder =  TrajDecoder(input_dims=hidden_dim, output_dims=traj_dim, encoder_hidden_dims1=traj_decoder_hidden1, encoder_hidden_dims2=traj_decoder_hidden2)
    motion_encoder =  MotionEncoder(input_dims=homo_dim * homo_dim, output_dims=hidden_dim, encoder_hidden_dims=motion_encoder_hidden)
    denoised_model = MADT(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim*2),
        hidden_t_dim=hidden_t_dim,
        dropout=dropout,
        depth=madt_depth,
    )

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    diffusion = HOIDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return sf_encoder, denoised_model, diffusion, traj_decoder, motion_encoder