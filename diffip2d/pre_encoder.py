# Developed by Junyi Ma
# Diff-IP2D: Diffusion-Based Hand-Object Interaction Prediction on Egocentric Videos
# https://github.com/IRMVLab/Diff-IP2D
# We thank OCT (Liu et al.), Diffuseq (Gong et al.), and USST (Bao et al.) for providing the codebases.

from transformers import AutoConfig
import torch

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)

class SideFusionEncoder(nn.Module):
    """
    transform 5 channel to 1 channel
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        encoder_hidden_dims,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_t_dim = encoder_hidden_dims
        self.output_dims = output_dims

        self.feat_embed = nn.Sequential(
            # linear(input_dims, encoder_hidden_dims),
            # SiLU(),
            # linear(encoder_hidden_dims, output_dims), 
            linear(input_dims, output_dims), 
        )

    def forward(self, x):

        B = x.shape[0]
        T = x.shape[1]
        F = x.shape[2]
        x = x.view(B*T, F)
        x = self.feat_embed(x) 
        x = x.view(B, T, x.shape[-1])

        return x

class MotionEncoder(nn.Module):
    """
    transform 9 channel to 512 channel
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        encoder_hidden_dims,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_t_dim = encoder_hidden_dims
        self.output_dims = output_dims

        self.feat_embed = nn.Sequential(
            linear(input_dims, output_dims), 
        )

    def forward(self, x):

        B = x.shape[0]
        T = x.shape[1]
        F = x.shape[2]
        x = x.view(B*T, F)
        x = self.feat_embed(x) 
        x = x.view(B, T, x.shape[-1])

        return x