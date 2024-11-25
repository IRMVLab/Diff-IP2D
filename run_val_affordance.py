# Developed by Junyi Ma
# Diff-IP2D: Diffusion-Based Hand-Object Interaction Prediction on Egocentric Videos
# https://github.com/IRMVLab/Diff-IP2D
# We thank OCT (Liu et al.), Diffuseq (Gong et al.), and USST (Bao et al.) for providing the codebases.

import sys
import os
sys.path.append('.')

if __name__ == '__main__':


    COMMANDLINE = f"python traineval.py --evaluate --ek_version=ek100 --resume=./diffip_weights/checkpoint_aff.pth.tar" \

    print(COMMANDLINE)
    os.system(COMMANDLINE)