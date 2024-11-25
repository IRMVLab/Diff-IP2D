# Developed by Junyi Ma
# Diff-IP2D: Diffusion-Based Hand-Object Interaction Prediction on Egocentric Videos
# https://github.com/IRMVLab/Diff-IP2D
# We thank OCT (Liu et al.), Diffuseq (Gong et al.), and USST (Bao et al.) for providing the codebases.

import sys
import os
import argparse
import time
sys.path.append('.')

if __name__ == '__main__':

    # TODO add more options
    COMMANDLINE = f"python traineval.py  --ek_version=ek100" \

    print(COMMANDLINE)
    os.system(COMMANDLINE)