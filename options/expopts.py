def add_exp_opts(parser):
    parser.add_argument("--resume", type=str, nargs="+", metavar="PATH",
                        help="path to latest checkpoint (default: none)")
    parser.add_argument("--evaluate", dest="evaluate", action="store_true",
                        help="evaluate model on validation set")
    parser.add_argument("--test_freq", type=int, default=100,
                        help="testing frequency on evaluation dataset (set specific in traineval.py)")
    parser.add_argument("--snapshot", default=1, type=int, metavar="N",
                        help="How often to take a snapshot of the model (0 = never)")
    parser.add_argument("--use_cuda", default=1, type=int, help="use GPU (default: True)")
    parser.add_argument('--ek_version', default="ek55", choices=["ek55", "ek100"], help="epic dataset version")
    parser.add_argument("--traj_only", action="store_true", help="evaluate traj on validation dataset")
    parser.add_argument("--schedule_sampler_args", default="lossaware", choices=["uniform", "lossaware", "fixstep"], help="loss schedule for diffusion")
    parser.add_argument("--seq_len_obs", default=10, type=int, help="length of observed (past) sequence")
    parser.add_argument("--seq_len_unobs", default=4, type=int, help="length of unobserved (future) sequence")
    parser.add_argument("--learnable_weight", default=False, type=bool, help="whether to use learnable loss weights")
    parser.add_argument("--rec_loss_weight", default=1.0, type=float, help="initial value of diffusion losses")
    parser.add_argument("--reg_loss_weight", default=0.2, type=float, help="initial value of regularization loss")
    parser.add_argument("--use_schedule", default=False, type=bool, help="whether to specify optimizer schedule")
    parser.add_argument("--sample_times", default=10, type=int, help="how many samples for one prediction")
    parser.add_argument("--fast_test", default=True, type=bool, help="whether to use faster inference")

def add_path_opts(parser):
    parser.add_argument("--base_model", default="./base_models/model.pth.tar", type=str, nargs="+", metavar="PATH", help="path to base model")
    parser.add_argument("--log_path", default="./log", type=str, nargs="+", metavar="PATH", help="path to record logs")
    parser.add_argument("--checkpoint_path", default="./diffip_weights", type=str, nargs="+", metavar="PATH", help="path to save checkpoints")
    parser.add_argument("--collection_path_traj", default="./collected_pred_traj", type=str, nargs="+", metavar="PATH", help="path to gather traj eval results")
    parser.add_argument("--collection_path_aff", default="./collected_pred_aff", type=str, nargs="+", metavar="PATH", help="path to gather aff eval results")
