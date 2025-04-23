import argparse
import importlib
import json
import logging
import numpy as np
import os
import time
import torch
import Tools
import copy

from tester import eval_GT
from trainer import train

os.environ['CUDA_LAUNCH_BLOCKING']='1'

if torch.__version__[0] == '2':
    import torch._dynamo

def log_setting(cfg):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler() # Adding Terminal Logger

    # Adding File Logger
    log_dir = os.path.join(cfg['Rec']['dir'], cfg['Model']['name'], cfg['timestamp'])
    os.makedirs(log_dir, exist_ok=True)
    cfg['Rec']['dir'] = log_dir

    fh = logging.FileHandler(filename=os.path.join(log_dir, 'logger.txt'))
    fh.setFormatter(logging.Formatter("%(asctime)s  : %(message)s", "%b%d-%H:%M"))

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    logger.info(json.dumps(cfg, indent=4, separators=(',', ': ')))
    return logger

def init_seed(seed=1):
    import random
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enable = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--Model', type=str, default='DecoupleNet_test')
    args.add_argument('--Train_Dataset', type=str, default='DSEC')
    args.add_argument('--Test_Dataset', type=str, default='MVSEC')
    args.add_argument('--timestamp', type=str, default=None)
    args.add_argument('--refine_path', type=str, default=None)
    args.add_argument('--override', type=str, help='Arguments for overriding config')
    args = vars(args.parse_args())
    cfg = json.load(open(f"Config/{args['Model']}.json", 'r'))

    init_seed(2023)

    if args['override']:
        cfg = Tools.override(args['override'], cfg)

    if args['timestamp'] is None:
        cfg['timestamp'] = time.strftime('%m%d%H%M', time.localtime(time.time())) # Add timestamp with format mouth-day-hour-minute
    else:
        cfg['timestamp'] = args['timestamp']

    cfg['Data']['Train_Dataset'] = args['Train_Dataset']
    cfg['Data']['Test_Dataset'] = args['Test_Dataset']
    cfg['Model'].update(cfg['Data'])
    logger = log_setting(cfg)


    # ---- load model -----
    net = importlib.import_module(f"Models.{cfg['Model']['name']}").Model(**cfg['Model'])
    if args['refine_path'] is not None:
        net.load_state_dict(torch.load(args['refine_path']))
    if torch.__version__[0] == '2':
        torch._dynamo.config.suppress_errors = True
        net = torch.compile(net)

    params = np.sum([p.numel() for p in net.parameters()]).item() * 8 / (1024 ** 2)
    logger.info(f"Lodade {cfg['Model']['name']} parameters : {params:.3e} M")
    
    # Trainning
    train(copy.deepcopy(cfg), net, logger, dataset=args['Train_Dataset'])
    
    # Testing
    path = os.path.join(cfg['Rec']['dir'], 'best_checkpoint.pkl')
    net.load_state_dict(torch.load(path))
    eval_GT(copy.deepcopy(cfg), net, logger, dataset=args['Test_Dataset'], mode='flow_dt1')