import argparse
import importlib
import json
import logging
import numpy as np
import os
import time
import torch

from torch.cuda.amp import GradScaler, autocast
from Models.flow_metrics import EventWarping

import Tools
from Tools.stream_loader import H5Dataloader
from tester import eval_GT
from trainer import train

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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enable = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--Model', type=str, default='DecoupleNet')
    args.add_argument('--Train_Dataset', type=str, default='UZHFPV')
    args.add_argument('--Test_Dataset', type=str, default='MVSEC')
    args.add_argument('--timestamp', type=str, default='None')
    args.add_argument('--override', type=str, help='Arguments for overriding config')
    args = vars(args.parse_args())
    cfg = json.load(open(f"Config/{args['Model']}.json", 'r'))

    init_seed(2023)

    if args['override']:
        cfg = Tools.override(args['override'], cfg)

    if args['timestamp'] == 'None':
        cfg['timestamp'] = time.strftime('%m%d%H%M', time.localtime(time.time())) # Add timestamp with format mouth-day-hour-minute
    else:
        cfg['timestamp'] = args['timestamp']
    logger = log_setting(cfg)
    cfg['Model'].update(cfg['Data'])

    # ---- load model -----
    net = importlib.import_module(f"Models.{cfg['Model']['name']}").Model(**cfg['Model'])
    if torch.__version__[0] == '2':
        torch._dynamo.config.suppress_errors = True
        net = torch.compile(net)

    params = np.sum([p.numel() for p in net.parameters()]).item() * 8 / (1024 ** 2)
    logger.info(f"Lodade {cfg['Model']['name']} parameters : {params:.3e} M")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device
    net = net.to(device)
    
    # Trainning
    cfg['Data'].update(cfg['Train_Dataset'][args['Train_Dataset']])
    param_list = [{'params':net.parameters(), 'lr':cfg['Data']['lr']}]
    optimizer = torch.optim.Adam(param_list)
    train(cfg, net, optimizer, logger)
    
    # Testing
    cfg['Data'].update(cfg['Test_Dataset'][args['Test_Dataset']])
    path = os.path.join(cfg['Rec']['dir'], 'best_checkpoint.pkl')
    net.load_state_dict(torch.load(path))
    eval_GT(cfg, net, logger)