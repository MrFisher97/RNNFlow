import importlib
import json

import numpy as np
import os
import torch
from torch.cuda.amp import GradScaler, autocast

from Models.flow_metrics import EventWarping

import Tools
from Tools.visualize import VisdomPlotter

from tester import eval_GT
import copy

def train(cfg, net, logger, dataset='UZHFPV'):
    #---------- Environment ---------
    if cfg['Rec']['enable']:
        plotter = VisdomPlotter(env='train', port=7000)
    
    cfg['Data'].update(cfg['Train_Dataset'][dataset])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    param_list = [{'params':net.parameters(), 'lr':cfg['Data']['lr']}]
    optimizer = torch.optim.Adam(param_list)
    # loader = H5Dataloader(**cfg["Data"])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['Data']['num_epochs'], eta_min=1e-6, verbose=True)
    
    loss_func = EventWarping(**cfg['Loss']).to(device)
    scaler = GradScaler()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-6, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, verbose=True)
    
    #-------- Start Trainning --------
    min_loss = float('inf')
    loss_tracker = Tools.Param_Tracker()
    timer = Tools.Time_Tracker()
    
    # if dataset in ['DSEC']:
    #     loader = Tools.dsec_loader.H5Dataloader(**cfg["Data"], shuffle=True)
    # else:
    loader = Tools.stream_loader.H5Dataloader(**cfg["Data"])
    
    for epoch in range(cfg['Data']['num_epochs']):
        # if "VariNum" in cfg['Data']['augmentation'] and epoch % 5 == 4:
        #     loader = H5Dataloader(**cfg["Data"], shuffle=True)
        net.train()
        for i, item in enumerate(loader, start=1):
            # ---------- Forward ----------
            if item['cur_ts'][0] == 0:
                # variLen = int(self.cfg['Data']['seq_len'] * (1 - .4 * np.random.random()))
                optimizer.zero_grad()
                loss_func.reset()
                net.reset_states()

            # with autocast():
            out = net(item['input'].to(device), cfg['Data']['resolution'])
            # out = net(item['input'].to(device), cfg['Loss']['resolution'])
            loss_func.event_flow_association(out['flow'], item['event_list'].to(device), item['event_cnt'].to(device), item['sample_mask'].to(device))
        
            if (i % cfg['Data']['seq_len']) > 0:
                continue

            if cfg['Loss']['overwrite_intermediate']:
                loss_func.overwrite_intermediate_flow(out['flow'])
            
            loss, iwe = loss_func()

            loss_tracker(loss['cur'].item(), n=cfg['Data']['batch_size'])
            # ---------- Backward ----------
            # scaler.scale(loss['cur']).backward()
            loss['cur'].backward()

            if cfg['Loss']['clip_grad']:
                torch.nn.utils.clip_grad.clip_grad_norm_(net.parameters(), cfg['Loss']['clip_grad'])

            # scaler.step(optimizer)
            # scaler.update()

            optimizer.step()

           # ---------- Log and Visualize ----------
            if cfg['Rec']['enable'] and i % (cfg['Data']['seq_len'] * 2) == 0:
                flow = out['flow'][-1].detach().cpu()
                flow = flow * item['event_mask'][:, None]
                plotter.step(item['event_cnt'], 
                             flow, 
                             iwe.detach().cpu(),
                             loss_tracker.avg)

            timer(cfg['Data']['batch_size'])
            # if dataset in ['DSEC']:
            #     infor = f'{i:04d} / {len(loader.dataset):04d} '
            # else:
            infor = f'{loader.dataset.pos.value:03d} / {len(loader.dataset):03d}, '
            infor += f'loss track:{loss_tracker.avg:.3f}, '
            for k, v in loss.items():
                infor += f"{k}: {v.item() / cfg['Data']['batch_size']:.3f}, "
            print(infor + f'{timer.avg:.4f} seconds/batch', end="\r",)

            optimizer.zero_grad(set_to_none=True)
            net.detach_states()
            loss_func.reset()
            timer.start()

        # scheduler.step(train_result['loss'])
        logger.info(f"Epoch: {epoch}, loss:{loss_tracker.avg:.3f}, {timer.avg:.6f} seconds/batch")

        if cfg['Rec']['enable']:
            plotter.vis_curve(X=np.array([loss_tracker.avg]), Y=np.array([epoch]), win='loss curve')

        if loss_tracker.avg < min_loss:
            torch.save(net.state_dict(), os.path.join(cfg['Rec']['dir'], 'best_checkpoint.pkl'))
            min_loss = loss_tracker.avg
            best_epoch = epoch
        
        # TEST: For epoch recording
        if epoch % 10 == 0:
            # torch.save(net.state_dict(), os.path.join(cfg['Rec']['dir'], f'checkpoint_{epoch}.pkl'))
            eval_GT(copy.deepcopy(cfg), net, logger, dataset='MVSEC')
            # cfg['Data'].update(cfg['Train_Dataset'][dataset])
        
        loss_tracker.reset()
        timer.reset()

        # scheduler.step()

    logger.info(f"Min loss {min_loss:.3f} @ {best_epoch} epoch")

if __name__ == '__main__':
    from main import init_seed, log_setting
    import time, argparse

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
        cfg['timestamp'] = time.strftime('%m%d%H%M%S', time.localtime(time.time())) # Add timestamp with format mouth-day-hour-minute
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