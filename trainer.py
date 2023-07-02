import argparse
import importlib
import json
import logging
import numpy as np
import os
import torch
from torch.cuda.amp import GradScaler, autocast

from Models.flow_metrics import EventWarping
from Models.flow_metrics import AEE, FWL

import Tools
from Tools.stream_loader import H5Dataloader
from Tools.visualize import VisdomPlotter

def train(cfg, net, optimizer, logger):
    #---------- Environment ---------
    if cfg['Rec']['enable']:
        plotter = VisdomPlotter(env='train', port=7000)
    
    device = cfg['device']
    # loader = H5Dataloader(**cfg["Data"])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['Data']['num_epochs'], eta_min=1e-6, verbose=True)
    
    loss_func = EventWarping(**cfg['Loss'])
    scaler = GradScaler()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-6, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, verbose=True)
    
    #-------- Start Trainning --------
    min_loss = float('inf')
    loss_tracker = Tools.Param_Tracker()
    timer = Tools.Time_Tracker()
    
    loader = H5Dataloader(**cfg["Data"])
    for epoch in range(cfg['Data']['num_epochs']):
        if "VariNum" in cfg['Data']['augmentation'] and epoch % 5 == 4:
            loader = H5Dataloader(**cfg["Data"])
        net.train()
        for i, item in enumerate(loader, start=1):
            # ---------- Forward ----------
            if item['cur_ts'][0] == 0:
                # variLen = int(self.cfg['Data']['seq_len'] * (1 - .4 * np.random.random()))
                optimizer.zero_grad()
                loss_func.reset()
                timer(cfg['Data']['batch_size'])
                net.reset_states()

            with autocast():
                out = net(item['input'].to(device), cfg['Data']['resolution'])
                loss_func.event_flow_association(out['flow'], item['event_list'].to(cfg['device']), item['event_cnt'].to(cfg['device']))
                
                if (i % cfg['Data']['seq_len']) > 0:
                    continue

                if cfg['Loss']['overwrite_intermediate']:
                    loss_func.overwrite_intermediate_flow(out['flow'])
                
                loss, iwe = loss_func()

            loss_tracker(loss['cur'].item(), n=cfg['Data']['batch_size'])
            
            # ---------- Backward ----------
            scaler.scale(loss['cur']).backward()
            # loss_batch.backward()

            if cfg['Loss']['clip_grad']:
                torch.nn.utils.clip_grad.clip_grad_norm_(net.parameters(), cfg['Loss']['clip_grad'])

            scaler.step(optimizer)
            scaler.update()

            # optimizer.step()

           # ---------- Log and Visualize ----------
            if cfg['Rec']['enable'] and i % (cfg['Data']['seq_len'] * 2) == 0:
                flow = out['flow'][-1].detach().cpu() * item['event_mask'][:, None]
                plotter.vis_flow(flow.permute(0, 2, 3, 1), win='pred_flow')
                plotter.vis_event(item['event_cnt'], if_standard=True, win='raw')
                plotter.vis_event(iwe.detach().cpu(), if_standard=True, win='iwe',  title=f':{loss_tracker.avg:.3f}')

            infor = f'{loader.dataset.pos.value:03d} / {len(loader.dataset):03d}, '
            infor += f'loss track:{loss_tracker.avg:.3f}, '
            for k, v in loss.items():
                infor += f"{k}: {v.item() / cfg['Data']['batch_size']:.3f}, "
            print(infor + f'{timer.avg:.4f} seconds/batch', end="\r",)

            optimizer.zero_grad()
            net.detach_states()
            loss_func.reset()

        # scheduler.step(train_result['loss'])
        logger.info(f"Epoch: {epoch}, loss:{loss_tracker.avg:.3f}, {timer.avg:.6f} seconds/batch")

        if cfg['Rec']['enable']:
            plotter.vis_curve(X=np.array([loss_tracker.avg]), Y=np.array([epoch]), win='loss curve')

        # TEST: For epoch recording
        # if epoch % 10 == 0:
        #     torch.save(self.net.state_dict(), os.path.join(self.log_dir, f'checkpoint_{epoch}.pkl'))

        if loss_tracker.avg < min_loss:
            torch.save(net.state_dict(), os.path.join(cfg['Rec']['dir'], 'best_checkpoint.pkl'))
            min_loss = loss_tracker.avg
            best_epoch = epoch
        # scheduler.step()

    logger.info(f"Min loss {min_loss:.3f} @ {best_epoch} epoch")