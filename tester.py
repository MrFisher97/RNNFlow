import argparse
import importlib
import json
import logging
import numpy as np
import os
import torch
from torch.cuda.amp import GradScaler, autocast

from Models.iwe import compute_pol_iwe
from Models.flow_metrics import AEE, FWL

import Tools
from Tools.stream_loader import H5Dataloader
from Tools.visualize import VisdomPlotter, Visualization

def log_setting(cfg):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler() # Adding Terminal Logger

    # Adding File Logger

    fh = logging.FileHandler(filename=os.path.join(cfg['Rec']['dir'] , 'logger.txt'))
    fh.setFormatter(logging.Formatter("%(asctime)s  : %(message)s", "%b%d-%H:%M"))

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

def eval_FWL(cfg):
    if cfg['Rec']['enable']:
        plotter = VisdomPlotter(env='test', port=7000)
        vis = Visualization(resolution=cfg['Data']['resolution'][0], path_results=cfg['Rec']['dir'])

    #-------- Custom Setting --------
    cfg['Data'].update({'batch_size':1,
                        'resolution': cfg['Data']['test_orig_resolution'],
                        'path': cfg['Data']['test_path'],
                        'mode':'images',
                        'window':2,
                        'augment_prob':[],
                        'augmentation':[]})
    
    #-------- Start Testing ---------
    loader = H5Dataloader(**cfg["Data"], shuffle=False)
    metric = FWL(resolution=cfg['Data']['resolution'], flow_scaling=cfg['Loss']['flow_scaling'], device=device)
    val_results = {}

    net.eval()
    with torch.no_grad():
        for item in loader:
            if item['cur_ts'][0] == 0:
                print('\n')
                net.reset_states()
    
            # ---------- Predict ----------
            item = {k:v.to(device) if type(v) is torch.Tensor else v for k, v in item.items()}
            out = net(item['input'])
            flow = out['flow'][-1]

            # ---------- Metric Computation ----------
            metric.event_flow_association(flow, item['event_list'])
                
            if cfg['Loss']['overwrite_intermediate']:
                metric.overwrite_intermediate_flow(flow)

            if cfg['Data']['mode'] == 'events' and metric.num_events < cfg['Loss']['window']:
                continue

            val_metric, iwe, ie = metric()
            fn = item["name"][0].split("/")[-1]
            if fn not in val_results.keys():
                val_results[fn] = Tools.Param_Tracker()

            val_results[fn](val_metric.item())

           # ---------- Log and Visualize ----------
            if cfg['Rec']['enable']:
                flow = flow * item['event_mask'][:, None]
                plotter.vis_flow(flow.detach().cpu().permute(0, 2, 3, 1), win='pred_flow')
                plotter.vis_event(ie.detach().cpu(), if_standard=True, win='raw')
                plotter.vis_event(iwe.detach().cpu(), if_standard=True, win='iwe')

            infor = f'{loader.dataset.pos.value:03d} / {len(loader.dataset):03d}' +\
                    f'{fn} FWL: {val_results[fn].avg:.3f}'
            print(infor, end="\r",)

            metric.reset()

    for fn, val in val_results.items():
        logger.info(f"{fn} FWL: {val.avg:.3f}")

def eval_GT(cfg, net, logger, mode='flow_dt1'):
    if cfg['Rec']['enable']:
        plotter = VisdomPlotter(env='test', port=7000)
        vis = Visualization(resolution=cfg['Data']['resolution'][0], path_results=cfg['Rec']['dir'])

    #-------- Start Testing --------
    cfg['Data'].update({'mode': mode,
                        'window':1 if mode=='flow_dt1' else 4,
                        'batch_size':1,
                        'augment_prob':[],
                        'augmentation':[]})
    
    device = cfg['device']
    loader = H5Dataloader(**cfg["Data"], shuffle=False)

    metric = AEE(resolution=cfg['Data']['resolution'], flow_scaling=cfg['Loss']['flow_scaling'], device=device)
    val_results = {}

    net.eval()
    with torch.no_grad():
        for item in loader:
            if item['cur_ts'][0] == 0:
                print('\n')
                net.reset_states()

            # ---------- Predict --------
            item = {k:v.to(device) if type(v) is torch.Tensor else v for k, v in item.items()}
            # for k in item.keys():
            #     if type(item[k]) is torch.tensor:
            #         item[k].to(device)
    
            out = net(item['input'], cfg['Data']['resolution'])
            flow = out['flow'][-1]

            # ---------- Metric Computation ----------
            metric.event_flow_association(flow, **item)
                
            if cfg['Loss']['overwrite_intermediate']:
                metric.overwrite_intermediate_flow(flow, item['event_list'])

            if (item["dt_gt"] <= 0.0) or (metric.passes != np.round(1.0 / cfg["Data"]["window"])):
                continue

            val_metric, val_metric_per = metric()
            fn = item["name"][0].split("/")[-1]
            if fn not in val_results.keys():
                val_results[fn] = {'AEE': Tools.Param_Tracker(), 
                                   'Per': Tools.Param_Tracker()}

            val_results[fn]['AEE'](val_metric.item())
            val_results[fn]['Per'](val_metric_per.item())

           # ---------- Log and Visualize ----------
            if cfg['Rec']['enable']:
                flow = flow * item['event_mask'][:, None]
                plotter.vis_flow(flow.detach().cpu().permute(0, 2, 3, 1), win='pred_flow')
                plotter.vis_flow(item['gtflow'].detach().cpu().permute(0, 2, 3, 1), win='gt')
                iwe = compute_pol_iwe(item["event_list"].to(device), metric._flow_list, 1, cfg["Data"]["resolution"], cfg["Loss"]["flow_scaling"], True,)
            
            if cfg["Rec"]["store"]:
                sequence = item["name"][0].split("/")[-1].split(".")[0]
                vis.store(item, flow, iwe, sequence, ts=item['ts'], other_info = f'AEE:{val_metric.item():.3f}')

            infor = f"{loader.dataset.pos.value:03d} / {len(loader.dataset):03d} \t" +\
                    f"{fn} AEE: {val_results[fn]['AEE'].avg:.3f} Percent: {val_results[fn]['Per'].avg * 100:.3f}"
            print(infor, end="\r",)
            metric.reset()

    for fn, val in val_results.items():
        logger.info(f"{fn} AEE: {val['AEE'].avg:.3f} Percent: {val['Per'].avg * 100:.3f}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='DirectEncoder')
    args.add_argument('--Test_Dataset', type=str, default='MVSEC')
    args.add_argument('--dir', type=str, default='Output/DirectEncoder/06242235', help='Arguments for overriding config')
    args.add_argument('--override', type=str, help='Arguments for overriding config')
    args = vars(args.parse_args())
    cfg = json.load(open(f"Config/{args['config']}.json", 'r'))

    if args['override']:
        cfg = Tools.override(args['override'], cfg)

    cfg['Rec']['dir'] = args['dir']
    logger = log_setting(cfg)
    cfg['Model'].update(cfg['Data'])
    
    # ---- load model -----
    net = importlib.import_module(f"Models.{cfg['Model']['name']}").Model(**cfg['Model'])
    net.load_state_dict(torch.load(os.path.join(cfg['Rec']['dir'], 'best_checkpoint.pkl')))
    
    #---------- Environment ---------
    cfg['Data'].update(cfg['Test_Dataset'][args['Test_Dataset']])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device
    net = net.to(device)
    eval_GT(cfg, net, logger)
    # eval_FWL()