import argparse
import importlib
import json
import logging
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"

import torch
from torch.cuda.amp import GradScaler, autocast

from Models.iwe import compute_pol_iwe, forward_interpolate_tensor
from Models.flow_metrics import Validation

import copy
import Tools
# from Tools.stream_loader import H5Dataloader
# from Tools.dsec_loader import H5Dataloader

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

def eval_FWL(cfg, net, logger, dataset='MVSEC'):
    #-------- Start Testing --------
    cfg['Data'].update(cfg['Test_Dataset'][dataset])
    cfg['Data'].update({'batch_size':1,
                        'resolution': cfg['Data']['resolution'],
                        'path': cfg['Data']['path'],
                        'mode':'images',
                        'window':1,
                        'augment_prob':[],
                        'augmentation':[]})

    if cfg['Rec']['enable']:
        plotter = VisdomPlotter(env='test', port=7000)
        vis = Visualization(resolution=cfg['Data']['resolution'][0], path_results=cfg['Rec']['dir'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    if dataset in ['DSEC']:
        loader = Tools.dsec_loader.H5Dataloader(**cfg["Data"], shuffle=False)
    else:
        loader = Tools.stream_loader.H5Dataloader(**cfg["Data"], shuffle=False)
    
    metric = Validation(resolution=cfg['Data']['resolution'], flow_scaling=cfg['Loss']['flow_scaling'], device=device)
    val_results = {}

    net.eval()
    with torch.no_grad():
        for item in loader:
            if item['cur_ts'][0] == 0:
                print('\n')
                # net.reset_states()
    
            # ---------- Predict ----------
            item = {k:v.to(device) if type(v) is torch.Tensor else v for k, v in item.items()}
            out = net(item['input'])
            flow = out['flow'][-1]

            # ---------- Metric Computation ----------
            metric.event_flow_association(flow, item['event_list'])
                
            if cfg['Loss']['overwrite_intermediate']:
                metric.overwrite_intermediate_flow(flow)

            if cfg['Data']['mode'] == 'events' and metric.num_events < cfg['Data']['window']:
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
                plotter.vis_event(item['event_cnt'].detach().cpu(), if_standard=True, win='raw')
                plotter.vis_event(iwe.detach().cpu(), if_standard=True, win='iwe')

            infor = f'{loader.dataset.pos.value:03d} / {len(loader.dataset):03d}' +\
                    f'{fn} FWL: {val_results[fn].avg:.3f}'
            print(infor, end="\r",)

            metric.reset()

    for fn, val in val_results.items():
        logger.info(f"{fn} FWL: {val.avg:.3f}")

def eval_GT(cfg, net, logger, dataset='MVSEC'):   
    #-------- Start Testing --------
    cfg['Data'].update(cfg['Test_Dataset'][dataset])
    cfg['Data'].update({'batch_size':1,
                        'augment_prob':[],
                        'augmentation':[],
                        'shuffle':False})
    
    if cfg['Rec']['enable']:
        plotter = VisdomPlotter(env='test', port=7000)
        
        if cfg['Rec']['store']:
            vis = Visualization(resolution=cfg['Data']['resolution'][0], path_results=cfg['Rec']['dir'])
        # else:
            # vis = Visualization(resolution=cfg['Data']['resolution'][0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    if dataset in ['DSEC']:
        loader = Tools.dsec_loader.H5Dataloader(**cfg["Data"])
    else:
        loader = Tools.stream_loader.H5Dataloader(**cfg["Data"])

    metric = Validation(resolution=cfg['Data']['resolution'], 
                 flow_scaling=cfg['Loss']['flow_scaling'], 
                 device=device,
                 sampling_mode='grid')
    val_results = {}
    timer = Tools.Time_Tracker()

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
            
            if cfg['Model']['name'] in ['ERaft', 'taming']:
                out = net(item['input'])
            else:
                out = net(item['input'], cfg['Data']['resolution'])
            
            flow = out['flow'][-1]
            timer()
            # ---------- Metric Computation ----------
            metric.event_flow_association(flow, **item)
                
            if cfg['Loss']['overwrite_intermediate']:
                metric.overwrite_intermediate_flow(flow, item['event_list'])

            if (item["dt_gt"] <= 0.0) or (metric.passes < (1 / cfg["Data"]["window"])):
                continue
            
            FWL, fw_IWE, IE = metric.call_fwl()
            RAST = metric.call_rast()

            FWL = FWL.item()
            RAST = RAST.item()

            fn = item["name"][0].split("/")[-1]
            if fn not in val_results.keys():
                val_results[fn] = {'FWL': Tools.Param_Tracker(), 
                                'RAST': Tools.Param_Tracker()}

            val_results[fn]['FWL'](FWL)
            val_results[fn]['RAST'](RAST)
            
            aee, aee_per = 0, 0
            if 'flow' in cfg['Data']['mode']:
                aee, aee_per = metric.call_aee()
                fn = item["name"][0].split("/")[-1]
                if "AEE" not in val_results[fn].keys():
                    val_results[fn].update({'AEE': Tools.Param_Tracker(), 
                                            'Per': Tools.Param_Tracker()})
                
                aee = aee.item()
                aee_per = aee_per.item()
                val_results[fn]['AEE'](aee)
                val_results[fn]['Per'](aee_per * 100)
             

            # ---------- Log and Visualize ----------
                # if cfg['Rec']['enable']:
                    # plotter.vis_flow(item['gtflow'].detach().cpu().permute(0, 2, 3, 1)[0], win='gt')

            if cfg['Rec']['enable']:
                # flow = flow * item['event_mask'][:, None]
                # flow = metric.compute_window_flow(mask=True)
                # flow = metric._flow_map[:, -1] * metric.flow_scaling
                # flow = flow * metric._event_mask.sum(1).bool()
                # # flow[..., 193:, :] = 0
                # plotter.vis_flow(flow.detach().cpu().permute(0, 2, 3, 1)[0], win='pred_flow', title=f"pred_flow aee:{aee:.3f}")

                # plotter.vis_event(fw_IWE.detach().cpu()[0], if_standard=True, win='iwe', title=f"iwe fwl:{FWL:.3f}")
                # plotter.vis_event(IE.detach().cpu()[0], if_standard=True, win='raw')
                # vis.update(item, masked_window_flow=flow, iwe_window=fw_IWE)
                
                if cfg["Rec"]["store"]:
                    sequence = item["name"][0].split("/")[-1].split(".")[0]
                    # vis.store(item, masked_window_flow=flow, iwe_window=fw_IWE, sequence=sequence, ts=item['ts'])

                    flow_sub = None
                    if dataset in ['DSEC']:
                        flow_sub = metric.compute_window_flow(mode="backward", mask=False) / cfg["Data"]["window"]

                    vis.store(item, 
                              sequence,
                              flow, 
                              fw_IWE, 
                              ts=item['ts'],
                              flow_info = f'AEE:{aee:.3f} Outlier:{aee_per:.3f}',
                              warp_info = f'FWL:{FWL:.3f}',
                              flow_sub=flow_sub,)

            info = f"{loader.dataset.pos.value:03d} / {len(loader.dataset):03d} \t {fn} "
            
            for k, v in val_results[fn].items():
                info += f"{k}: {v.avg:.3f} "

            info += f"elpsed: {timer.avg:.4f} s"

            print(info, end="\r",)
            metric.reset()
            # timer.start()

    final_result = {}
    result_len = {}
    for fn in val_results.keys():
        info = f"{fn} "
        for k, v in val_results[fn].items():
            info += f"{k}: {v.avg:.3f} "
            if k not in final_result:
                final_result[k] = v.tot
                result_len[k] = v.num
            final_result[k] += v.tot
            result_len[k] += v.num
        logger.info(info)
    
    info = "Final Result \t "
    for k, v in final_result.items():
        info += f"{k} : {v / result_len[k]:.3f} " 
    logger.info(info)

# def save(cfg, net, logger, dataset='MVSEC'):
#     #-------- Start Testing --------
#     cfg['Data'].update(cfg['Test_Dataset'][dataset])
#     cfg['Data'].update({'batch_size':1,
#                         'resolution': cfg['Data']['resolution'],
#                         'path': cfg['Data']['path'],
#                         'mode':'events',
#                         'window':30000,
#                         'augment_prob':[],
#                         'augmentation':[]})
#     # cfg['Data'].update({'batch_size':1,
#     #                     'resolution': cfg['Data']['resolution'],
#     #                     'path': cfg['Data']['path'],
#     #                     'mode':'images',
#     #                     'window':2,
#     #                     'augment_prob':[],
#     #                     'augmentation':[]})

#     if cfg['Rec']['enable']:
#         plotter = VisdomPlotter(env='test', port=7000)
#         vis = Visualization(resolution=cfg['Data']['resolution'][0], path_results=cfg['Rec']['dir'])
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = net.to(device)
    
#     if dataset in ['DSEC']:
#         loader = Tools.dsec_loader.H5Dataloader(**cfg["Data"])
#     else:
#         loader = Tools.stream_loader.H5Dataloader(**cfg["Data"])
    
#     metric = Validation(resolution=cfg['Data']['resolution'], flow_scaling=cfg['Loss']['flow_scaling'], device=device)
#     timer = Tools.Time_Tracker()

#     net.eval()
#     with torch.no_grad():
#         for i, item in enumerate(loader):
#             if item['cur_ts'][0] == 0:
#                 print('\n')
#                 if dataset == 'DSEC':
#                     flow_init = None
#                     valid_index = np.genfromtxt(
#                             f"Datasets/DSEC/test/raw/{item['name'][0]}/test_forward_flow_timestamps.csv",
#                             delimiter=',')[:,2].tolist()
#                 if cfg['Model']['name'] not in ['ERaft']:
#                     net.reset_states()
    
#             # ---------- Predict ----------
#             item = {k:v.to(device) if type(v) is torch.Tensor else v for k, v in item.items()}
#             if cfg['Model']['name'] in ['ERaft']:
#                 flow_low, out = net(item['input'], flow_init = flow_init)
#                 flow = out['flow'][-1]
#                 flow_init = forward_interpolate_tensor(flow_low)
#             else:
#                 out = net(item['input'], cfg['Data']['resolution'])
#                 flow = out['flow'][-1] * 128 / 0.1
#             timer()

#             # ---------- Metric Computation ----------
#             metric.event_flow_association(flow, item['event_list'])
                
#             # if cfg['Loss']['overwrite_intermediate']:
#             #     metric.overwrite_intermediate_flow(flow)

#             if cfg['Data']['mode'] == 'events' and metric.num_events < cfg['Data']['window']:
#                 continue

#             fn = item["name"][0].split("/")[-1]

#             # ---------- Log and Visualize ----------
#             sequence = item["name"][0].split("/")[-1].split(".")[0]
#             # iwe = compute_pol_iwe(item["event_list"].to(device), metric._flow_list, 1, cfg["Data"]["resolution"], cfg["Loss"]["flow_scaling"], True,)
#             iwe = compute_pol_iwe(item["event_list"].to(device), metric._flow_list, 1, cfg["Data"]["resolution"], 1, True,)

#             # flow = flow * item['event_mask'][:, None]
#             if dataset == 'DSEC':
#                 if len(valid_index) > 0 and int(item['idx'][0]) == int(valid_index[0]):
#                     import imageio
#                     dir = f"Output/{cfg['Model']['name']}/DSEC/Upload/{sequence}"
#                     os.makedirs(dir, exist_ok=True)
#                     img = np.zeros([flow.shape[-2], flow.shape[-1], 3], dtype=np.uint16)
#                     out = flow.detach().cpu().numpy().transpose(0, 2, 3, 1)[-1]
#                     img[..., :2] = np.rint(out * 128  + 2 ** 15)
#                     imageio.imwrite(os.path.join(dir, f"{item['idx'][-1]:06d}.png"), img, 'PNG-FI')
#                     valid_index.pop(0)

#             if cfg['Rec']['enable']:
#                 vis.store(item, flow, iwe=iwe, sequence=sequence, ts=item['ts'])
#                 plotter.vis_flow(flow.detach().cpu().permute(0, 2, 3, 1), win='pred_flow')
#                 plotter.vis_event(item['event_cnt'].detach().cpu(), if_standard=True, win='raw')
#                 # plotter.vis_event(iwe.detach().cpu(), if_standard=True, win='iwe')

#             # infor = f'{loader.dataset.pos.value:03d} / {len(loader.dataset):03d}' +\
#             #         f"elpsed: {timer.avg:.4f} s"
#             infor = f'{i:04d} / {len(loader.dataset):04d} ' +\
#                     f"elpsed: {timer.avg:.4f} s"
#             print(infor, end="\r",)

#             metric.reset()
#             timer.start()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='taming')
    args.add_argument('--Test_Dataset', type=str, default='MVSEC')
    args.add_argument('--dir', type=str, default='Output/taming', help='Arguments for overriding config')
    args.add_argument('--override', type=str, help='Arguments for overriding config')
    args = vars(args.parse_args())
    cfg = json.load(open(f"Config/{args['config']}.json", 'r'))

    if args['override']:        
        cfg = Tools.override(args['override'], cfg)

    cfg['Data']['Dataset'] = args['Test_Dataset']
    cfg['Rec']['dir'] = args['dir']
    logger = log_setting(cfg)
    # cfg['Model'].update(cfg['Data'])
    
    # ---- load model -----
    # net = importlib.import_module(f"Models.{cfg['Model']['name']}").Model(**cfg['Model'])
    # if cfg['Model']['name'] in ['SpikeFlowNet', 'STEFlowNet']:
    #     checkpoint = torch.load(os.path.join(cfg['Rec']['dir'], 'steflow_dt1.pth.tar'))['state_dict']
    # elif cfg['Model']['name'] in ['ERaft']:
    #     checkpoint = torch.load(os.path.join(cfg['Rec']['dir'], 'dsec.tar'))['model']
    # else:
    #     checkpoint = torch.load(os.path.join(cfg['Rec']['dir'], 'best_checkpoint.pkl'))
    
    # net.load_state_dict(checkpoint)

    net = importlib.import_module(f"models.model").RecEVFlowNet(cfg["Model"].copy())
    model_loaded = torch.load(os.path.join(cfg['Rec']['dir'], 'model/model.pth')).state_dict()

    # check for input-dependent layers
    for key in model_loaded.keys():
        if key.split(".")[1] == "pooling" and key.split(".")[-1] in ["weight", "weight_f"]:
            net.encoder_unet.pooling = net.encoder_unet.build_pooling(model_loaded[key].shape)
            net.encoder_unet.get_axonal_delays()

    new_params = net.state_dict()
    new_params.update(model_loaded)
    net.load_state_dict(new_params)
    net.eval()

    params = np.sum([p.numel() for p in net.parameters()]).item() * 8 / (1024 ** 2)
    print(f"Lodade {cfg['Model']['name']} parameters : {params:.3e} M")

    #---------- Environment ---------
    eval_GT(copy.deepcopy(cfg), net, logger, args['Test_Dataset'])
    # eval_FWL(cfg, net, logger, args['Test_Dataset'])
    # save(cfg, net, logger, args['Test_Dataset'])