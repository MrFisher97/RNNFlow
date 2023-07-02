import os
import cv2
import json
import torch
import argparse
import importlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import Tools
from Tools.visualize import VisdomPlotter, Visualization
from Tools.stream_loader import H5Dataloader

def visual(cfg, net):
    cfg['Data'].update({'mode': 'events',
                        'window':8000,
                        'batch_size':1,
                        'augment_prob':[],
                        'augmentation':[]})

    loader = H5Dataloader(**cfg["Data"], shuffle=False)

    os.makedirs('Tmp', exist_ok=True)
    plotter = VisdomPlotter(env='test', port=7000)
    vis = Visualization(resolution=cfg['Data']['resolution'][0], path_results='Tmp')

    ind = 0
    device = cfg['device']
    with torch.no_grad():
        for item in loader:
            if item['cur_ts'][0] == 0:
                net.reset_states()

            f1_list = []
            f2_list = []
            def hook1(moudle, fi, fo):
                f1_list.append(fi)
                f2_list.append(fo)
                return None

            net.encoders[0].decouple.dec.register_forward_hook(hook1)
            out = net(item['input'].to(device))

            c = 2
            img = f2_list[-1][0, c]
            img = img.abs() 
            # mask = ~(img == 0)
            img = img - img.min()
            img = img / img.max()
            # img = img ** 0.8
            img = 1 - img
            # img = img * mask
            
            plt.imshow(img.cpu().numpy(), cmap=mpl.colormaps['cubehelix'])
            plt.colorbar()
            plt.show()
            exit(0)

def visual_event(cfg):
    cfg['Data'].update({'mode': 'events',
                        'window':50000,
                        'batch_size':1,
                        'augment_prob':[],
                        'augmentation':[]})

    loader = H5Dataloader(**cfg["Data"], shuffle=False)
    for item in loader:
        vimg = item['input'][0].permute(1, 2, 0).numpy()
        vimg = (vimg * 255).astype(np.uint8)
        mask = np.where(vimg == 0)
        vimg = cv2.applyColorMap(vimg, 8)
        vimg[mask[:2]] = 255
        cv2.imshow('timesurface', vimg)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            exit(0)
        if k == ord('n'):
            break

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='DecoupleNet')
    args.add_argument('--Test_Dataset', type=str, default='tmp')
    args.add_argument('--dir', type=str, default='Output/DecoupleNet/06281018', help='Arguments for overriding config')
    args.add_argument('--override', type=str, help='Arguments for overriding config')
    args = vars(args.parse_args())
    cfg = json.load(open(f"Config/{args['config']}.json", 'r'))

    if args['override']:
        cfg = Tools.override(args['override'], cfg)

    cfg['Rec']['dir'] = args['dir']
    cfg['Model'].update(cfg['Data'])
    cfg['Data'].update(cfg['Test_Dataset'][args['Test_Dataset']])
    
    # # ---- load model -----
    # net = importlib.import_module(f"Models.{cfg['Model']['name']}").Model(**cfg['Model'])
    # net.load_state_dict(torch.load(os.path.join(cfg['Rec']['dir'], 'best_checkpoint.pkl')))
    
    # #---------- Environment -------
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cfg['device'] = device
    # net = net.to(device)

    # visual(cfg, net)
    visual_event(cfg)

