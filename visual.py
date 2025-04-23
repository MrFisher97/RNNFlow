import os
import cv2
import json
import torch
import time
import argparse
import importlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import Tools
from Tools.visualize import VisdomPlotter, Visualization
from Tools.stream_loader import H5Dataloader

# def visual_feat(cfg, net):
#     cfg['Data'].update({'mode': 'events',
#                         'window':8000,
#                         'batch_size':1,
#                         'augment_prob':[],
#                         'augmentation':[]})

#     loader = H5Dataloader(**cfg["Data"], shuffle=False)

#     os.makedirs('Tmp', exist_ok=True)
#     # plotter = VisdomPlotter(env='test', port=7000)
#     # vis = Visualization(resolution=cfg['Data']['resolution'][0], path_results='Tmp')

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = net.to(device)
#     cnt = 0

#     f1_list = []
#     f2_list = []
#     def hook1(moudle, fi, fo):
#         f1_list.append(fi[0])
#         f2_list.append(fo)
#         return None

#     net.encoders[0].res.register_forward_hook(hook1)

#     with torch.no_grad():
#         for item in loader:
#             if item['cur_ts'][0] == 0:
#                 cnt = 0
#                 net.reset_states()

#             st = time.time()
#             f1_list = []
#             f2_list = []
            # def hook1(moudle, fi, fo):
            #     f1_list.append(fi[0])
            #     f2_list.append(fo)
            #     return None

            # net.encoders[0].dec.register_forward_hook(hook1)
            # out = net(item['input'].to(device), cfg['Data']['resolution'])


            # c = 2
            # img = f2_list[-1][0, c]
            # img = img.abs()
            # # mask = ~(img == 0)
            # # img = img - img.mode()[0].mode()[0]
            # img = img - img.min()
            # # img = img.abs()
            # img = img / img.max()
            # # img = img ** 0.8
            # img = 1 - img
            # # img = img * mask

            # plt.subplot(221)
            # ev_img = np.zeros([3, ] + list(item['event_cnt'].size()[-2:]))
            # ev_img[:2] = item['event_cnt'].detach().numpy()
            # ev_img[ev_img > 0] = 1
            # # ev_img /= ev_img.max()
            # mask = (ev_img[0] == 0) & (ev_img[1] == 0)
            # ev_img[:, mask] = 1
            # ev_img = ev_img[(0, 2, 1), ...]
            # plt.imshow(ev_img.transpose(1, 2, 0))
            # plt.xticks([])
            # plt.yticks([])

            # plt.subplot(222)
            # plt.imshow(img.cpu().numpy(), cmap=mpl.colormaps['cubehelix'])
            # # plt.colorbar()
            # plt.xticks([])
            # plt.yticks([])

            # plt.subplot(223)
            # flow = out['flow'][-1].detach().cpu()
            # flow = flow * item['event_mask'][:, None]
            # flow = flow[0].numpy().transpose(1, 2, 0)
            # flow = Visualization.flow_to_image(flow)
            # plt.imshow
            # plt.colorbar()
            # plt.xticks([])
            # plt.yticks([])

            # plt.show()
            # path = 'Tmp/En/' + cfg['Data']['name'] + '/' + item['name'][0]
            # os.makedirs(path, exist_ok=True)
            # plt.savefig(path + '/' + str(cnt) + '.jpg')
            # plt.clf()
            # cnt += 1
 
            # elp = time.time() - st
            # print(cnt, f'{elp:.3f}', end='\r')

def visual_feat(cfg, net):
    cfg['Data'].update({'mode': 'events',
                        'window':8000,
                        'batch_size':1,
                        'augment_prob':[],
                        'augmentation':[]})

    loader = H5Dataloader(**cfg["Data"], shuffle=False)

    os.makedirs('Tmp', exist_ok=True)
    # plotter = VisdomPlotter(env='test', port=7000)
    # vis = Visualization(resolution=cfg['Data']['resolution'][0], path_results='Tmp')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    cnt = 0

    f1_list = []
    f2_list = []
    def hook1(moudle, fi, fo):
        f1_list.append(fi[0])
        f2_list.append(fo)
        return None

    net.encoders[0].res.register_forward_hook(hook1)

    with torch.no_grad():
        for item in loader:
            if item['cur_ts'][0] == 0:
                cnt = 0
                net.reset_states()

            st = time.time()
            f1_list = []
            f2_list = []
            # def hook1(moudle, fi, fo):
            #     f1_list.append(fi[0])
            #     f2_list.append(fo)
            #     return None

            # net.encoders[0].dec.register_forward_hook(hook1)
            out = net(item['input'].to(device), cfg['Data']['resolution'])


            c = 2
            img = f2_list[-1][0, c]
            img = img.abs()
            # mask = ~(img == 0)
            # img = img - img.mode()[0].mode()[0]
            img = img - img.min()
            # img = img.abs()
            img = img / img.max()
            # img = img ** 0.8
            img = 1 - img
            # img = img * mask

            plt.subplot(221)
            ev_img = np.zeros([3, ] + list(item['event_cnt'].size()[-2:]))
            ev_img[:2] = item['event_cnt'].detach().numpy()
            ev_img[ev_img > 0] = 1
            # ev_img /= ev_img.max()
            mask = (ev_img[0] == 0) & (ev_img[1] == 0)
            ev_img[:, mask] = 1
            ev_img = ev_img[(0, 2, 1), ...]
            plt.imshow(ev_img.transpose(1, 2, 0))
            plt.xticks([])
            plt.yticks([])

            plt.subplot(222)
            plt.imshow(img.cpu().numpy(), cmap=mpl.colormaps['cubehelix'])
            # plt.colorbar()
            plt.xticks([])
            plt.yticks([])

            plt.subplot(223)
            flow = out['flow'][-1].detach().cpu()
            flow = flow * item['event_mask'][:, None]
            flow = flow[0].numpy().transpose(1, 2, 0)
            flow = Visualization.flow_to_image(flow)
            plt.imshow(flow)
            # plt.colorbar()
            plt.xticks([])
            plt.yticks([])

            plt.show()
            # path = 'Tmp/En/' + cfg['Data']['name'] + '/' + item['name'][0]
            # os.makedirs(path, exist_ok=True)
            # plt.savefig(path + '/' + str(cnt) + '.jpg')
            # plt.clf()
            cnt += 1
 
            elp = time.time() - st
            print(cnt, f'{elp:.3f}', end='\r')

def visual_event(cfg):
    cfg['Data'].update({'mode': 'events',
                        'window':30000,
                        'batch_size':1,
                        'encoding': 'cnt',
                        'augment_prob':[],
                        'augmentation':[]})

    loader = H5Dataloader(**cfg["Data"], shuffle=False)
    # len_list = [[] for _ in range(4)]
    i = -1
    item_name = []
    for item in loader:
        if item['cur_ts'][0] == 0:
            i += 1
            item_name.append(item['file_name'])
        # len_list[i].append(item['input'].size(1) / item['event_mask'].sum((1, 2)))
        vimg = item['input'][0].permute(1, 2, 0).numpy()
        # vimg = vimg ** 1.5
        vimg = (vimg * 255).astype(np.uint8)
        mask = np.where(vimg == 0)
        vimg = cv2.applyColorMap(vimg, cv2.COLORMAP_PLASMA)
        vimg[mask[:2]] = 255
        cv2.imshow('timesurface', vimg)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows() 
            exit(0)
        if k == ord('n'):
            break
    # print(item_name)
    # for i in range(4):
    #     plt.subplot(4, 1, i + 1)
    #     plt.plot(len_list[i])
    #     avg = sum(len_list[i]) / len(len_list[i])
    #     plt.plot([avg] * len(len_list[i]))
    #     print(avg)
    # plt.show()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='DecoupleNet')
    args.add_argument('--Test_Dataset', type=str, default='tmp')
    # args.add_argument('--dir', type=str, default='Output/DecoupleNet/Bitahub/norm_conv', help='Arguments for overriding config')
    args.add_argument('--dir', type=str, default='Output/DecoupleNet/07181741', help='Arguments for overriding config')

    args.add_argument('--override', type=str, help='Arguments for overriding config')
    args = vars(args.parse_args())
    cfg = json.load(open(f"Config/{args['config']}.json", 'r'))

    if args['override']:
        cfg = Tools.override(args['override'], cfg)

    cfg['Rec']['dir'] = args['dir']
    cfg['Model'].update(cfg['Data'])
    cfg['Data'].update(cfg['Test_Dataset'][args['Test_Dataset']])
    
    # # ---- load model -----
    net = importlib.import_module(f"Models.{cfg['Model']['name']}").Model(**cfg['Model'])
    net.load_state_dict(torch.load(os.path.join(cfg['Rec']['dir'], 'best_checkpoint.pkl')))
    
    # #---------- Environment -------
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cfg['device'] = device
    # net = net.to(device)

    visual_feat(cfg, net)

    # visual_event(cfg)
