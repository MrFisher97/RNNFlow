import os
import sys

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from iwe import get_interpolation, interpolate

def get_event_flow(flow, loc, mask, mode='gather'):
    """
    Sample optical flow maps using event indices
    :param flow: [batch_size x 2 x H x W] horizontal optical flow map
    :param loc: [batch_size x N x 2] event locations(x, y)
    :param mask: [batch_size x N] event polarity mask
    :return event_flow: [batch_size x N x 2] per-event optical flow (x, y)
    """
    _, _, h, w = flow.shape

    if mode == 'grid':
        loc = loc.float()
        loc[..., 0] = 2 * loc[..., 0] / (w - 1) - 1
        loc[..., 1] = 2 * loc[..., 1] / (h - 1) - 1
        loc = loc[mask].view(flow.size(0), 1, -1, 2)
        ev_flow = F.grid_sample(flow, loc, mode="bilinear", align_corners=True)
        ev_flow = ev_flow.squeeze(2)
        ev_flow = ev_flow.permute(0, 2, 1)
    elif mode == 'gather':
        loc[..., 1] *= w  # torch.view is row-major
        loc = torch.sum(loc, dim=2).long() # B x N x 2
        loc = loc[mask].view(flow.size(0), -1)
        flow = flow.view(flow.shape[0], 2, -1).permute(0, 2, 1) # B x (HW) x 2
        # event_flow = flow[batch_idx, flow_idx, :]
        ev_flow = torch.gather(flow, 1, loc[..., None].repeat(1, 1, 2))
    return ev_flow
    
def Gradient(I):
    I = F.pad(I, (0, 1, 0, 1))
    # x = gaussian_blur(x, kernel_size=3)
    mask = (I[..., 1:, 1:] > 0) & (I[..., 1:, :-1] > 0)
    I_x = I[..., 1:, 1:] - I[..., 1:, :-1]
    I_x = I_x * mask


    mask = (I[..., 1:, 1:] > 0) & (I[..., :-1, 1:] > 0) 
    I_y = I[..., 1:, 1:] - I[..., :-1, 1:]
    I_y = I_y * mask
    return I_x ** 2 + I_y ** 2

# class SobelTorch(nn.Module):
#     def __init__(
#         self, ksize: int = 3, in_channels: int = 2):
#         super().__init__()
#         self.in_channels = in_channels
#         self.filter_dx = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=1,
#             kernel_size=ksize,
#             stride=1,
#             padding=1,
#             bias=False,
#         )
#         self.filter_dy = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=1,
#             kernel_size=ksize,
#             stride=1,
#             padding=1,
#             bias=False,
#         )
#         # x in height direction
  
#         Gx = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
#         Gy = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])

#         self.filter_dx.weight = nn.Parameter(Gx.unsqueeze(0).unsqueeze(0), requires_grad=False)
#         self.filter_dy.weight = nn.Parameter(Gy.unsqueeze(0).unsqueeze(0), requires_grad=False)

#     def forward(self, img):
#         if self.in_channels == 2:
#             dxx = self.filter_dx(img[..., [0], :, :])
#             dyy = self.filter_dy(img[..., [1], :, :])
#             dyx = self.filter_dx(img[..., [1], :, :])
#             dxy = self.filter_dy(img[..., [0], :, :])
#             return torch.cat([dxx, dyy, dyx, dxy], dim=1)
#         elif self.in_channels == 1:
#             dx = self.filter_dx(img[..., [0], :, :])
#             dy = self.filter_dy(img[..., [0], :, :])
#             return torch.cat([dx, dy], dim=1)

def iwe_formation(events, flow, tref, max_ts, resolution=(128, 128), flow_scaling=128):
    warp_pos, warp_weights = get_interpolation(events[..., 1:3], events[..., 0:1], flow, tref, resolution, flow_scaling, round_idx=False)
    iwe = torch.stack([
            interpolate(warp_pos, warp_weights, resolution, events[..., -1] == 1),
            interpolate(warp_pos, warp_weights, resolution, events[..., -1] == -1)], dim=1)
    ts = 1 - (tref - events[..., 0:1]).abs() / max_ts
    ts_iwe = torch.stack([
            interpolate(warp_pos, warp_weights * ts, resolution, events[..., -1] == 1),
            interpolate(warp_pos, warp_weights * ts, resolution, events[..., -1] == -1),
            ], dim=1)
    
    return {'iwe':iwe, 
            'ts_iwe':ts_iwe}

def warp_contrast(tref, max_ts, flow, events, resolution=(128, 128), flow_scaling=128, nograd_flow=None, nograd_events=None):
    '''
    Warp Image contrast maximization Loss
    Input:
        tref: target time
        max_ts: max timestamp
        flow: event flow [batch_size x N x 2] (vx, vy)
        events: [batch_size x N x 4] (ts, x, y, p)
        pol_mask: polarity mask of events [batch_size x N x 2]
        resolution: resolution of the image space (int, int)
        flow_scaling: scalar that multiplies the optical flow map
    '''
    # per-polarity image of (forward) warped events
    # B * 2 * H * W
    iwe_dict = iwe_formation(events, flow, tref, max_ts, resolution, flow_scaling)

    # w/o pol.
    iwe = iwe_dict['iwe']
    ts_iwe = iwe_dict['ts_iwe']
    
    if nograd_events.size(1) > 0:
        nograd_iwe_dict = iwe_formation(nograd_events, nograd_flow, tref, max_ts, resolution, flow_scaling)
        iwe += nograd_iwe_dict['iwe']
        ts_iwe += nograd_iwe_dict['ts_iwe']

    ts_iwe = ts_iwe / (iwe + 1e-9)
    loss = (ts_iwe ** 2).sum(dim=(1, 2, 3))
    nonzero_px = torch.sum(iwe, dim=1).bool()
    loss /= (nonzero_px.sum(dim=(1, 2)) + 1e-9)

    N = iwe.size(2) * iwe.size(3)
    loss_e = (N / torch.exp(- 0.6 * iwe).sum(dim=(2,3))).sum(1) - 2
    loss += loss_e
    return loss, iwe

# def flow_smooth(flow, events):
#     '''
#     Flow smoothing Loss
#     Input:
#         flow: event flow [batch_size x N x 2] (x, y)
#         events: [batch_size x N x 4] (ts, x, y, p)
#     '''

#     sqrdist = square_distance(events[..., 1:3], events[..., 1:3]) # B N N

#     #Smoothness
#     d, kidx = torch.topk(sqrdist, 4, dim=-1, largest=False, sorted=False)
#     grouped_flow = grouping_operation(flow.transpose(1, 2).contiguous(), kidx).transpose(1, 2) # B x N x 4 x 2
#     diff_flow = (grouped_flow - flow.unsqueeze(-1)) ** 2

#     # mask_d = (d < 9).sum(axis=2)
#     # zero_flow = torch.norm(flow, p=2, dim=2)

#     loss = diff_flow.mean(axis=(-1, -2)).mean()
#     return loss

def charbonnier_loss(x, a = 0.5, mask=None):
    x = x ** 2
    if mask is not None:
        return torch.pow(x + 1e-6, a) * mask
    else:
        return torch.pow(x + 1e-6, a)
    # x = (x ** 2)
    # x = x * mask if mask is not None else x
    # return torch.pow(x[x > 0], a)


def flow_smooth(flow, event_cnt):
    mask = event_cnt.sum(dim=2, keepdim=True) > 0
    flow_lr = (flow[..., :-1] - flow[..., 1:])
    flow_ud = (flow[..., :-1, :] - flow[..., 1:, :])
    flow_lurd = (flow[..., :-1, :-1] - flow[..., 1:, 1:])
    flow_ldru = (flow[..., :-1, 1:] - flow[..., 1:, -1:])

    loss = 0
    loss += charbonnier_loss(flow_lr, mask=(mask[..., :-1] & mask[..., 1:])).sum()
    loss += charbonnier_loss(flow_ud, mask=(mask[..., :-1, :] & mask[..., 1:, :])).sum()
    loss += charbonnier_loss(flow_lurd, mask=(mask[..., :-1, :-1] & mask[..., 1:, 1:])).sum()
    loss += charbonnier_loss(flow_ldru, mask=(mask[..., :-1, 1:] & mask[..., 1:, -1:])).sum()

    i = 4
    if flow.size(1) > 1:
        flow_dt = flow[:, :-1] - flow[:, 1:]
        loss += charbonnier_loss(flow_dt, mask=(mask[:, :-1] & mask[:, 1:])).sum()
        i += 1
    loss /= flow.size(1)
    return loss / i

class EventWarping(torch.nn.Module):
    """
    Contrast maximization loss, as described in Section 3.2 of the paper 'Unsupervised Event-based Learning
    of Optical Flow, Depth, and Egomotion', Zhu et al., CVPR'19.
    The contrast maximization loss is the minimization of the per-pixel and per-polarity image of averaged
    timestamps of the input events after they have been compensated for their motion using the estimated
    optical flow. This minimization is performed in a forward and in a backward fashion to prevent scaling
    issues during backpropagation.
    """

    def __init__(self, resolution, flow_regul_weight,
                mask_output=True,
                overwrite_intermediate=False,
                flow_scaling=None,
                loss_scaling=True,
                sampling_mode='gather',
                **kwargs):
        super(EventWarping, self).__init__()
        self.loss_scaling = loss_scaling
        self.res = (resolution, ) * 2 if type(resolution) == int else resolution
        self.flow_scaling = flow_scaling if flow_scaling else max(resolution)
        self.flow_regul_weight = flow_regul_weight
        self.mask_output = mask_output
        self.overwrite_intermediate = overwrite_intermediate
        self.sampling_mode = sampling_mode

        self.reset()

    def event_flow_association(self, flow_list, event_list, event_cnt, sample_mask):
        """
        :param flow_list: [batch_size x N x 2] list of optical flow (x, y) map
        :param event_list: [batch_size x N x 4] input events (ts, x, y, p)
        """

        # get flow for every event in the list and update flow maps
        flowmap_list = []
   
        # B = flow_idx.shape[0]
        # batch_idx = torch.arange(B).view(B, 1).expand(-1, flow_idx.shape[1])
        event_flow = []
        nograd_event_flow = []

        flow_idx = event_list[:, :, 1:3]
    
        for i, flow in enumerate(flow_list):
            flowmap_list.append(flow[:, None])
            event_flow.append(get_event_flow(flow, flow_idx.clone(), sample_mask, self.sampling_mode))

            with torch.no_grad():
                nograd_event_flow.append(get_event_flow(flow, flow_idx.clone(), ~sample_mask, self.sampling_mode))
            
            # flow = flow.view(flow.shape[0], 2, -1).permute(0, 2, 1) # B x (HW) x 2
            # # event_flow = flow[batch_idx, flow_idx, :]
            # ev_flow = torch.gather(flow, 1, flow_idx[..., None].repeat(1, 1, 2))
            # event_flow.append(ev_flow)

        if len(self._flow_list) == 0:
            self._flow_list = event_flow
        else:
            self._flow_list = [torch.cat([self._flow_list[i], flow], dim=1) for i, flow in enumerate(event_flow)]

        if len(self._nograd_flow_list) == 0:
            self._nograd_flow_list = nograd_event_flow
        else:
            self._nograd_flow_list = [torch.cat([self._nograd_flow_list[i], flow], dim=1) for i, flow in enumerate(nograd_event_flow)]

        if len(self._flow_maps) == 0:
            self._flow_maps = flowmap_list
        else:
            self._flow_maps = [torch.cat([self._flow_maps[i], flow], dim=1) for i, flow in enumerate(flowmap_list)]

        if len(self._event_map) == 0:
            self._event_map = event_cnt[:, None]
        else:
            self._event_map = torch.cat([self._event_map, event_cnt[:, None]], dim=1)

        # update internal event list
        event_list[:, :, 0:1] += self.passes# only nonzero second time
        if len(self._event_list) == 0:
            self._event_list = event_list[sample_mask].view(flow.size(0), -1, 4)
            self._nograd_event_list = event_list[~sample_mask].view(flow.size(0), -1, 4)
        else:
            self._event_list = torch.cat([self._event_list, event_list[sample_mask].view(flow.size(0), -1, 4)], dim=1)
            self._nograd_event_list = torch.cat([self._nograd_event_list, event_list[~sample_mask].view(flow.size(0), -1, 4)], dim=1)

        # if len(self._out_ts_map) == 0:
        #     self._out_ts_map = out_ts[:, None]
        # else:
        #     self._out_ts_map = torch.cat([self._out_ts_map, out_ts[:, None]], dim=1)

        # if len(self._inp_ts_map) == 0:
        #     self._inp_ts_map = inp_ts[:, None]
        # else:
        #     self._inp_ts_map = torch.cat([self._inp_ts_map, inp_ts[:, None]], dim=1)

        # update timestamp index
        self.passes += 1

    def overwrite_intermediate_flow(self, flow_list, sample_mask):
        """
        :param flow_maps: [[batch_size x 2 x H x W]] list of optical flow (x, y) maps
        """
        self._flow_list = []
        self._flow_maps = []
        if self.sampling_mode == 'gather':
            flow_idx = self._event_list[:, :, 1:3].clone()
            flow_idx[:, :, 1] *= self.res[1]  # torch.view is row-major
            flow_idx = torch.sum(flow_idx, dim=2).long() # B x N x 2
        elif self.sampling_mode == 'grid':
            flow_idx = self._event_list[:, :, 1:3].clone()
            flow_idx[..., 0] = 2 * flow_idx[..., 0] / (self.res[0] - 1) - 1
            flow_idx[..., 1] = 2 * flow_idx[..., 1] / (self.res[0] - 1) - 1
            # flow_idx = torch.roll(flow_idx, 1, dims=-1) #(y, x) to (x, y)
    
        for i, flow in enumerate(flow_list):
            self._flow_list.append(get_event_flow(flow, flow_idx, sample_mask, self.sampling_mode))

            with torch.no_grad():
                self._nograd_flow_list.append(get_event_flow(flow, flow_idx, ~sample_mask, self.sampling_mode))
            
            # flow = flow.view(flow.shape[0], 2, -1).permute(0, 2, 1) # B x (HW) x 2
            # # event_flow = flow[batch_idx, flow_idx, :]
            # ev_flow = torch.gather(flow, 1, flow_idx[..., None].repeat(1, 1, 2))
            # self._flow_list.append(ev_flow)

            self._flow_maps.append(flow[:, None])

    def reset(self):
        self.passes = 0
        self._event_map = []
        self._event_list = []
        self._nograd_event_list = []
        self._flow_list = []
        self._nograd_flow_list = []
        self._flow_maps = []
        self._out_ts_map = []
        self._inp_ts_map = []

    @property
    def num_events(self):
        return self._event_list.size(1)

    @property
    def max_ts(self):
        return self.passes

    def forward(self):
        loss = {'cur': [],
                'fw': [],
                'bw': [],              
                # 'mid': [],
                # 'smooth': [],
                }

        for i in range(len(self._flow_maps)):
            tot = 0
            if 'fw' in loss.keys():
                fw, iwe = warp_contrast(self.max_ts, 
                                        self.max_ts, 
                                        self._flow_list[i], 
                                        self._event_list, 
                                        self.res, 
                                        self.flow_scaling, 
                                        self._nograd_flow_list[i], 
                                        self._nograd_event_list)
                # non_zero = fw > 0
                # fw = torch.sum(fw) / torch.sum(non_zero)
                # loss['fw'].append(fw.mean(dim=0))
                loss['fw'].append(fw.sum())
                tot += loss['fw'][i]
                # loss['fw'].append(torch.ones(1).to(raw))

            # mid, _ = warp_contrast(self.max_ts / 2, self.max_ts, self._flow_list[i], self._event_list, self.res, self.flow_scaling, self.loss_scaling)
            # loss['mid'].append(mid / raw_contrast
            # loss['mid'].append(mid.sum())
            # loss['mid'].append(torch.ones(1).to(fw))

            if 'bw' in loss.keys():
                bw, _ = warp_contrast(0, 
                                    self.max_ts, 
                                    self._flow_list[i], 
                                    self._event_list, 
                                    self.res, 
                                    self.flow_scaling, 
                                    self._nograd_flow_list[i], 
                                    self._nograd_event_list)

                # loss['bw'].append(bw.mean(dim=0))
                loss['bw'].append(bw.sum())
                # loss['bw'].sum()append(torch.ones(1).to(raw))
                tot += loss['bw'][i]
                tot /= 2

            if 'smooth' in loss.keys():
                fs = flow_smooth(self._flow_maps[i], self._event_map)
                # loss['smooth'].append(fs.mean(dim=0))
                loss['smooth'].append(fs.sum())
                # loss['smooth'].append(torch.ones(1).to(fw))
                tot = tot + self.flow_regul_weight * loss['smooth'][i]

            loss['cur'].append(tot)

        def mean(data:list):
            return sum(data) / len(data)
        
        loss = {k:mean(v) for k, v in loss.items()}

        # sim = charbonnier_loss((self._out_ts_map - self._inp_ts_map) * (self._inp_ts_map > 0))
        # loss['sim'] = 0.01 * sim.reshape(self._inp_ts_map.size(0), -1).sum(dim=1).mean()
        # tot += loss['sim']
        
        return loss, iwe

class Validation(torch.nn.Module):
    """
    Base class for validation metrics.
    """

    def __init__(self, resolution, device, overwrite_intermediate=False, flow_scaling=128, sampling_mode='gather', warp_mode='linear', **kwargs):
        super(Validation, self).__init__()
        self.res = resolution
        self.flow_scaling = flow_scaling  # should be specified by the user
        self.overwrite_intermediate = overwrite_intermediate
        self.device = device
        self.sampling_mode = sampling_mode
        self.warp_mode = warp_mode

        my, mx = torch.meshgrid(torch.arange(self.res[0]), torch.arange(self.res[1]))
        self.indices_map = torch.stack([mx, my], dim=0).unsqueeze(0).to(device) # 1x2xHxW
        self.indices_mask = torch.ones((1, self.res[0] * self.res[1], 1), dtype=bool).to(device)

        self.reset()

    @property
    def num_events(self):
        if self._event_list is None:
            return 0
        else:
            return self._event_list.shape[1]

    def event_flow_association(self, flow, event_list, event_mask, dt_input, dt_gt, gtflow=None, **kwargs):
        """
        :param flow_list: [batch_size x N x 2] list of optical flow (x, y) maps
        :param event_list: [batch_size x N x 4] input events (ts, x, y, p)
        :param gtflow: [batch_size x H x W x 2]
        """
        # valid = (event_list[..., 1] < self.res[1]) & (event_list[..., 1] >= 0)
        # valid &= (event_list[..., 2] < self.res[0]) & (event_list[..., 2] >= 0)
        # event_list[..., -1][~valid] = -np.float('inf')

        flow_idx = event_list[..., 1:3].clone()
        # flow_idx = torch.roll(flow_idx, 1, dims=-1) #(y, x) to (x, y)

        event_flow = get_event_flow(flow, flow_idx, torch.ones_like(flow_idx[..., 0]).to(torch.bool), self.sampling_mode)

        ########################
        # ACCUMULATED FLOW (BACKWARD WARPING)
        ########################
    
        indices = self.indices_map.clone()
        if self._flow_warping_indices is not None:
            indices = self._flow_warping_indices.clone()
        mask_valid = (
            (indices[:, 0:1] >= 0)
            * (indices[:, 0:1] <= self.res[1] - 1.0)
            * (indices[:, 1:2] >= 0)
            * (indices[:, 1:2] <= self.res[0] - 1.0)
        )
        self._flow_out_mask += mask_valid.float()

        indices_list = indices.view(1, 2, -1).permute(0, 2, 1).float().clone()
        curr_flow = get_event_flow(flow, indices_list, torch.ones_like(indices_list).to(torch.bool), mode='grid')
        curr_flow = curr_flow.view(1, self.res[0], self.res[1], 2).permute(0, 3, 1, 2)

        warped_indices = indices + curr_flow * mask_valid.float() * self.flow_scaling
        self._accum_flow_map = warped_indices - self.indices_map
        self._flow_warping_indices = warped_indices

        if self._flow_list is None:
            self._flow_list = event_flow
        else:
            self._flow_list = torch.cat([self._flow_list, event_flow], dim=1)

        if self._flow_map is None:
            self._flow_map = flow[:, None]
        else:
            self._flow_map = torch.cat([self._flow_map, flow[:, None]], dim=1)

        # update internal event list
        if self._event_list is None:
            self._event_list = event_list
        else:
            event_list = event_list.clone()  # to prevent issues with other metrics
            event_list[:, :, 0:1] += self.passes  # only nonzero second time
            self._event_list = torch.cat([self._event_list, event_list], dim=1)

        if self._event_mask is None:
            self._event_mask = event_mask[:, None]
        else:
            self._event_mask = torch.cat([self._event_mask, event_mask[:, None]], dim=1)

        # update ground-truth optical flow
        self._gtflow = gtflow

        # update timestamps
        self._dt_input = dt_input
        self._dt_gt = dt_gt

        # update timestamp index
        self.passes += 1

    def overwrite_intermediate_flow(self, flow_map):
        """
        :param flow_list: [batch_size x N x 2] list of optical flow (x, y)
        :param event_list: [batch_size x N x 4] list of events
        """

        if self.sampling_mode == 'gather':
            flow_idx = self._event_list[:, :, 1:3].clone()
            flow_idx[:, :, 1] *= self.res[1]  # torch.view is row-major
            flow_idx = torch.sum(flow_idx, dim=2).long() # B x N x 2
        elif self.sampling_mode == 'grid':
            flow_idx = self._event_list[:, :, 1:3].clone()
            flow_idx[..., 0] = 2 * flow_idx[..., 0] / (self.res[0] - 1) - 1
            flow_idx[..., 1] = 2 * flow_idx[..., 1] / (self.res[0] - 1) - 1
            # flow_idx = torch.roll(flow_idx, 1, dims=-1) #(y, x) to (x, y)

        valid = (self._event_list[..., 1] < self.res[1]) & (self._event_list[..., 1] >= 0)
        valid &= (self._event_list[..., 2] < self.res[0]) & (self._event_list[..., 2] >= 0)
        event_flow = get_event_flow(flow, flow_idx, valid, self.sampling_mode)
        
        self._flow_list = event_flow
        self._flow_map = flow_map
        self._event_mask = torch.sum(self._event_mask, dim=1, keepdim=True)
        self._event_mask[self._event_mask > 1] = 1

    def reset(self):
        self.passes = 0
        self._event_list = None
        self._flow_list = None
        self._flow_map = None
        self._event_mask = None
        self._accum_flow_map = None
        self._flow_warping_indices = None
        self._flow_out_mask = torch.zeros(1, 1, self.res[0], self.res[1]).to(self.device)

    @property
    def flow_map(self):
        return self._flow_map[:, -1]

    def forward_prop_flow(self, i, tref, flow):
        """
        Forward propagation of the estimated optical flow using bilinear interpolation.
        :param i: time at which the flow map to be warped is defined
        :param tref: reference time for the forward propagation
        :return warped_flow: [[batch_size x H x W x 2]] warped
        """

        # sample per-pixel optical flow
        indices = self.indices_map.view(1, 2, -1).permute(0, 2, 1).clone()
        indices_mask = self.indices_mask[..., 0].clone()
        indices_flow = get_event_flow(flow, indices.clone(), mask=indices_mask, mode=self.sampling_mode)

        # optical flow (forward) propagation
        warp_pos, warp_weights = get_interpolation(indices, i, indices_flow, tref, self.res, self.flow_scaling, round_idx=False)
        warped_flow = torch.stack([
            interpolate(warp_pos, warp_weights * indices_flow[..., 0:1], self.res),
            interpolate(warp_pos, warp_weights * indices_flow[..., 1:2], self.res)], dim=1)
        
        warp_weights = interpolate(warp_pos, warp_weights, self.res)
        warped_flow /= warp_weights[:, None] + 1e-9

        return warped_flow

    def compute_window_events(self):
        warp_pos, warp_weights = get_interpolation(
            self._event_list[..., 1:3], self._event_list[..., 0:1], 0, 0, self.res, self.flow_scaling, round_idx=True
            )

        return torch.stack([
                interpolate(warp_pos, warp_weights, self.res, self._event_list[..., -1] == 1), 
                interpolate(warp_pos, warp_weights, self.res, self._event_list[..., -1] == -1)
                ], dim=1)

    def compute_window_flow(self, mode="forward", mask=False):
        
        if mode == "forward":
            flow_map = self._flow_map.clone()
            if self.warp_mode == "iter":
                for i in range(self.passes):
                    for j in range(i):
                        warped_flow = self.forward_prop_flow(j, j + 1, flow_map[:, j])
                        # update lists
                        flow_map[:, j, ...] = warped_flow
            else:
                for i in range(self.passes - 1):
                    warped_flow = self.forward_prop_flow(i, self.passes - 1, self._flow_map[:, i])
                    # update lists
                    flow_map[:, i, ...] = warped_flow
            flow_map *= self.flow_scaling
        else:
            flow_map = self._accum_flow_map / self._flow_out_mask
            flow_map = flow_map.unsqueeze(dim=1)

        avg_flow = flow_map.sum(dim=1)
        cnt = (flow_map[:, :, 0] != 0.) | (flow_map[:, :, 1] != 0.)
        cnt = cnt.float().sum(dim=1)
        avg_flow /= cnt + 1e-9
        
        if mask:
            mask = torch.sum(self._event_mask, dim=1, keepdim=True) > 0.0
            avg_flow *= mask.float()

        return avg_flow

    def compute_window_iwe(self, tref=1, round_idx=True):
        fw_idx, fw_weights = get_interpolation(
            self._event_list[..., 1:3], self._event_list[..., 0:1], self._flow_list, tref, self.res, self.flow_scaling, round_idx=round_idx
        )
        iwe = torch.stack([
            interpolate(fw_idx.long(), fw_weights, self.res, self._event_list[..., -1] == 1),
            interpolate(fw_idx.long(), fw_weights, self.res, self._event_list[..., -1] == -1)
        ], dim=1)

        return iwe

    def call_fwl(self):
        max_ts = self.passes

        # image of (forward) warped events
        fw_IWE = self.compute_window_iwe(tref=max_ts, round_idx=True)

        # image of non-warped events
        IE = self.compute_window_events()

        # Forward Warping Loss (FWL)
        FWL = fw_IWE.sum(dim=1).var() / IE.sum(dim=1).var()
        # FWL = FWL.view((fw_IWE.shape[0]))  # one metric per batch

        return FWL, fw_IWE, IE

    def call_rast(self):
        max_ts = self.passes

        # image of (forward) warped averaged timestamps
        ts_list = self._event_list[:, :, 0:1]
        fw_idx, fw_weights = get_interpolation(
            self._event_list[..., 1:3], self._event_list[..., 0:1], self._flow_list, max_ts, self.res, self.flow_scaling, round_idx=True
        )
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, mask=self._event_list[..., -1] == 1)
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, mask=self._event_list[..., -1] == -1)
        fw_iwe_pos_ts = interpolate(
            fw_idx.long(), fw_weights * ts_list, self.res, mask=self._event_list[..., -1] == 1
        )
        fw_iwe_neg_ts = interpolate(
            fw_idx.long(), fw_weights * ts_list, self.res, mask=self._event_list[..., -1] == -1
        )
        fw_iwe_pos_ts /= fw_iwe_pos + 1e-9
        fw_iwe_neg_ts /= fw_iwe_neg + 1e-9
        fw_iwe_pos_ts = fw_iwe_pos_ts / max_ts
        fw_iwe_neg_ts = fw_iwe_neg_ts / max_ts

        # image of non-warped averaged timestamps
        zero_idx, zero_weights = get_interpolation(
            self._event_list[..., 1:3], self._event_list[..., 0:1], 0, max_ts, self.res, self.flow_scaling, round_idx=True
        )
        zero_iwe_pos = interpolate(
            zero_idx.long(), zero_weights, self.res, mask=self._event_list[..., -1] == 1
        )
        zero_iwe_neg = interpolate(
            zero_idx.long(), zero_weights, self.res, mask=self._event_list[..., -1] == -1
        )
        zero_iwe_pos_ts = interpolate(
            zero_idx.long(), zero_weights * ts_list, self.res, mask=self._event_list[..., -1] == 1
        )
        zero_iwe_neg_ts = interpolate(
            zero_idx.long(), zero_weights * ts_list, self.res, mask=self._event_list[..., -1] == -1
        )
        zero_iwe_pos_ts /= zero_iwe_pos + 1e-9
        zero_iwe_neg_ts /= zero_iwe_neg + 1e-9
        zero_iwe_pos_ts = zero_iwe_pos_ts / max_ts
        zero_iwe_neg_ts = zero_iwe_neg_ts / max_ts

        # (scaled) sum of the squares of the per-pixel and per-polarity average timestamps
        fw_iwe_pos_ts = fw_iwe_pos_ts.view(fw_iwe_pos_ts.shape[0], -1)
        fw_iwe_neg_ts = fw_iwe_neg_ts.view(fw_iwe_neg_ts.shape[0], -1)
        fw_iwe_pos_ts = torch.sum(fw_iwe_pos_ts ** 2, dim=1)
        fw_iwe_neg_ts = torch.sum(fw_iwe_neg_ts ** 2, dim=1)
        fw_ts_sum = fw_iwe_pos_ts + fw_iwe_neg_ts

        fw_nonzero_px = fw_iwe_pos + fw_iwe_neg
        fw_nonzero_px[fw_nonzero_px > 0] = 1
        fw_nonzero_px = fw_nonzero_px.view(fw_nonzero_px.shape[0], -1)
        fw_ts_sum /= torch.sum(fw_nonzero_px, dim=1)

        zero_iwe_pos_ts = zero_iwe_pos_ts.view(zero_iwe_pos_ts.shape[0], -1)
        zero_iwe_neg_ts = zero_iwe_neg_ts.view(zero_iwe_neg_ts.shape[0], -1)
        zero_iwe_pos_ts = torch.sum(zero_iwe_pos_ts ** 2, dim=1)
        zero_iwe_neg_ts = torch.sum(zero_iwe_neg_ts ** 2, dim=1)
        zero_ts_sum = zero_iwe_pos_ts + zero_iwe_neg_ts

        zero_nonzero_px = zero_iwe_pos + zero_iwe_neg
        zero_nonzero_px[zero_nonzero_px > 0] = 1
        zero_nonzero_px = zero_nonzero_px.view(zero_nonzero_px.shape[0], -1)
        zero_ts_sum /= torch.sum(zero_nonzero_px, dim=1)

        return fw_ts_sum / zero_ts_sum

    def call_aee(self):

        # convert flow
        # flow = self.compute_window_flow(mode='backward', mask=False)
        flow = self._flow_map.sum(1) * self.flow_scaling
        # flow = self._flow_map.sum(1) * self.flow_scaling
        # flow = self._flow_map.sum(1) / 4 * self.flow_scaling
        # flow = self._flow_map[:, -1] * self.flow_scaling
        # flow *= self._dt_gt.to(self.device) / self._dt_input.to(self.device)
        flow_mag = flow.pow(2).sum(1).sqrt()

        # compute AEE
        error = (flow - self._gtflow).pow(2).sum(1).sqrt()

        # AEE not computed in pixels without events
        # event_mask = self._event_mask[:, -1, :, :].bool()
        event_mask = self._event_mask.sum(1).bool()


        # AEE not computed in pixels without valid ground truth
        # gtflow_mask_x = self._gtflow[:, 0, :, :] == 0.0
        # gtflow_mask_y = self._gtflow[:, 1, :, :] == 0.0
        # gtflow_mask = gtflow_mask_x * gtflow_mask_y
        # gtflow_mask = ~gtflow_mask
        
        gtflow_mask = (self._gtflow[:, 0] != 0) | (self._gtflow[:, 1] != 0)

        # mask AEE and flow
        mask = event_mask * gtflow_mask
        mask = mask.reshape(self._flow_map[:, -1].shape[0], -1)
        error = error.view(self._flow_map[:, -1].shape[0], -1)
        flow_mag = flow_mag.view(self._flow_map[:, -1].shape[0], -1)
        error = error * mask
        flow_mag = flow_mag * mask

        # compute AEE and percentage of outliers
        num_valid_px = torch.sum(mask, dim=1)
        AEE = torch.sum(error, dim=1) / (num_valid_px + 1e-9)

        outliers = (error > 3.0) * (error > 0.05 * flow_mag)  # AEE larger than 3px and 5% of the flow magnitude
        percent_AEE = outliers.sum() / (num_valid_px + 1e-9)

        return AEE, percent_AEE


class AveragedIWE(nn.Module):
    """
    Returns an image of the per-pixel and per-polarity average number of warped events given
    an optical flow map.
    """

    def __init__(self, config, device):
        super(AveragedIWE, self).__init__()
        self.res = config["loader"]["resolution"]
        self.flow_scaling = max(config["loader"]["resolution"])
        self.batch_size = config["loader"]["batch_size"]
        self.device = device

    def forward(self, flow, event_list, pol_mask):
        """
        :param flow: [batch_size x 2 x H x W] optical flow maps
        :param event_list: [batch_size x N x 4] input events (y, x, ts, p)
        :param pol_mask: [batch_size x N x 2] per-polarity binary mask of the input events
        """

        # original location of events
        idx = event_list[:, :, 1:3].clone()
        idx[:, :, 1] *= self.res[1]  # torch.view is row-major
        idx = torch.sum(idx, dim=2, keepdim=True)

        # flow vector per input event
        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx[:, :, 1] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        # get flow for every event in the list
        flow = flow.view(flow.shape[0], 2, -1)
        event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
        event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
        event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
        event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
        event_flow = torch.cat([event_flowy, event_flowx], dim=2)

        # interpolate forward
        fw_idx, fw_weights = get_interpolation(event_list[..., 1:3], event_list[..., 0:1], event_flow, 1, self.res, self.flow_scaling, round_idx=True)

        # per-polarity image of (forward) warped events
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 1:2])
        if fw_idx.shape[1] == 0:
            return torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)

        # make sure unfeasible mappings are not considered
        pol_list = event_list[:, :, 3:4].clone()
        pol_list[pol_list < 1] = 0  # negative polarity set to 0
        pol_list[fw_weights == 0] = 2  # fake polarity to detect unfeasible mappings

        # encode unique ID for pixel location mapping (idx <-> fw_idx = m_idx)
        m_idx = torch.cat([idx.long(), fw_idx.long()], dim=2)
        m_idx[:, :, 0] *= self.res[0] * self.res[1]
        m_idx = torch.sum(m_idx, dim=2, keepdim=True)

        # encode unique ID for per-polarity pixel location mapping (pol_list <-> m_idx = pm_idx)
        pm_idx = torch.cat([pol_list.long(), m_idx.long()], dim=2)
        pm_idx[:, :, 0] *= (self.res[0] * self.res[1]) ** 2
        pm_idx = torch.sum(pm_idx, dim=2, keepdim=True)

        # number of different pixels locations from where pixels originate during warping
        # this needs to be done per batch as the number of unique indices differs
        fw_iwe_pos_contrib = torch.zeros((flow.shape[0], self.res[0] * self.res[1], 1)).to(self.device)
        fw_iwe_neg_contrib = torch.zeros((flow.shape[0], self.res[0] * self.res[1], 1)).to(self.device)
        for b in range(0, self.batch_size):

            # per-polarity unique mapping combinations
            unique_pm_idx = torch.unique(pm_idx[b, :, :], dim=0)
            unique_pm_idx = torch.cat(
                [
                    torch.div(unique_pm_idx, (self.res[0] * self.res[1]) ** 2, rounding_mode='trunc'),
                    unique_pm_idx % ((self.res[0] * self.res[1]) ** 2),
                ],
                dim=1,
            )  # (pol_idx, mapping_idx)
            unique_pm_idx = torch.cat(
                [unique_pm_idx[:, 0:1], unique_pm_idx[:, 1:2] % (self.res[0] * self.res[1])], dim=1
            )  # (pol_idx, fw_idx)
            unique_pm_idx[:, 0] *= self.res[0] * self.res[1]
            unique_pm_idx = torch.sum(unique_pm_idx, dim=1, keepdim=True)

            # per-polarity unique receiving pixels
            unique_pfw_idx, contrib_pfw = torch.unique(unique_pm_idx[:, 0], dim=0, return_counts=True)
            unique_pfw_idx = unique_pfw_idx.view((unique_pfw_idx.shape[0], 1))
            contrib_pfw = contrib_pfw.view((contrib_pfw.shape[0], 1))
            unique_pfw_idx = torch.cat(
                [torch.div(unique_pfw_idx, self.res[0] * self.res[1], rounding_mode='trunc'), unique_pfw_idx % (self.res[0] * self.res[1])],
                dim=1,
            )  # (polarity mask, fw_idx)

            # positive scatter pixel contribution
            mask_pos = unique_pfw_idx[:, 0:1].clone()
            mask_pos[mask_pos == 2] = 0  # remove unfeasible mappings
            b_fw_iwe_pos_contrib = torch.zeros((self.res[0] * self.res[1], 1)).to(self.device)
            b_fw_iwe_pos_contrib = b_fw_iwe_pos_contrib.scatter_add_(
                0, unique_pfw_idx[:, 1:2], mask_pos.float() * contrib_pfw.float()
            )

            # negative scatter pixel contribution
            mask_neg = unique_pfw_idx[:, 0:1].clone()
            mask_neg[mask_neg == 2] = 1  # remove unfeasible mappings
            mask_neg = 1 - mask_neg  # invert polarities
            b_fw_iwe_neg_contrib = torch.zeros((self.res[0] * self.res[1], 1)).to(self.device)
            b_fw_iwe_neg_contrib = b_fw_iwe_neg_contrib.scatter_add_(
                0, unique_pfw_idx[:, 1:2], mask_neg.float() * contrib_pfw.float()
            )

            # store info
            fw_iwe_pos_contrib[b, :, :] = b_fw_iwe_pos_contrib
            fw_iwe_neg_contrib[b, :, :] = b_fw_iwe_neg_contrib

        # average number of warped events per pixel
        fw_iwe_pos_contrib = fw_iwe_pos_contrib.view((flow.shape[0], 1, self.res[0], self.res[1]))
        fw_iwe_neg_contrib = fw_iwe_neg_contrib.view((flow.shape[0], 1, self.res[0], self.res[1]))
        fw_iwe_pos[fw_iwe_pos_contrib > 0] /= fw_iwe_pos_contrib[fw_iwe_pos_contrib > 0]
        fw_iwe_neg[fw_iwe_neg_contrib > 0] /= fw_iwe_neg_contrib[fw_iwe_neg_contrib > 0]

        return torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)

class Flow_Benchmark:
    def __init__(self, infer_loss_fname, config, device):
        self.config = config
        self.infer_loss_fname = infer_loss_fname
        self.fn = {}
        self.cur = {}
        self.tot = {}
        for metric in config['metrics']["method"]:
            self.fn[metric] = eval(metric)(config, device, flow_scaling=config["metrics"]["flow_scaling"])
            self.cur[metric] = {}
            self.tot[metric] = {}
        self.reset(self.cur)
        self.reset(self.tot)
        self.idx_AEE = 0

    def reset(self, seq):
        for metric in self.cur.keys():
            seq[metric]["value"] = 0
            seq[metric]["it"] = 0
            if metric == "AEE":
                seq[metric]["outliers"] = 0

    def update(self):
        for metric in self.cur.keys():
            self.tot[metric]["value"] += self.cur[metric]["value"]
            self.tot[metric]["it"] += self.cur[metric]["it"]
            if metric == "AEE":
                self.tot[metric]["outliers"] += self.cur[metric]["outliers"]

    def __call__(self, pred_flow, inputs):       
        for metric in self.fn.keys():
            self.fn[metric].event_flow_association(pred_flow, inputs)

            if self.fn[metric].num_events >= self.config["data"]["window_eval"]:
                # overwrite intermedia flow estimates with the final ones
                if self.config["loss"]["overwrite_intermediate"]:
                    self.fn[metric].overwrite_intermediate_flow(pred_flow)
                if metric == "AEE":
                    if inputs["dt_gt"] <= 0.0:
                        continue
                    self.idx_AEE += 1
                    if self.idx_AEE != np.round(1.0 / self.config["data"]["window"]):
                        continue

                # compute metric
                val_metric = self.fn[metric]()
                if metric == "AEE":
                    self.idx_AEE = 0

                # accumulate results
                for batch in range(self.config["loader"]["batch_size"]):
                    self.cur[metric]["it"] += 1
                    if metric == "AEE":
                        self.cur[metric]["value"] += val_metric[0][batch].cpu().numpy()
                        self.cur[metric]["outliers"] += val_metric[1][batch].cpu().numpy()
                    else:
                        self.cur[metric]["value"] += val_metric[batch].cpu().numpy()

    def write(self, seq_name=None):
            # store validation config and results
        results = {}
        self.update()
        with open(self.infer_loss_fname, 'a') as f:
            # f.write('perceptual loss for each step:{}\n'.format(self.loss['perceptual_loss']))
            # f.write('mse loss for each step:{}\n'.format(self.loss['mse_loss']))
            # f.write('ssim loss for each step:{}\n'.format(self.loss['ssim_loss']))
            # f.write('******************************\n')
            seq = self.cur if seq_name else self.tot
            seq_name = "whole" if not seq_name else seq_name
            for metric in seq.keys():
                results[metric] = seq[metric]["value"] / seq[metric]["it"]
                f.write(f"mean {metric} for {seq_name} sequences:{results[metric]:.3f}\n")
                if metric == "AEE":
                    results["AEE_outliers"] = seq[metric]["outliers"] / seq[metric]["it"]
                    f.write(f"mean AEE_outliers for {seq_name} sequences:{results['AEE_outliers']:.3f}\n")
        self.reset(self.cur)
        return results

if __name__ == '__main__':
    parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(parent_dir_name)
    from Tools.stream_loader import H5Dataset, StreamDataLoader
    config = {
        "resolution": [128, 128],
        "loss_window": 10000,
        "flow_regul_weight": 0.001,
        "clip_grad": 100,
        "overwrite_intermediate": False
    }
    loss1 = EventWarping(**config)

    data_config = {
        "path": "/home/wan97/Workspace/DVS/Optical_flow/TimeFlow/Datasets/UZHFPV/Optical_Flow",
        "mode": "events",
        "__mode": "events/time/frames",
        "window": 1000,
        "num_bins": 2,
        "resolution": [128, 128],
        "batch_size": 1,
        "encoding": "cnt",
        # "augmentation": ["Horizontal", "Vertical", "Polarity"],
        # "augment_prob": [0.5, 0.5, 0.5],
        "debug": False
    }
    dataset = H5Dataset(**data_config)
    loader = StreamDataLoader(dataset, num_workers=1)

    torch.manual_seed(2022)
    flow = [torch.rand((1, 2, 128, 128))]

    for item in loader:
        loss1.event_flow_association(flow, item['event_list'], item['event_cnt'])
        loss1_batch = loss1()
        print(loss1_batch)
        break
    exit()
