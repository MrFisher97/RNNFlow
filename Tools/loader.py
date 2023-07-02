"""
The StreamLoader is a class built on top of DataLoader,
that fuses batches together so batches are always temporally
coherent.

Here we use a different strategy than the one described in
https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd

We just return the torch worker's id every batch, and create a fifo per worker on the main
thread side.
"""
import multiprocessing
import os
import random
from collections import deque
from itertools import chain

import h5py
import numpy as np
import torch
from torch.utils.data import ChainDataset, IterableDataset

import data_func

class Map:
    """
    Utility class for reading the APS frames/ Optical flow maps encoded in the HDF5 files.
    """

    def __init__(self):
        self.ts = []
        self.names = []
        self.event_idx = []

    def __call__(self, name, h5obj):
        if hasattr(h5obj, "dtype") and name not in self.names:
            self.names += [name]
            self.ts += [h5obj.attrs["timestamp"]]
            if 'event_idx' in h5obj.attrs.keys():
                self.event_idx += [h5obj.attrs['event_idx']]

def get_event_info(fname, mode, window):
    file = h5py.File(fname, "r")
    cur_ts, last_ts = 0, 0
    maps = None
    t0 = file.attrs.get("t0", 0)
    if "ts" in file['events'].keys():
        kn = {'t': 'events/ts', 'x': 'events/xs', 'y': 'events/ys', 'p': 'events/ps'}
    else:
        kn = {'t': 'events/t', 'x': 'events/x', 'y': 'events/y', 'p': 'events/p'}

    if mode in ["images", "flow_dt1", "flow_dt4"]:
        maps = Map()
        file[mode].visititems(maps)
        last_ts = len(maps.ts)
    elif mode == "time":
        last_ts = file[kn['t']][-1] - t0
    else:
        last_ts = len(file[kn['t']])

    events= {'x':file[kn['x']],
            'y':file[kn['y']],
            't':file[kn['t']],
            'p':file[kn['p']],}
    idx_list = []
    
    ms_to_idx = file['events'].get('ms_to_idx', None)
    while check_seq(mode, cur_ts, last_ts, window):
        idx0, idx1 = get_event_index(mode, events['t'], t0, cur_ts, window, maps, ms_to_idx)
        idx_list.append((idx0, idx1))
        cur_ts += window

    return idx_list, events, maps

    # if 'indoor_flying3_data' in fname:
    #     st = 200
    #     self.idx[fname] = self.idx[fname][st:]
    #     self.maps[fname].names = self.maps[fname].names[st:]
    #     self.maps[fname].ts = self.maps[fname].ts[st:]

def get_event_index(mode, ts, t0, cur_ts, window=0, maps=None, ms_to_idx=None):
    """
    Get all the event indices to be used for reading.
    :param batch: batch index
    :param window: input window
    :return event_idx: event index
    """

    event_idx0 = None
    event_idx1 = None
    if mode == "events":
        event_idx0 = cur_ts
        event_idx1 = cur_ts + int(window)
    elif mode == "time":
        if ms_to_idx is not None:
            event_idx0 = ms_to_idx[(cur_ts + t0) // 1000]
            event_idx1 = ms_to_idx[(cur_ts + t0 + window) // 1000]
        else:
            event_idx0 = data_func.binary_search_array(ts, cur_ts + t0)
            event_idx1 = data_func.binary_search_array(ts, cur_ts + t0 + window)
    elif mode in ["images", "flow_dt1", "flow_dt4"]:                
        idx0 = int(np.floor(cur_ts))
        idx1 = int(np.ceil(cur_ts + window))
        if window < 1.0 and idx1 - idx0 > 1:
            idx0 += idx1 - idx0 - 1
        
        if len(maps.event_idx) > 0:
            event_idx0 = maps.ts[idx0]
            event_idx1 = maps.ts[idx1]
        else:
            event_idx0 = data_func.binary_search_array(ts, maps.ts[idx0])
            event_idx1 = data_func.binary_search_array(ts, maps.ts[idx1])
            
        if window < 1.0:
            event_idx0, event_idx1 = data_func.delta_time(cur_ts, window, event_idx0, event_idx1)
    else:
        print("DataLoader error: Unknown mode.")
        raise AttributeError
    
    return event_idx0, event_idx1

def check_seq(mode, cur_ts, last_ts, window):
    return (mode in ["images", "flow_dt1", "flow_dt4"]
                and int(np.ceil(cur_ts + window)) < last_ts) \
            or (mode in ["time", "events"]
                and (cur_ts + window) < last_ts)


class H5_Stream(IterableDataset):
    def __init__(self,
                fname,
                mode,
                window, 
                resolution,
                orig_resolution,
                num_bins,
                encoding,
                augmentation,
                augment_prob,
                round_ts,
                **kwargs):
        super(IterableDataset).__init__()

        self.fname = fname
        self.sequence_name = fname.split("/")[-1].split(".")[0]
        self.idx_list, self.events, self.maps = get_event_info(fname, mode, window)
        
        self.mode = mode
        self.window = window
        self.resolution = resolution
        self.orig_resolution = orig_resolution
        self.num_bins = num_bins
        self.encoding = encoding
        self.round_ts = round_ts

        self.augmentation = []
        for i, mechanism in enumerate(augmentation):
            if np.random.random() < augment_prob[i]:
                self.augmentation.append(mechanism)


    def hot_filter(self, batch, event_voxel, event_cnt, event_mask):
        hot_mask = self.create_hot_mask(event_cnt, batch)
        hot_mask_voxel = torch.stack([hot_mask] * self.num_bins, axis=2).permute(2, 0, 1)
        hot_mask_cnt = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
        event_voxel = event_voxel * hot_mask_voxel
        event_cnt = event_cnt * hot_mask_cnt
        event_mask *= hot_mask.view((1, hot_mask.shape[0], hot_mask.shape[1]))
        return event_voxel, event_cnt, event_mask

    def load_frames(self, file, maps, cur_ts):
        curr_idx = cur_ts
        next_idx = int(np.ceil(cur_ts + self.window))

        frames = np.zeros((2, self.resolution[1], self.resolution[0]))
        img0 = file["images"][maps.names[curr_idx]][:]
        img1 = file["images"][maps.names[next_idx]][:]
        frames[0, :, :] = data_func.augment_frames(img0, self.augmentation)
        frames[1, :, :] = data_func.augment_frames(img1, self.augmentation)
        frames = torch.from_numpy(frames.astype(np.uint8))
        return frames

    def load_flow(self, file, maps, cur_ts):
        idx = int(np.ceil(cur_ts + self.window))
        flowmap = file[self.mode][maps.names[idx]][:]
        flowmap = data_func.augment_flowmap(flowmap, self.augmentation)
        flowmap = torch.from_numpy(flowmap.copy())
        if idx > 0:
            dt_gt = maps.ts[idx] - maps.ts[idx - 1]
        return flowmap, dt_gt

    def load_events(self, cur_idx):
        """
        Get all the events in between two indices.
        :param file: file to read from
        :param idx0: start index
        :param idx1: end index
        :return xs: [N] numpy array with event x location
        :return ys: [N] numpy array with event y location
        :return ts: [N] numpy array with event timestamp
        :return ps: [N] numpy array with event polarity ([-1, 1])
        """
        idx0, idx1 = cur_idx[0], cur_idx[1]
        ys = self.events['y'][idx0:idx1]
        xs = self.events['x'][idx0:idx1]
        ts = self.events['t'][idx0:idx1] - self.file.attrs.get("t0", 0)
        ps = self.events['p'][idx0:idx1]
        # ts -= self.events['t0']  # sequence starting at t0 = 0

        # handle case with very few events
        if xs.shape[0] <= 10:
            xs, ys, ts, ps = np.split(np.empty([40, 0]), 4)

        # event formatting and timestamp normalization
        dt_input = np.asarray(0.0)
        if ts.shape[0] > 0:
            dt_input = np.asarray(ts[-1] - ts[0], dtype=np.float32)
        
        last_ts = ts[-1]
        xs, ys, ts, ps = data_func.event_formatting(xs, ys, ts, ps)

        # data augmentation
        xs, ys, ps = data_func.augment_events(xs, ys, ps, self.augmentation, self.resolution)

        return xs, ys, ts, ps, dt_input, last_ts

    def __iter__(self):  
        cur_ts = 0

        for cur_idx in self.idx_list:
            # load events
            xs, ys, ts, ps, dt_input, seq_last_ts = self.load_events(cur_idx)

            event_list = torch.stack([ts, xs, ys, ps], dim=-1)
            event_cnt = data_func.ev_to_channels(xs, ys, ps, self.resolution)
            timesurface = data_func.ev_to_timesurface(xs, ys, ts, ps, self.num_bins, self.resolution)
            event_mask = (event_cnt[0] + event_cnt[1]) > 0
            event_mask = event_mask.float()

            # # hot pixel removal
            # if self.config["hot_filter"]["enabled"]:
            #     event_voxel, event_cnt, event_mask = self.hot_filter(batch, event_voxel, event_cnt, event_mask)

            idx = '0'
            if self.mode == "images":
                # frames = self.load_frames(self.file, self.maps, cur_ts)
                frames = None
                idx = self.maps.names[int(np.ceil(cur_ts + self.window))][-6:]

            dt_gt = 0.0
            if self.mode in ["flow_dt1", "flow_dt4"]:
                flow_map, dt_gt = self.load_flow(self.file, self.maps, cur_ts)
                idx = self.maps.names[int(np.ceil(cur_ts + self.window))][-6:]
                # frames = self.load_frames(file, maps, cur_ts)
            dt_gt = np.asarray(dt_gt)

            # prepare output
            output = {
                'cur_ts': cur_ts,
                'ts': seq_last_ts,
                'name': self.sequence_name,
                'idx': idx,
                'dt_gt': torch.from_numpy(dt_gt),
                'dt_input': torch.from_numpy(dt_input),
                'file_name': self.fname,
                'event_list': event_list,
                'event_mask': event_mask,
                'event_cnt': event_cnt,
            }

            if self.mode == "images":
                output['frames'] = frames
            elif self.mode in ["flow_dt1", "flow_dt4"]:
                output['gtflow'] = flow_map
            
            if self.encoding == 'cnt':
                output['input'] = event_cnt
            elif self.encoding == 'timesurface':
                output['input'] = timesurface
            elif self.encoding == 'mixture':
                output['input'] = torch.cat([event_cnt, timesurface])

             # update window
            cur_ts += self.window
            yield output

class H5_ChainDataset(ChainDataset):
    def __init__(self, 
                path, mode, window, 
                resolution=[255, 255],
                orig_resolution=None,
                debug=False, 
                num_bins=2, 
                batch_size=1,
                num_workers=4,
                encoding='cnt',
                augmentation=[],
                augment_prob=[],
                predict_load=False,
                predict_dir=None,
                round_ts=False,
                shuffle=False,
                **kwargs):
        # input event sequences
        super(H5_ChainDataset).__init__()
        self.files = []
        self.events = {}
        self.idx_list = {}
        self.maps = {}
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".h5"):
                    fname = os.path.join(root, file)
                    # vari = 1
                    # if 'VariNum' in augmentation:
                    #     vari = (1 - augment_prob[augmentation.index('VariNum')] * np.random.random())
                    self.get_event_info(fname, mode, window)
                    if debug and len(self.files) == batch_size:
                        break

        def iterator_func(file_name):
            events = self.events[file_name]
            idx_list = self.idx_list[file_name]
            maps = self.maps[file_name]
            return H5Stream(file_name, idx_list, events, maps, mode, window, resolution, orig_resolution, num_bins, encoding, augmentation, augment_prob, predict_load, predict_dir, round_ts)
            
        super().__init__(self.files, iterator_func, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

   

if __name__ == '__main__':
    # data = [[10, 11, 12, 13],
    #         [20, 21, 22],
    #         [42, 43],
    #         [90],
    #         # [100]
    #         ]
    # def temp(x):
    #     for t in x:
    #         yield t
    # # dataset = StreamDataset(data, temp, batch_size=4)
    # # dataset = H5Loader()
    # # dataset=None
    # loader = StreamDataLoader(data, temp, 4, collate_fn=lambda x: x)
    # for i in loader:
    #     print(list(i))
    # file = h5py.File('/home/wan97/Workspace/Dataset/DVS/ssl_E2VID/UZHFPV/Optical_Flow/indoor_forward_3_davis_with_gt_0.h5', 'r')
    # print(file['events'].keys())

    cfg = {
        "path": "Datasets/DSEC/train/h5_128/",
        "mode": "events",
        "__mode": "events/time/frames",
        "window": 3000,
        "seq_len": 5,
        "num_bins": 2,
        "resolution": [128, 128],
        "orig_resolution": [128, 128],
        "_orig_resolution": [480, 640],
        "batch_size": 8,
        "encoding": "timesurface",
        "augmentation": ["Horizontal", "Vertical", "Transpose"],
        "augment_prob": [0.5, 0.5, 0.5],
        "debug": False
    }

    loader = H5Dataloader(**cfg)

    for i, date in enumerate(loader):
        print(i, loader.dataset.pos.value, '/', len(loader.dataset),
            int(100 * loader.dataset.pos.value / len(loader.dataset)), '%', end='\r')