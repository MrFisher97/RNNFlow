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

import time
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

import data_func

def split_batch_size(batch_size, num_workers):
    """Returns a list of batch_size

    Args:
        batch_size (int): total batch size
        num_workers (int): number of workers
    """
    num_workers = min(num_workers, batch_size)
    split_size = batch_size // num_workers
    total_size = 0
    split_sizes = [split_size] * (num_workers - 1)
    split_sizes += [batch_size - sum(split_sizes)]
    return split_sizes

class StreamDataset(IterableDataset):
    """Stream Dataset
    An Iterable Dataset zipping a group of iterables streams together.

    Args:
        stream_list (list): list of streams (path/ metadata)
        streamer (object): an iterator (user defined)
        batch_size (int): total batch size
        padding_mode (str): "zeros" "data" or "none", see "get_zip" function
        padding_value (object): padding value
    """
    def __init__(self,
                stream_list,
                streamer,
                batch_size,
                padding_mode,
                padding_value,
                pos,
                mutex,
                num_actives,
                ):

        self.stream_list = stream_list
        self.batch_size = batch_size
        self.streamer = streamer
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        assert padding_mode in ['zeros', 'data']
        self.pos = pos
        self.mutex = mutex
        self.num_actives = num_actives
        self.cnt = 0
        self._set_seed()

    def shuffle(self):
        random.shuffle(self.stream_list)

    def _set_seed(self):
        """ so that data is different along threads and epochs"""
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        seed = int(time.time()) + worker_id
        np.random.seed(seed)
        random.seed(seed)

    def init_position(self):
        self.mutex.acquire()
        self.pos.value = 0
        self.num_actives.value = 0
        self.mutex.release()
    
    def increment_pos(self):
        self.mutex.acquire()
        pos = self.pos.value
        stream = self.stream_list[pos%len(self.stream_list)]
        self.pos.value = pos + 1
        self.mutex.release()
        return stream

    def get_value(self, iterators, i, actives):
        done = False
        while not done:
            try:
                if actives[i] or self.padding_mode == 'data':
                    value = next(iterators[i])
                    assert value is not None
                elif self.padding_mode == 'zeros':
                    value = self.padding_value
                done = True
            except StopIteration:
                self.mutex.acquire()
                if actives[i] and (self.pos.value >= len(self.stream_list)):
                    self.num_actives.value -= 1
                actives[i] = 1 * (self.pos.value < len(self.stream_list))
                self.mutex.release()
                stream = self.increment_pos()
                if self.padding_mode == 'data' or actives[i]:
                    assert stream is not None, self.pos.value
                    iterators[i] = iter(self.streamer(stream))
        return value

    def __len__(self):
        return len(self.stream_list)
    
    def __iter__(self):
        """Iterates over stream files

        Note: Here we use a mutex (WIP, pytest not working!)

        Note: Here the scheduling of iterable is done at the beginning.
        Instead User can change this code to map lazily iterables.
        """
        assert self.mutex, "Not initialize parallize"

        #initialization this should be done in worker_init_fnx
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0

        num_workers = 1 if worker is None else worker.num_workers
        split_sizes = split_batch_size(self.batch_size, num_workers)
        worker = torch.utils.data.get_worker_info()
        worker_id = int(worker.id) if worker is not None else 0
        split_size = split_sizes[worker_id]

        if len(self) < split_size:
            print('worker#', worker_id, ': Stopping... Number of streams < split_size')
            raise StopIteration

        """
        Just-in-time mapping
        The scheduling is done as we iterate.

        EDIT 9/7/2021: The position in the stream is shared accross workers
        This allows us to avoid the non ideal pre-iteration splitting of the dataset
        """

        iterators = []
        for i in range(split_size):
            stream = self.increment_pos()
            stream = iter(self.streamer(stream))
            iterators.append(stream)

        actives = [1 for _ in range(len(iterators))]
        _num_actives = sum(actives)
        self.mutex.acquire()
        self.num_actives.value += _num_actives
        self.mutex.release()

        while True:
            values = []
            for i in range(len(iterators)):
                values.append(self.get_value(iterators, i, actives))
            if self.num_actives.value:
                yield tuple(values), worker_id
            else:
                break

class StreamDataLoader(object):
    """StreamDataLoader

    Wraps around the DataLoader to handle the asynchronous batches.
    We now handle one single list of streams read from multiple workers with a mutex.

    Args:
        iterator_fun (lambda): function to create one stream
        batch_size (int): number of streams read at the same time
        num_workers (int): number of workers
        collate_fn (function): function to collate batch parts
        padding_mode (str): "data" or "zeros", what to do when all streams have been read but you still but one thread of streaming needs to output something
        padded_value (object): object or None
    """
    def __init__(self, 
                files,
                iterator_fun,
                batch_size=1,
                padding_mode='data',
                padding_value=None,
                shuffle=True,
                num_workers=1,
                collate_fn=data_func.custom_collate, 
                ):
        mutex = multiprocessing.Lock()
        pos = multiprocessing.Value('i', 0)
        num_actives = multiprocessing.Value('i', 0)
        dataset = StreamDataset(files, iterator_fun, batch_size, padding_mode, padding_value, pos, num_actives, mutex)

        self.dataset = dataset
        num_workers = min(dataset.batch_size, num_workers)
        assert isinstance(dataset, StreamDataset)
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            collate_fn=lambda x: x,
            pin_memory=True,
            drop_last=False)
        self.collate_fn = collate_fn
        self.num_workers = max(1, num_workers)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.dataloader.dataset.shuffle()
        self.dataloader.dataset.init_position()

        cache = [deque([]) for _ in range(self.num_workers)]
        for data in self.dataloader:
            data, worker_id = data
            cache[worker_id].append(data)
            num = sum([len(v) > 0 for v in cache])
            if num == self.num_workers:
                batch = [item.popleft() for item in cache]
                batch = list(chain.from_iterable(iter(batch)))
                 # Check if batch is all padding_value, do not yield
                all_pad = all([item == self.dataset.padding_value for item in batch])
                if all_pad:
                    continue
                batch = self.collate_fn(batch)
                yield batch

        # Empty remaining cache
        # Assert no value is a true value
        for fifo in cache:
            if not len(fifo):
                continue
            while fifo:
                # print(fifo)
                item = fifo.pop()[0]
                # assert item == self.dataset.padding_value, 'code is broken, cache contained real data'

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

class H5Stream(object):
    def __init__(self,
                file_name,
                idx_list,
                events,
                maps,
                mode,
                window, 
                resolution,
                orig_resolution,
                num_bins,
                encoding,
                augmentation,
                augment_prob,
                predict_load,
                predict_dir,
                round_ts,
                **kwargs):

        self.file = h5py.File(file_name, "r")
        self.fname = file_name
        self.sequence_name = file_name.split("/")[-1].split(".")[0]

        self.idx_list = idx_list
        self.events = events
        self.maps = maps
        self.mode = mode
        self.window = window
        self.resolution = resolution
        self.orig_resolution = orig_resolution
        self.num_bins = num_bins
        self.encoding = encoding
        self.predict_load = predict_load
        self.predict_dir = predict_dir
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
        return frames

    def load_flow(self, file, maps, cur_ts):
        idx = int(np.ceil(cur_ts + self.window))
        flowmap = file[self.mode][maps.names[idx]][:]
        flowmap = data_func.augment_flowmap(flowmap, self.augmentation)
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
        ts = self.events['t'][idx0:idx1]
        ps = self.events['p'][idx0:idx1]
        # ts -= self.events['t0']  # sequence starting at t0 = 0

        # handle case with very few events
        # if xs.shape[0] <= 100:
        #     xs, ys, ts, ps = np.split(np.empty([4 * int(1e3), 0]), 4)

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
            event_cnt = data_func.ev_to_channels(xs, ys, ps, self.resolution)
            timesurface = data_func.ev_to_timesurface(xs, ys, ts, ps, self.num_bins, self.resolution)
            event_mask = (event_cnt[0] + event_cnt[1]) > 0
            event_mask = event_mask

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
            '''
            Only for 'time' mode, choosing the fixed number events
            '''
            if self.mode == "time":
                if xs.shape[0] <= 3e3:
                    xs = np.pad(xs, pad_width=(0, int(3e3) - xs.shape[0]), mode='reflect')
                    ys = np.pad(ys, pad_width=(0, int(3e3) - ys.shape[0]), mode='reflect')
                    ts = np.pad(ts, pad_width=(0, int(3e3) - ts.shape[0]), mode='reflect')
                    ps = np.pad(ps, pad_width=(0, int(3e3) - ps.shape[0]), mode='constant', constant_values=0)
                randind = np.random.choice(xs.shape[0], size=int(3e3), replace=False)
                xs, ys, ts, ps = xs[randind], ys[randind], ts[randind], ps[randind]
            event_list = np.stack([ts, xs, ys, ps], axis=-1).astype(np.float32)

            output = {
                'cur_ts': cur_ts,
                'ts': seq_last_ts,
                'name': self.sequence_name,
                'idx': idx,
                'dt_gt': torch.from_numpy(dt_gt),
                'dt_input': torch.from_numpy(dt_input),
                'file_name': self.fname,
                'event_list': torch.from_numpy(event_list),
                'event_mask': torch.from_numpy(event_mask).float(),
                'event_cnt': torch.from_numpy(event_cnt),
            }

            if self.mode == "images":
                output['frames'] = torch.from_numpy(frames)
            elif self.mode in ["flow_dt1", "flow_dt4"]:
                output['gtflow'] = torch.from_numpy(flow_map)
            
            if self.encoding == 'cnt':
                output['input'] = torch.from_numpy(event_cnt)
            elif self.encoding == 'timesurface':
                output['input'] = torch.from_numpy(timesurface)
            elif self.encoding == 'mixture':
                output['input'] = torch.from_numpy(np.concatenate([event_cnt, timesurface]))
            elif self.encoding == 'list':
                output['input'] = torch.from_numpy(event_list)

             # update window
            cur_ts += self.window
            yield output

class H5Dataloader(StreamDataLoader):
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
        self.files = []
        self.events = {}
        self.idx_list = {}
        self.maps = {}

        if 'VariNum' in augmentation:
            vari = (1 - augment_prob[augmentation.index('VariNum')] * np.random.random())
            window = int(window * vari)
        print(f'sample events: {window}')

        for root, dirs, files in os.walk(path):
            for file in files:
                # if file.endswith("shapes_6dof.h5"):
                if file.endswith(".h5"):
                    fname = os.path.join(root, file)
                    self.get_event_info(fname, mode, window)
                    if debug and len(self.files) == batch_size:
                        break

        def iterator_func(file_name):
            events = self.events[file_name]
            idx_list = self.idx_list[file_name]
            maps = self.maps[file_name]
            return H5Stream(file_name, idx_list, events, maps, mode, window, resolution, orig_resolution, num_bins, encoding, augmentation, augment_prob, predict_load, predict_dir, round_ts)
            
        super().__init__(self.files, iterator_func, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    def get_event_info(self, fname, mode, window):
        '''
        统一单位为us
        UZHFPV 单位为 s
        '''

        if ('UZHFPV' in fname) or ('MVSEC' in fname):
            unit = 1e6
        else:
            unit = 1

        file = h5py.File(fname, "r")
        cur_ts, last_ts = 0, 0
        maps = None

        if "ts" in file['events'].keys():
            kn = {'t': 'events/ts', 'x': 'events/xs', 'y': 'events/ys', 'p': 'events/ps'}
        else:
            kn = {'t': 'events/t', 'x': 'events/x', 'y': 'events/y', 'p': 'events/p'}

        if 't0' in file.attrs.keys():
            t0 = file.attrs['t0']
        else:
            t0 = file[kn['t']][0]

        if mode in ["images", "flow_dt1", "flow_dt4"]:
            maps = Map()
            file[mode].visititems(maps)
            maps.ts = (maps.ts - t0) * unit
            last_ts = len(maps.ts)
        elif mode == "time":
            last_ts = file[kn['t']][-1] - t0
            last_ts *= unit
        else:
            last_ts = len(file[kn['t']])

        events= {'x':file[kn['x']],
                'y':file[kn['y']],
                't':(file[kn['t']] - t0) * unit,
                'p':file[kn['p']],}
        idx_list = []
        
        ms_to_idx = file['events'].get('ms_to_idx', None)
        while self.check_seq(mode, cur_ts, last_ts, window):
            idx0, idx1 = self.get_event_index(mode, events['t'], cur_ts, window, maps, ms_to_idx)
            if mode == 'time':
                if (idx1 - idx0) > 100:
                    idx_list.append((idx0, idx1))
            else:
                idx_list.append((idx0, idx1))
            cur_ts += window

        self.idx_list[fname] = idx_list
        self.events[fname] = events
        self.maps[fname] = maps
        self.files.append(fname)

        # if 'indoor_flying3_data' in fname:
        #     st = 200
        #     self.idx[fname] = self.idx[fname][st:]
        #     self.maps[fname].names = self.maps[fname].names[st:]
        #     self.maps[fname].ts = self.maps[fname].ts[st:]

    def get_event_index(self, mode, ts, cur_ts, window=0, maps=None, ms_to_idx=None):
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
            if ms_to_idx is None:
                event_idx0 = data_func.binary_search_array(ts, cur_ts)
                event_idx1 = data_func.binary_search_array(ts, cur_ts + window)
            else:
                event_idx0 = ms_to_idx[cur_ts // 1000]
                event_idx1 = ms_to_idx[(cur_ts + window) // 1000]
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

    def check_seq(self, mode, cur_ts, last_ts, window):
        return (mode in ["images", "flow_dt1", "flow_dt4"]
                    and int(np.ceil(cur_ts + window)) < last_ts) \
                or (mode in ["time", "events"]
                    and (cur_ts + window) < last_ts)
    

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