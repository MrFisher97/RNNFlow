import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
import stream_loader
import dsec_loader

def override(info, item):
    def dfs_dict(dic, str_list, v):
        if len(str_list) > 1:
            dfs_dict(dic[str_list[0].strip()], str_list[1:], v)
        else:
            dic[str_list[0].strip()] = v
        return True

    for term in info.split(','):
        k, v = term.split('=')
        if ('encoding' in k) or ('downsample_method' in k):
            dfs_dict(item, k.split('.'), v.strip())
        else:
            dfs_dict(item, k.split('.'), eval(v.strip()))
    return item

class Param_Tracker:
    def __init__(self):
        self.tot = 0
        self.num = 0

    def reset(self):
        self.tot = 0
        self.num = 0

    def __call__(self, val, n=1):
        self.tot += val
        self.num += n

    @property
    def avg(self):
        return self.tot / self.num if self.num > 0 else 0

class Time_Tracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.orig = time.time()
        self.rec = 0
        self.num = 0

    def start(self):
        self.orig = time.time()

    def __call__(self, n=1):
        self.num += n
        self.rec += self.elapsed

    @property
    def avg(self):
        return self.rec / self.num if self.num > 0 else self.elapsed
    
    @property
    def elapsed(self):
        return time.time() - self.orig