import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def override(info, item):
    def dfs_dict(dic, str_list, v):
        if len(str_list) > 1:
            dfs_dict(dic[str_list[0].strip()], str_list[1:], v)
        else:
            dic[str_list[0].strip()] = v
        return True

    for term in info.split(','):
        k, v = term.split('=')
        dfs_dict(item, k.split('.'), eval(v.strip()))
    return item

class Param_Tracker:
    def __init__(self):
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
        self.start = time.time()
        self.num = 0
    
    def reset(self):
        self.start = time.time()
        self.num = 0

    def __call__(self, n=1):
        self.num += n

    @property
    def avg(self):
        return self.elapsed / self.num if self.num > 0 else self.elapsed
    
    @property
    def elapsed(self):
        return time.time() - self.start