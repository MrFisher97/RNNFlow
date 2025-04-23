import numpy as np
import torch
import cv2

def ev_to_2Dmap(xs, ys, ws, size=(180, 240), accumulate=True):
    """
    Accumulate events into an image according to the weight of each event.
    ws: the weight for each event
    """
    img = np.zeros(size, dtype=np.float32)
    if accumulate:
        np.add.at(img, (ys, xs), ws)
        # img[(ys, xs)] += ws
    else:
        img[(ys, xs)] = ws
    return img

def ev_to_3Dmap(xs, ys, ts, ws, size=(3, 180, 240), accumulate=True):
    """
    Accumulate events into an image according to the weight of each event.
    ws: the weight for each event
    """
    img = np.zeros(size, dtype=np.float32)
    if accumulate:
        x0 = xs.astype(int)
        y0 = ys.astype(int)
        for xlim in [x0, x0 + 1]:
            for ylim in [y0, y0 + 1]:
                mask = (ylim < size[-2]) & (xlim < size[-1]) & (xlim >= 0) & (ylim >= 0)              
                ws = ws * (1 - abs(xlim - xs)) * (1 - abs((ylim - ys)))
                # ti, yi, xi, wi = ts[mask], ys[mask], xs[mask], ws[mask]
                ti, yi, xi, wi = ts[mask], ylim[mask], xlim[mask], ws[mask]
                np.add.at(img, (ti.astype(int), yi, xi), wi)
    else:
        img[(ts.astype(int), ys.astype(int), xs.astype(int))] = ws
    return img

def ev_to_channels(evs, size=(180, 240)):
    """
    Generate a two-channel event image containing event counters.
    """
    xs, ys, ps = evs['x'], evs['y'], evs['p']
    assert len(xs) == len(ys) == len(ps)

    return np.stack([ev_to_2Dmap(xs, ys, ps * (ps > 0), size), 
                        ev_to_2Dmap(xs, ys, - ps * (ps < 0), size)])

def ev_to_3Dcnt(evs, num_bins, size=[180, 240]):
    """
    Generate a two-channel event image containing event counters.
    """

    ts, xs, ys, ps = evs['t'], evs['x'], evs['y'], evs['p']
    assert len(xs) == len(ys) == len(ts) ==  len(ps)

    size = [num_bins, ] + size
    ts = ts / ts[-1] * num_bins * 0.99

    return np.stack([ev_to_3Dmap(xs, ys, ts, ps * (ps > 0), size), 
                    ev_to_3Dmap(xs, ys, ts, - ps * (ps < 0), size)])

def ev_to_3Dgauss(evs, num_bins, size=[180, 240]):
    ts, xs, ys, ps = evs['t'], evs['x'], evs['y'], evs['p']

    ts = ts / ts[-1]
    t_avg = ts.mean()
    t_std = ts.std() + 0.0001 if ts.shape[0] > 1 else 1

    gaussian_part1 = 1 / np.sqrt(2*np.pi) / t_std
    gaussian_part2 = - (ts - t_avg)**2 / (2 * t_std**2)
    gaussian_value_at_event = gaussian_part1 * np.exp(gaussian_part2)

    # normalizing factor
    lam = np.abs(ps).sum() / gaussian_value_at_event.sum()
    ws = np.ceil(gaussian_value_at_event * lam)
    
    num_bins //= 2 #consider the polarity as bins
    size = [num_bins, ] + size
    ts = ts * num_bins * 0.99

    ev_map = np.stack([ev_to_3Dmap(xs, ys, ts, ws * (ps > 0), size), 
                    ev_to_3Dmap(xs, ys, ts, ws * (ps < 0), size)], axis=1)

    ev_map = ev_map.reshape(-1, ev_map.shape[-2], ev_map.shape[-1])
    return ev_map

def ev_to_voxel(evs, num_bins, size=[180, 240], round_ts=False):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    """
    ts, xs, ys, ps = evs['t'], evs['x'], evs['y'], evs['p']
    assert len(xs) == len(ys) == len(ts) == len(ps)
    size = [num_bins, ] + size
    ts = ts / ts[-1] * num_bins * 0.99
    if round_ts:
        ts = np.round(ts)
    
    ind = ts.astype(np.int32)

    vox = ev_to_3Dmap(xs, ys, ind, ps * (1.0 - (ts - ind)), size)
    vox[1:] += ev_to_3Dmap(xs, ys, ind, ps * (ts - ind), size)[:-1]
    # for b_idx in range(num_bins):
    #     weights = np.clip(1.0 - np.abs(ts - b_idx), a_min=0, a_max=1)
    #     vox[b_idx] = ev_to_2Dmap(xs, ys, ps * weights, size)

    return vox

def ev_to_univoxel(evs, num_bins, size=[180, 240], round_ts=False):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    """
    ts, xs, ys, ps = evs['t'], evs['x'], evs['y'], evs['p']
    assert len(xs) == len(ys) == len(ts) == len(ps)
    size = [num_bins + 1, ] + size
    tau = ts[-1] / (num_bins + 1)

    ts = ts / tau * 0.99
    if round_ts:
        ts = np.round(ts)
    ind = ts.astype(np.int32)

    vox = ev_to_3Dmap(xs, ys, ind, ps * (ts - ind), size)
    vox[:-1] += ev_to_3Dmap(xs, ys, ind, ps * (1 - (ts - ind)), size)[1:]
    # for b_idx in range(num_bins):
    #     weights = np.clip(1.0 - np.abs(ts - b_idx), a_min=0, a_max=1)
    #     vox[b_idx] = ev_to_2Dmap(xs, ys, ps * weights, size)

    return vox[:-1]

def ev_to_voxel(evs, num_bins, size=[180, 240], round_ts=False):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    """
    ts, xs, ys, ps = evs['t'], evs['x'], evs['y'], evs['p']
    assert len(xs) == len(ys) == len(ts) == len(ps)
    size = [num_bins, ] + size
    ts = ts / ts[-1] * num_bins * 0.99
    if round_ts:
        ts = np.round(ts)
    
    ind = ts.astype(np.int32)

    vox = ev_to_3Dmap(xs, ys, ind, ps * (1.0 - (ts - ind)), size)
    vox[1:] += ev_to_3Dmap(xs, ys, ind, ps * (ts - ind), size)[:-1]
    # for b_idx in range(num_bins):
    #     weights = np.clip(1.0 - np.abs(ts - b_idx), a_min=0, a_max=1)
    #     vox[b_idx] = ev_to_2Dmap(xs, ys, ps * weights, size)

    return vox

def ev_to_timesurface(evs, num_bins, size=(180, 240)):

    ts, xs, ys, ps = evs['t'], evs['x'], evs['y'], evs['p']
    assert len(xs) == len(ys) == len(ts) == len(ps)
    ts = ts - ts[0]
    tbins = ts / ts[-1] * num_bins * (1 - 1e-6)
    size = (num_bins,) + tuple(size)
    timg = ev_to_3Dmap(xs, ys, tbins, ts, size)
    cimg = ev_to_3Dmap(xs, ys, tbins, np.ones_like(ps), size)

    timg = timg / (cimg + 1e-6)
    timg = timg / timg.max()
    return timg


def get_hot_event_mask(event_rate, idx, max_px=100, min_obvs=5, max_rate=0.8):
    """
    Returns binary mask to remove events from hot pixels.
    """

    mask = np.ones(event_rate.shape)
    if idx > min_obvs:
        for i in range(max_px):
            argmax = np.argmax(event_rate)
            index = (np.divide(argmax, event_rate.shape[1]), argmax % event_rate.shape[1])
            if event_rate[index] > max_rate:
                event_rate[index] = 0
                mask[index] = 0
            else:
                break
    return mask

def event_formatting(xs, ys, ts, ps):
    """
    Reset sequence-specific variables.
    :param xs: [N] numpy array with event x location
    :param ys: [N] numpy array with event y location
    :param ts: [N] numpy array with event timestamp
    :param ps: [N] numpy array with event polarity ([-1, 1])
    :return xs: [N] tensor with event x location
    :return ys: [N] tensor with event y location
    :return ts: [N] tensor with normalized event timestamp
    :return ps: [N] tensor with event polarity ([-1, 1])
    """

    xs = xs.astype(np.int32)
    ys = ys.astype(np.int32)
    ts = ts.astype(np.float32)
    ps = ps.astype(np.float32)
    if min(ps) == 0:
        ps = 2 * ps - 1
    ts = (ts - ts[0]) / (ts[-1] - ts[0])
    return xs, ys, ts, ps

def binary_search_array(array, x, left=None, right=None, side="left"):
    """
    Binary search through a sorted array.
    """

    left = 0 if left is None else left
    right = len(array) - 1 if right is None else right
    mid = left + (right - left) // 2

    if left > right:
        return left if side == "left" else right

    if array[mid] == x:
        return mid

    if x < array[mid]:
        return binary_search_array(array, x, left=left, right=mid - 1)

    return binary_search_array(array, x, left=mid + 1, right=right)

def delta_time(ts, window, event_idx0, event_idx1):
    floor_row = int(np.floor(ts))
    ceil_row = int(np.ceil(ts + window))
    if ceil_row - floor_row > 1:
        floor_row += ceil_row - floor_row - 1

    idx0_change = ts - floor_row
    idx1_change = ts + window - floor_row

    delta_idx = event_idx1 - event_idx0
    event_idx1 = int(event_idx0 + idx1_change * delta_idx)
    event_idx0 = int(event_idx0 + idx0_change * delta_idx)
    return event_idx0, event_idx1

def create_polarity_mask(ps):
    """
    Creates a two channel tensor that acts as a mask for the input event list.
    :param ps: [N] tensor with event polarity ([-1, 1])
    :return [N x 2] event representation
    """

    inp_pol_mask = np.stack([ps, ps])
    inp_pol_mask[0, :][inp_pol_mask[0, :] < 0] = 0
    inp_pol_mask[1, :][inp_pol_mask[1, :] > 0] = 0
    inp_pol_mask[1, :] *= -1
    return inp_pol_mask

def augment_frames(img, augmentation):
    """
    Augment APS frame with horizontal and vertical flips.
    :param img: [H x W] numpy array with APS intensity
    :param batch: batch index
    :return img: [H x W] augmented numpy array with APS intensity
    """
    if "Horizontal" in augmentation:
        img = np.flip(img, 1)
    if "Vertical" in augmentation:
        img = np.flip(img, 0)
    return img

def augment_flowmap(flowmap, augmentation):
    """
    Augment ground-truth optical flow map with horizontal and vertical flips.
    :param flowmap: [2 x H x W] numpy array with ground-truth (x, y) optical flow
    :param batch: batch index
    :return flowmap: [2 x H x W] augmented numpy array with ground-truth (x, y) optical flow
    """
    if "Horizontal" in augmentation:
        flowmap = np.flip(flowmap, 2)
        flowmap[0, :, :] *= -1.0
    if "Vertical" in augmentation:
        flowmap = np.flip(flowmap, 1)
        flowmap[1, :, :] *= -1.0
    return flowmap

def augment_events(xs, ys, ps, augmentation=["Horizontal", "Vertical", "Polarity", "VariNum"], resolution=[255, 255]):
    """
    Augment event sequence with horizontal, vertical, and polarity flips, and
    artificial event pauses.
    :return xs: [N] tensor with event x location
    :return ys: [N] tensor with event y location
    :return ps: [N] tensor with event polarity ([-1, 1])
    :param batch: batch index
    :return xs: [N] tensor with augmented event x location
    :return ys: [N] tensor with augmented event y location
    :return ps: [N] tensor with augmented event polarity ([-1, 1])
    """

    for i, mechanism in enumerate(augmentation):
        if mechanism == "Horizontal":
            xs = resolution[1] - 1 - xs
        elif mechanism == "Vertical":
            ys = resolution[0] - 1 - ys
        elif mechanism == "Polarity":
            ps *= -1
            # ts = ts[-1] - ts
        elif mechanism == 'Transpose':
            xs, ys = ys, xs

        # # shared among batch elements
        # elif (
        #     batch == 0
        #     and mechanism == "Pause"
        #     and tc_idx > config["loss"]["reconstruction_tc_idx_threshold"]
        # ):
        #     if augmentation["Pause"]:
        #         if np.random.random() < config["loader"]["augment_prob"][i][1]:
        #             self.batch_augmentation["Pause"] = False
        #     elif np.random.random() < config["loader"]["augment_prob"][i][0]:
        #             self.batch_augmentation["Pause"] = True

    return xs, ys, ps

def rectification_mapping(intrinsics, extrinsics, disparity_to_depth, res, augmentation:list=[]):
    """
    Compute the backward rectification map for the input representations.
    See https://github.com/uzh-rpg/DSEC/issues/14 for details.
    :param batch: batch index
    :return K_rect: intrinsic matrix of rectified image
    :return mapping: rectification map
    :return Q_rect: scaling matrix to convert disparity to depth
    """

    # distorted image
    K_dist = intrinsics["cam0"]["camera_matrix"]

    # rectified image
    K_rect = intrinsics["camRect0"]["camera_matrix"]
    R_rect = extrinsics["R_rect0"]
    dist_coeffs = intrinsics["cam0"]["distortion_coeffs"]

    # formatting
    K_dist = np.array([[K_dist[0], 0, K_dist[2]], [0, K_dist[1], K_dist[3]], [0, 0, 1]])
    K_rect = np.array([[K_rect[0], 0, K_rect[2]], [0, K_rect[1], K_rect[3]], [0, 0, 1]])
    R_rect = np.array(
        [
            [R_rect[0][0], R_rect[0][1], R_rect[0][2]],
            [R_rect[1][0], R_rect[1][1], R_rect[1][2]],
            [R_rect[2][0], R_rect[2][1], R_rect[2][2]],
        ]
    )
    dist_coeffs = np.array([dist_coeffs[0], dist_coeffs[1], dist_coeffs[2], dist_coeffs[3]])

    # backward mapping
    mapping = cv2.initUndistortRectifyMap(
        K_dist,
        dist_coeffs,
        R_rect,
        K_rect,
        (res[1], res[0]),
        cv2.CV_32FC2,
    )[0]

    # disparity to depth (onyl used for evaluation)
    Q_rect = disparity_to_depth["cams_03"]
    Q_rect = np.array(
        [
            [Q_rect[0][0], Q_rect[0][1], Q_rect[0][2], Q_rect[0][3]],
            [Q_rect[1][0], Q_rect[1][1], Q_rect[1][2], Q_rect[1][3]],
            [Q_rect[2][0], Q_rect[2][1], Q_rect[2][2], Q_rect[2][3]],
            [Q_rect[3][0], Q_rect[3][1], Q_rect[3][2], Q_rect[3][3]],
        ]
    ).astype(np.float32)

    if "Horizontal" in augmentation:
        K_rect[0, 2] = res[1] - 1 - K_rect[0, 2]
        mapping[:, :, 0] = res[1] - 1 - mapping[:, :, 0]
        mapping = np.flip(mapping, axis=1)
        Q_rect[0, 3] = -K_rect[0, 2]

    if "Vertical" in augmentation:
        K_rect[1, 2] = res[0] - 1 - K_rect[1, 2]
        mapping[:, :, 1] = res[0] - 1 - mapping[:, :, 1]
        mapping = np.flip(mapping, axis=0)
        Q_rect[1, 3] = -K_rect[1, 2]

    return {'K': K_rect,
            'Q': Q_rect,
            'map': mapping
            }

def rect_repr(x, rect_map):
    x = x.permute(1, 2, 0).numpy()
    x = cv2.remap(x, rect_map, None, cv2.INTER_NEAREST)
    x = torch.tensor(x)
    if x.dim() == 2:
        x = x.unsqueeze(-1)
    x = x.permute(2, 0, 1)
    return x

def custom_collate(batch, max_num_events, need_pad=False):
    """
    Collects the different event representations and stores them together in a dictionary.
    """
    batch_dict = {k:[] for k in batch[0].keys()}
    for entry in batch:
        for k, v in entry.items():
            batch_dict[k].append(v)
    for k, v in batch_dict.items():
        if type(v[0]) is torch.Tensor:
            if need_pad and k in ['event_list']:
                batch_dict[k] = torch.nn.utils.rnn.pad_sequence(v, batch_first=True, padding_value=0)
                sample_mask = torch.ones(batch_dict[k].shape[0], batch_dict[k].shape[1], dtype=torch.bool)
                if sample_mask.size(1) > max_num_events:
                    sample_idx = torch.multinomial(sample_mask.float(), max_num_events, replacement=False)
                    sample_mask[torch.arange(sample_mask.size(0))[:, None], sample_idx] = 0
                    sample_mask = ~sample_mask
            else:
                batch_dict[k] = torch.stack(v)
                if k in ['event_list']:
                    sample_mask = torch.ones(batch_dict[k].shape[0], batch_dict[k].shape[1], dtype=torch.bool)
    batch_dict['sample_mask'] = sample_mask

    # events = []
    # for i, d in enumerate(batch_dict["event_list"]):
    #     ev = np.concatenate([d, i*np.ones((len(d),1), dtype=np.float32)],1)
    #     events.append(ev)
    # batch_dict["event_list_unroll"] = torch.from_numpy(np.concatenate(events,0))
    return batch_dict