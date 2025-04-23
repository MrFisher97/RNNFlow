import torch

def grid_sample_values(input, h, w):
    device = input.device
    ceil = torch.stack([torch.ceil(input[0, :]), torch.ceil(input[1, :]),input[2, :]])
    floor = torch.stack([torch.floor(input[0, :]), torch.floor(input[1, :]), input[2, :]])
    z = input[2, :].clone()

    values_ipl = torch.zeros(h * w, device = device)
    weights_acc = torch.zeros(h * w, device =device)
    for x_vals in [floor[0], ceil[0]]:
        for y_vals in [floor[1], ceil[1]]:
            in_bound_mask = (x_vals < w) & (x_vals >= 0) & (y_vals < h) & (y_vals >= 0)
            weights = (1 - (input[0] - x_vals).abs()) * (1 - (input[1] - y_vals).abs())

            indices = (x_vals + w * y_vals).long()
            values_ipl.put_(indices[in_bound_mask], (z * weights)[in_bound_mask], accumulate=True)
            weights_acc.put_(indices[in_bound_mask], weights[in_bound_mask], accumulate=True)
    
    valid_mask = weights_acc.clone()
    valid_mask[valid_mask > 0] = 1
    valid_mask = valid_mask.bool().reshape([h, w])

    values_ipl = values_ipl / (weights_acc + 1e-15)
    values_ipl = values_ipl.reshape([h, w])

    return values_ipl.unsqueeze(0).clone(), valid_mask.unsqueeze(0).clone()

def forward_interpolate_tensor(flow_prev):
    flow = flow_prev.clone()
    if len(flow.shape) < 4:
        flow = flow.unsqueeze(0)
    
    b, _, h ,w = flow.shape
    device = flow.device

    dx, dy = flow[:, 0], flow[:, 1]
    y0, x0 = torch.meshgrid(torch.arange(0, h, 1), torch.arange(0, w, 1))
    x0 = torch.stack([x0] * b).to(device)
    y0 = torch.stack([y0] * b).to(device)
    
    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.flatten(start_dim=1)
    y1 = y1.flatten(start_dim=1)
    dx = dx.flatten(start_dim=1)
    dy = dy.flatten(start_dim=1)

    flow_next = torch.zeros(flow.shape, device=device)
    for i in range(b):
        flow_next[i, 0] = grid_sample_values(torch.stack([x1[i], y1[i], dx[i]]), h, w)[0]
        flow_next[i, 1] = grid_sample_values(torch.stack([x1[i], y1[i], dy[i]]), h, w)[0]
    
    return flow_next


def purge_unfeasible(x, res):
    """
    Purge unfeasible event locations by setting their interpolation weights to zero.
    :param x: location of motion compensated events
    :param res: resolution of the image space
    :return masked indices
    :return mask for interpolation weights
    """

    # mask = torch.ones(x.shape[:-1]).to(x.device)
    # mask_x = (x[..., 0] < 0) | (x[..., 0] >= res[0])
    # mask_y = (x[..., 1] < 0) | (x[..., 1] >= res[1])
    # mask[mask_x | mask_y] = 0
    # return x * mask[..., None], mask
    mask = (x[..., 0] < 0) | (x[..., 0] >= res[1])
    mask |= (x[..., 1] < 0) | (x[..., 1] >= res[0])
    return ~mask

def get_billinear_interp(pos, res):
    """
    compute the bilinear interpolation (or rounding) weights to distribute the events to the closes (integer) locations in the image space.
    :param events: [batch_size x N x 4] input events (ts, x, y, p)
    :param flow: [batch_size x N x 2] optical flow map
    :param tref: reference time toward which events are warped
    :param res: resolution of the image space
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = False)
    :return interpolated event indices
    :return interpolation weights
    """

    # get scattering indices
    left_x = torch.floor(pos[:, :, 0])
    right_x = torch.floor(pos[:, :, 0] + 1)
    top_y = torch.floor(pos[:, :, 1]) # floor 向下取整
    bot_y = torch.floor(pos[:, :, 1] + 1)

    top_left = torch.stack([left_x, top_y], dim=2)
    top_right = torch.stack([right_x, top_y], dim=2)
    bottom_left = torch.stack([left_x, bot_y], dim=2)
    bottom_right = torch.stack([right_x, bot_y], dim=2)
    bi_pos = torch.stack([top_left, top_right, bottom_left, bottom_right], dim=-2)

    return bi_pos

def get_interpolation(pos, ts, flow, tref, res, flow_scaling, round_idx=False):
    """
    Warp the input events according to the provided optical flow map and compute the bilinar interpolation
    (or rounding) weights to distribute the events to the closes (integer) locations in the image space.
    :param pos: [batch_size x N x 2] input events (x, y)
    :param ts: [batch_size x N x 1] input events (t)
    :param flow: [batch_size x N x 2] optical flow map
    :param tref: reference time toward which events are warped
    :param res: resolution of the image space
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = False)
    :return interpolated event indices
    :return interpolation weights
    """

    # event propagation
    warped_pos = pos + (tref - ts) * flow * flow_scaling

    if round_idx:
        # no bilinear interpolation
        warped_pos = torch.round(warped_pos)[..., None, :] # round 四舍五入
        weights = torch.ones(warped_pos.shape).to(pos.device)

    else:
        warped_bi_pos = get_billinear_interp(warped_pos, res)
        # get scattering interpolation weights
        weights = torch.clamp(1 - torch.abs(warped_pos.unsqueeze(2) - warped_bi_pos), min=0)
        warped_pos = warped_bi_pos

    # purge unfeasible indices
    # pos, mask = purge_unfeasible(pos, res)
    mask = purge_unfeasible(warped_pos, res)

    # make unfeasible weights zero
    weights = torch.prod(weights, dim=-1) * mask  # bilinear interpolation
    warped_pos = warped_pos * mask[..., None]

    # prepare indices
    warped_pos[..., 1] *= res[1]  # torch.view is row-major
    warped_pos = torch.sum(warped_pos, dim=-1)

    return warped_pos, weights


def interpolate(pos, weights, res, mask=None):
    """
    Create an image-like representation of the warped events.
    :param pos: [batch_size x N x (1 or 4) x 2] warped event locations
    :param weights: [batch_size x N x (1 or 4)] interpolation weights for the warped events
    :param res: resolution of the image space
    :param polarity_mask: [batch_size x N] polarity mask for the warped events (default = None)
    :return image of warped events
    """

    if mask is not None:
        weights = weights * mask[..., None]
    weights = weights.view(weights.size(0), -1)
    pos = pos.view(pos.size(0), -1)
    iwe = torch.zeros((pos.shape[0], res[0] * res[1])).to(pos.device)

    iwe = iwe.scatter_add_(1, pos.long(), weights)
    iwe = iwe.view((pos.shape[0], res[0], res[1]))
    return iwe


# def interpolate(pos, weights, res, mask=None):
#     """
#     Create an image-like representation of the warped events.
#     :param pos: [batch_size x N x (1 or 4) x 2] warped event locations
#     :param weights: [batch_size x N x (1 or 4)] interpolation weights for the warped events
#     :param res: resolution of the image space
#     :param polarity_mask: [batch_size x N] polarity mask for the warped events (default = None)
#     :return image of warped events
#     """

#     if mask is not None:
#         weights = weights * mask[..., None]
#     B = pos.shape[0]
#     iwe = torch.zeros((B, res[0] * res[1])).to(pos.device)
#     # scatter_add is non-deteministic
#     weights = weights.ravel()
#     batch = torch.arange(B).view(B, 1).expand((-1, pos.shape[1] * pos.shape[2])).ravel().long()
#     pos = pos.ravel().long()
#     iwe = iwe.index_put_((batch, pos), weights, accumulate=True)
#     iwe = iwe.view((B, res[0], res[1]))
#     return iwe


def compute_pol_iwe(events, flow, tref, res, flow_scaling=128, round_idx=True):
    """
    Create a per-polarity image of warped events given an optical flow map.
    Event timestamp needs to be normalized between 0 and 1.
    :param events: [batch_size x N x 4] input events (ts, x, y, p)
    :param flow: [batch_size x N x 2] optical flow map
    :param res: resolution of the image space
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = False)
    :return iwe: [batch_size x 2 x H x W] image of warped events
    """

    # flow vector per input event

    # get flow for every event in the list
    # flow = flow.view(flow.shape[0], 2, -1)

    # event_flowx = torch.gather(flow[:, 0], 1, flow_idx.long())  # vertical component
    # event_flowy = torch.gather(flow[:, 1], 1, flow_idx.long())  # horizontal component
    # event_flow = torch.stack([event_flowx, event_flowy], dim=-1)   # batch_size x N x 2

    # interpolate forward
    fw_idx, fw_weights = get_interpolation(events[..., 1:3], events[..., 0:1], flow, tref, res, flow_scaling, round_idx=round_idx)

    # image of (forward) warped events

    iwe = torch.stack([
            interpolate(fw_idx.long(), fw_weights, res, events[..., -1] == 1),
            interpolate(fw_idx.long(), fw_weights, res, events[..., -1] == -1)
        ], dim=1)

    return iwe

def get_flow(pos, flow, res):
    pos[..., 1] *= res[1]  # torch.view is row-major
    pos = torch.sum(pos, dim=-1)
    weight = torch.ones_like(pos)
    flow_map = torch.stack([
        interpolate(pos.long(), weight * flow[..., 0], res),
        interpolate(pos.long(), weight * flow[..., 1], res)
    ], dim=1)
    event_cnt = interpolate(pos.long(), weight, res)
    event_cnt = event_cnt[:, None].expand(-1, 2, -1, -1)
    flow_map /= event_cnt
    flow_map[event_cnt == 0] = 0

    return flow_map
