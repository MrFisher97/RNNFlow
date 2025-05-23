import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import visdom
import imageio

class Visualization:
    """
    Utility class for the visualization and storage of rendered image-like representation
    of multiple elements of the optical flow estimation and image reconstruction pipeline.
    """

    def __init__(self, resolution: int, path_results=None,**kwargs):
        self.img_idx = 0
        self.px = resolution + 100
        self.color_scheme = "green_red"  # gray / blue_red / green_red

        if path_results is not None:
            self.store_dir = os.path.join(path_results, "results")
            if os.path.exists(self.store_dir):
                i = 1
                while os.path.exists(self.store_dir + "_" + str(i)):
                    i += 1
                self.store_dir = self.store_dir + "_" + str(i)
            os.makedirs(self.store_dir, exist_ok=True)
            self.store_file = None

    def update(self, visulize_inp, flow=None, iwe=None, events_window=None, masked_window_flow=None, iwe_window=None):
        """
        Live visualization.
        :param visulize_inp: dataloader dictionary
        :param flow: [batch_size x 2 x H x W] optical flow map
        :param iwe: [batch_size x 1 x H x W] image of warped events
        """

        events = visulize_inp.get("event_cnt", None)
        frames = visulize_inp.get("frames", None)
        gtflow = visulize_inp.get("gtflow", None)
        height = events.shape[2]
        width = events.shape[3]

        # input events
        events = events.detach()
        events_npy = events.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
        events_npy = np.flipud(events_npy)
        cv2.namedWindow("Input Events", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Input Events", int(self.px), int(self.px))
        cv2.imshow("Input Events", self.events_to_image(events_npy))

        # input events
        if events_window is not None:
            events_window = events_window.detach()
            events_window_npy = events_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
            cv2.namedWindow("Input Events - Eval window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Input Events - Eval window", int(self.px), int(self.px))
            cv2.imshow("Input Events - Eval window", self.events_to_image(events_window_npy))

        # input frames
        if frames is not None:
            frame_image = np.zeros((height, 2 * width))
            frames = frames.detach()
            frames_npy = frames.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            frame_image[:height, 0:width] = frames_npy[:, :, 0] / 255.0
            frame_image[:height, width : 2 * width] = frames_npy[:, :, 1] / 255.0
            cv2.namedWindow("Input Frames (Prev/Curr)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Input Frames (Prev/Curr)", int(2 * self.px), int(self.px))
            cv2.imshow("Input Frames (Prev/Curr)", frame_image)

        # optical flow
        if flow is not None:
            flow = flow.detach()
            flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            flow_npy = self.flow_to_image(flow_npy)
            flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Estimated Flow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Estimated Flow", int(self.px), int(self.px))
            cv2.imshow("Estimated Flow", flow_npy)

        # optical flow
        if masked_window_flow is not None:
            masked_window_flow = masked_window_flow.detach()
            masked_window_flow_npy = masked_window_flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            masked_window_flow_npy = self.flow_to_image(masked_window_flow_npy)
            masked_window_flow_npy = cv2.cvtColor(masked_window_flow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Estimated Flow - Eval window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Estimated Flow - Eval window", int(self.px), int(self.px))
            cv2.imshow("Estimated Flow - Eval window", masked_window_flow_npy)

        # ground-truth optical flow
        if gtflow is not None:
            gtflow = gtflow.detach()
            gtflow_npy = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            gtflow_npy = self.flow_to_image(gtflow_npy)
            gtflow_npy = cv2.cvtColor(gtflow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Ground-truth Flow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Ground-truth Flow", int(self.px), int(self.px))
            cv2.imshow("Ground-truth Flow", gtflow_npy)

        # image of warped events
        if iwe is not None:
            iwe = iwe.detach()
            iwe_npy = iwe.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            iwe_npy = self.events_to_image(iwe_npy)
            iwe_npy = np.flipud(iwe_npy)
            cv2.namedWindow("Image of Warped Events", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image of Warped Events", int(self.px), int(self.px))
            cv2.imshow("Image of Warped Events", iwe_npy)

        # image of warped events - evaluation window
        if iwe_window is not None:
            iwe_window = iwe_window.detach()
            iwe_window_npy = iwe_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            iwe_window_npy = self.events_to_image(iwe_window_npy)
            cv2.namedWindow("Image of Warped Events - Eval window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image of Warped Events - Eval window", int(self.px), int(self.px))
            cv2.imshow("Image of Warped Events - Eval window", iwe_window_npy)

        cv2.waitKey(1)

    def store(self, 
              visulize_inp, 
              sequence=None, 
              flow=None, 
              iwe=None, 
              events_window=None, 
              masked_window_flow=None, 
              iwe_window=None, 
              ts=None, 
              flow_info=None,
              warp_info=None,
              flow_sub=None):
        """
        Store rendered images.
        :param visulize_inp: dataloader dictionary
        :param flow: [batch_size x 2 x H x W] optical flow map
        :param iwe: [batch_size x 1 x H x W] image of warped events
        :param sequence: filename of the event sequence under analysis
        :param ts: timestamp associated with rendered files (default = None)
        """

        events = visulize_inp.get("event_cnt", None)
        frames = visulize_inp.get("frames", None)
        gtflow = visulize_inp.get("gtflow", None)
        height = events.shape[2]
        width = events.shape[3]

        # check if new sequence
        path_to = self.store_dir + "/" + sequence + "/"
        if not os.path.exists(path_to):
            os.makedirs(path_to)
            os.makedirs(path_to + "events/")
            os.makedirs(path_to + "events_window/")
            os.makedirs(path_to + "flow/")
            os.makedirs(path_to + "flow_window/")
            os.makedirs(path_to + "gtflow/")
            os.makedirs(path_to + "frames/")
            os.makedirs(path_to + "iwe/")
            os.makedirs(path_to + "iwe_window/")
            os.makedirs(path_to + "submission/")
            if self.store_file is not None:
                self.store_file.close()
            self.store_file = open(path_to + "timestamps.txt", "w")
            self.img_idx = 0          

        # input events
        event_image = np.zeros((height, width))
        events = events.detach()
        events_npy = events.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
        event_image = self.events_to_image(events_npy)
        # event_image = np.flipud(event_image)
        filename = path_to + "events/ev_%09d.png" % self.img_idx
        cv2.imwrite(filename, event_image * 255)

        # input events
        if events_window is not None:
            events_window = events_window.detach()
            events_window_npy = events_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
            events_window_npy = self.events_to_image(events_window_npy)
            filename = path_to + "events_window/%09d.png" % self.img_idx
            cv2.imwrite(filename, events_window_npy * 255)

        # input frames
        if frames is not None:
            frames = frames.detach()
            frames_npy = frames.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            filename = path_to + "frames/f_%09d.png" % self.img_idx
            cv2.imwrite(filename, frames_npy[:, :, 1])

        # optical flow
        if flow is not None:
            flow = flow.detach()
            flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            flow_npy = self.flow_to_image(flow_npy)
            # flow_bgr = flow_npy[..., (2, 1, 0)]
            flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
            filename = path_to + "flow/flow_%09d.png" % self.img_idx
            cv2.putText(flow_npy, flow_info, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imwrite(filename, flow_npy)
            # imageio.imsave(filename, flow_bgr.astype('uint8'))


        # optical flow
        if masked_window_flow is not None:
            masked_window_flow = masked_window_flow.detach()
            masked_window_flow_npy = masked_window_flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            masked_window_flow_npy = self.flow_to_image(masked_window_flow_npy)
            masked_window_flow_npy = cv2.cvtColor(masked_window_flow_npy, cv2.COLOR_RGB2BGR)
            filename = path_to + "flow_window/floww_%09d.png" % self.img_idx
            cv2.imwrite(filename, masked_window_flow_npy)

        # ground-truth optical flow
        if gtflow is not None:
            gtflow = gtflow.detach()
            gtflow_npy = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            gtflow_npy = self.flow_to_image(gtflow_npy)
            gtflow_npy = cv2.cvtColor(gtflow_npy, cv2.COLOR_RGB2BGR)
            filename = path_to + "gtflow/gtflow_%09d.png" % self.img_idx
            cv2.imwrite(filename, gtflow_npy)

        # image of warped events
        if iwe is not None:
            iwe = iwe.detach()
            iwe_npy = iwe.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            iwe_npy = self.events_to_image(iwe_npy).copy()
            filename = path_to + "iwe/iwe_%09d.png" % self.img_idx
            cv2.rectangle(iwe_npy, (0, 0), (0 + 80, 0 + 15), (1., 1., 1.), -1)
            cv2.putText(iwe_npy, warp_info, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.imwrite(filename, iwe_npy * 255)

        # image of warped events - evaluation window
        if iwe_window is not None:
            iwe_window = iwe_window.detach()
            iwe_window_npy = iwe_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            iwe_window_npy = self.events_to_image(iwe_window_npy)
            filename = path_to + "iwe_window/iwew_%09d.png" % self.img_idx
            cv2.imwrite(filename, iwe_window_npy * 255)

        if flow_sub is not None:
            flow_sub = flow_sub.detach()
            flow_sub = flow_sub.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))    
            flow_sub = flow_sub * 128 + 2**15
            flow_sub = flow_sub.astype(np.uint16)[..., (1, 0)]
            flow_sub = np.pad(flow_sub, ((0, 0), (0, 0), (0, 1)), constant_values=0)
            flow_sub = np.flip(flow_sub, axis=-1)
            filename = path_to + "submission/%09d.png" % self.img_idx
            cv2.imwrite(filename, flow_sub)

        # store timestamps
        if ts is not None:
            self.store_file.write(str(ts) + "\n")
            self.store_file.flush()

        self.img_idx += 1
        cv2.waitKey(1)

    @staticmethod
    def flow_to_image(flows):
        """
        Use the optical flow color scheme from the supplementary materials of the paper 'Back to Event
        Basics: Self-Supervised Image Reconstruction for Event Cameras via Photometric Constancy',
        Paredes-Valles et al., CVPR'21.
        :param flow: [H x W x 2] optical flow
        :return flow_rgb: [H x W x 3] color-encoded optical flow
        """
        # flows = np.stack((flow_x, flow_y), axis=2)
        # mag = np.linalg.norm(flows, axis=2) ** 0.5
        # ang = np.arctan2(flows[..., 1], flows[..., 0])
        # ang[ang < 0] += np.pi * 2
        # ang = ang / np.pi / 2.
        # # ang *= 180. / np.pi / 2.

        # hsv = np.zeros([flows.shape[0], flows.shape[1], 3])
        # hsv[:, :, 0] = ang
        # hsv[:, :, 1] = 1
        # # hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # hsv[:, :, 2] = (mag - mag.min()) / (mag - mag.min()).max()

        # flow_vis = matplotlib.colors.hsv_to_rgb(hsv) * 255
        # flow_vis = flow_vis.astype(np.uint8)
        # # flow_vis = flow_rgb[..., (2, 1, 0)] # rgb -> bgr
        # # flow_rgb = cv2.cvtColor(hsv * 255, cv2.COLOR_HSV2RGB)
        # return flow_vis
        
        mag = np.linalg.norm(flows, axis=2) **0.5
        ang = np.arctan2(flows[..., 0], flows[..., 1]) + np.pi
        ang *= 180. / np.pi / 2.

        hsv = np.zeros([flows.shape[0], flows.shape[1], 3], dtype=np.uint8)
        hsv[:, :, 0] = ang
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = cv2.normalize(mag.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)

        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return flow_rgb

        # flow_y, flow_x = flows[..., 1], flows[..., 1]
        # mag = np.linalg.norm(flows, axis=2)
        # min_mag = np.min(mag)
        # mag_range = np.max(mag) - min_mag

        # ang = np.arctan2(flow_y, flow_x) + np.pi
        # ang *= 1.0 / np.pi / 2.0

        # hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3])
        # hsv[:, :, 0] = ang
        # hsv[:, :, 1] = 1.0
        # hsv[:, :, 2] = mag - min_mag
        # if mag_range != 0.0:
        #     hsv[:, :, 2] /= mag_range

        # flow_rgb = matplotlib.colors.hsv_to_rgb(hsv)
        # return (255 * flow_rgb).astype(np.uint8)

    @staticmethod
    def minmax_norm(x):
        """
        Robust min-max normalization.
        :param x: [H x W x 1]
        :return x: [H x W x 1] normalized x
        """
        den = np.percentile(x, 99) - np.percentile(x, 1)
        if den != 0:
            x = (x - np.percentile(x, 1)) / den
        return np.clip(x, 0, 1)

    @staticmethod
    def events_to_image(event_cnt, color_scheme="green_red"):
        """
        Visualize the input events.
        :param event_cnt: [batch_size x 2 x H x W] per-pixel and per-polarity event count
        :param color_scheme: green_red/gray
        :return event_image: [H x W x 3] color-coded event image
        """
        pos = event_cnt[:, :, 0]
        neg = event_cnt[:, :, 1]
        pos_max = np.percentile(pos, 99)
        pos_min = np.percentile(pos, 1)
        neg_max = np.percentile(neg, 99)
        neg_min = np.percentile(neg, 1)
        max = pos_max if pos_max > neg_max else neg_max

        if pos_min != max:
            pos = (pos - pos_min) / (max - pos_min)
        if neg_min != max:
            neg = (neg - neg_min) / (max - neg_min)

        pos = np.clip(pos, 0, 1)
        neg = np.clip(neg, 0, 1)

        event_image = np.ones((event_cnt.shape[0], event_cnt.shape[1]))
        if color_scheme == "gray":
            event_image *= 0.5
            pos *= 0.5
            neg *= -0.5
            event_image += pos + neg

        elif color_scheme == "green_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            event_image *= 0
            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            event_image[:, :, 0][mask_pos] = 0
            event_image[:, :, 1][mask_pos] = pos[mask_pos]
            event_image[:, :, 2][mask_pos * mask_not_neg] = 0
            event_image[:, :, 0][mask_neg] = 0
            event_image[:, :, 2][mask_neg] = neg[mask_neg]
            event_image[:, :, 1][mask_neg * mask_not_pos] = 0

            event_image[mask_not_pos * mask_not_neg] = 1

            event_image = event_image[..., (1, 0, 2)] # bgr -> gbr
        return event_image


def vis_activity(activity, activity_log):
    # start of new sequence
    if activity_log is None:
        plt.close("activity")
        activity_log = []

    # update log
    activity_log.append(activity)
    df = pd.DataFrame(activity_log)

    # retrieves fig if it exists
    fig = plt.figure("activity")
    # make axis if it doesn't exist
    if not fig.axes:
        ax = fig.add_subplot()
    else:
        ax = fig.axes[0]
    lines = ax.lines

    # plot data
    if not lines:
        for name, data in df.iteritems():
            ax.plot(data.index.to_numpy(), data.to_numpy(), label=name)
        ax.grid()
        ax.legend()
        ax.set_xlabel("step")
        ax.set_ylabel("fraction of nonzero outputs")
        plt.show(block=False)
    else:
        for line in lines:
            label = line.get_label()
            line.set_data(df[label].index.to_numpy(), df[label].to_numpy())

    # update figure
    fig.canvas.draw()
    ax.relim()
    ax.autoscale_view(True, True, True)
    fig.canvas.flush_events()

    return activity_log

import torch

def standard(data, dim=(-2, -1)):
    max_data, min_data = torch.amax(data, dim=dim, keepdim=True), torch.amin(data, dim=dim, keepdim=True)
    delta = max_data - min_data
    delta[delta == 0] = 1e9
    data = (data - min_data) / delta
    return data

class VisdomPlotter:
    def __init__(self, env, port):
        self.plotter = visdom.Visdom(env = env, port = port)
    
    def vis_curve(self, X, Y, win=None):
        self.plotter.line(X, Y, win=win, update='append')

    def vis_event(self, data, if_standard=False, win=None, title=None):
        if if_standard: 
            data = standard(data)
        if len(data.shape) == 3:
            image = np.zeros((3, *data.shape[-2:]))
            image[:2] = data
        else:
            image = data
        if title is None:
            title = win
        self.plotter.image(image, win=win, opts=dict(title=title, ))

    def vis_flow(self, flow, win=None, title=None):
        if title is None:
            title = win
        vis_flow = Visualization.flow_to_image(flow)
        self.plotter.image(vis_flow.transpose(2, 0, 1), win=win, opts=dict(title=title, ))

    def vis_event_list(self, data, if_standard=False, win=None, title=None):
        data = data[:4]
        if if_standard: 
            data = standard(data)
        images = np.zeros((data.shape[0], 3, *data.shape[-2:]))
        if len(data.shape) == 3:
            images[:, 0] = data
        elif len(data.shape) == 4:
            images[:, :2] = data
        if title is None:
            title = win
        self.plotter.images(images, win=win, nrow=4, opts=dict(title=title, ))
    
    def vis_flow_list(self, flow, win=None, title=None):
        vis_flow = []
        flow = flow[:4].permute(0, 2, 3, 1)
        if title is None:
            title = win
        for i in flow:
            vis_flow.append(Visualization.flow_to_image(i))
        self.plotter.images(np.array(vis_flow).transpose(0, 3, 1, 2), win=win, nrow=4, opts=dict(title=win, ))
    
    def step(self, event_cnt, flow, iwe, loss):
        self.vis_flow_list(flow, win='pred_flow')
        self.vis_event_list(event_cnt, if_standard=True, win='raw')
        self.vis_event_list(iwe, if_standard=True, win='iwe',  title=f'iwe:{loss:.3f}')

