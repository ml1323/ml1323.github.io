from scipy import ndimage
import numpy as np
import torch
from torchvision import transforms
import os
import argparse


class heatmap_generation(object):
    def __init__(self, dataset, obs_len, sg_idx=None, device='cpu'):
        self.obs_len = obs_len
        self.device = device
        self.sg_idx = sg_idx
        if dataset == 'pfsd':
            self.make_heatmap = self.make_psfd_heatmap
        elif dataset == 'sdd':
            self.make_heatmap = self.make_sdd_heatmap

    def make_psfd_heatmap(self, local_ic, local_map, aug=False):
        heatmaps = []
        for i in range(len(local_ic)):
            ohm = [local_map[i, 0]]
            heat_map_traj = np.zeros((160, 160))
            for t in range(self.obs_len):
                heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                # as Y-net used variance 4 for the GT heatmap representation.
            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
            ohm.append(heat_map_traj / heat_map_traj.sum())

            if self.sg_idx is None:
                heat_map_traj = np.zeros((160, 160))
                heat_map_traj[local_ic[i, -1, 0], local_ic[i, -1, 1]] = 1
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                ohm.append(heat_map_traj)
            else:
                for t in (self.sg_idx + self.obs_len):
                    heat_map_traj = np.zeros((160, 160))
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                    heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                    ohm.append(heat_map_traj)
            heatmaps.append(np.stack(ohm))

        heatmaps = torch.tensor(np.stack(heatmaps)).float().to(self.device)
        if aug:
            degree = np.random.choice([0, 90, 180, -90])
            heatmaps = transforms.Compose([
                transforms.RandomRotation(degrees=(degree, degree))
            ])(heatmaps)

        if self.sg_idx is None:
            return heatmaps[:, :2], heatmaps[:, 2:]
        else:
            return heatmaps[:, :2], heatmaps[:, 2:], heatmaps[:, -1].unsqueeze(1)



    def make_sdd_heatmap(self, obs_len, local_ic, local_map, device='cpu', aug=False):
        heatmaps = []
        for i in range(len(local_ic)):
            ohm = [local_map[i, 0]]
            heat_map_traj = np.zeros((160, 160))
            for t in range(obs_len):
                heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                # as Y-net used variance 4 for the GT heatmap representation.
            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
            ohm.append(heat_map_traj / heat_map_traj.sum())

            if self.sg_idx is None:
                heat_map_traj = np.zeros((160, 160))
                heat_map_traj[local_ic[i, -1, 0], local_ic[i, -1, 1]] = 1
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                ohm.append(heat_map_traj)
            else:
                for t in (self.sg_idx + obs_len):
                    heat_map_traj = np.zeros((160, 160))
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                    heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                    ohm.append(heat_map_traj)
            heatmaps.append(np.stack(ohm))

        heatmaps = torch.tensor(np.stack(heatmaps)).float().to(device)
        if aug:
            degree = np.random.choice([0, 90, 180, -90])
            heatmaps = transforms.Compose([
                transforms.RandomRotation(degrees=(degree, degree))
            ])(heatmaps)

        if self.sg_idx is None:
            return heatmaps[:, :2], heatmaps[:, 2:]
        else:
            return heatmaps[:, :2], heatmaps[:, 2:], heatmaps[:, -1].unsqueeze(1)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def derivative_of(x, dt=1):

    if x[~np.isnan(x)].shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[~np.isnan(x)] = np.gradient(x[~np.isnan(x)], dt)
    return dx

def integrate_samples(v, p_0, dt=1):
    """
    Integrates deterministic samples of velocity.

    :param v: Velocity samples
    :return: Position samples
    """
    v=v.permute(1, 0, 2)
    abs_traj = torch.cumsum(v, dim=1) * dt + p_0.unsqueeze(1)
    return  abs_traj.permute((1, 0, 2))





def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    https://github.com/agrimgupta92/sgan
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    # loss = pred_traj_gt - pred_traj
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    https://github.com/agrimgupta92/sgan
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
