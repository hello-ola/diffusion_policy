from typing import Dict
import copy
import numpy as np
import torch

from threadpoolctl import threadpool_limits
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    array_to_stats, get_image_range_normalizer)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusion_policy.dataset.base_dataset import BaseImageDataset

# Recorder schema:
#   ACTION_COLUMNS = [base_x, base_y, base_theta, lift, arm,
#                     wrist_yaw, wrist_pitch, wrist_roll, grip_mm]
#   STATE_COLUMNS  = ACTION_COLUMNS + [base_vx, base_vy, base_omega]
#   action[t] = state[t+1][0:9]; final frame of each episode dropped.
BASE_SLICE = slice(0, 3)      # base_x, base_y, base_theta -- raw absolute odom
JOINT_SLICE = slice(3, 9)     # lift, arm, wrist_yaw, wrist_pitch, wrist_roll, grip_mm
VEL_SLICE = slice(9, 12)      # base_vx, base_vy, base_omega


def se2_relative(poses, anchor):
    """Express SE(2) poses in the frame of `anchor`.

    poses:  (..., 3) [x, y, theta] in odom frame
    anchor: (3,)     [x, y, theta] in odom frame
    returns (..., 3) [dx, dy, dtheta] in the anchor's body frame
    """
    x0, y0, th0 = anchor[0], anchor[1], anchor[2]
    c, s = np.cos(th0), np.sin(th0)
    ex = poses[..., 0] - x0
    ey = poses[..., 1] - y0
    dth = np.arctan2(np.sin(poses[..., 2] - th0), np.cos(poses[..., 2] - th0))
    return np.stack([c * ex + s * ey, -s * ex + c * ey, dth], axis=-1)


def se2_compose(anchor, rel):
    """Inverse of se2_relative -- turns predicted relative actions back into
    odom-frame targets at execution time."""
    x0, y0, th0 = anchor[0], anchor[1], anchor[2]
    c, s = np.cos(th0), np.sin(th0)
    dx, dy, dth = rel[..., 0], rel[..., 1], rel[..., 2]
    return np.stack([x0 + c * dx - s * dy,
                     y0 + s * dx + c * dy,
                     th0 + dth], axis=-1)


def make_masked_range_normalizer(data, frozen_std=1e-3, pct=1.0, verbose=True,
                                 names=None):
    """Per-dimension [-1, 1] range normalizer that refuses to amplify dead dimensions.

    Dimensions whose std is below `frozen_std` never actually moved during the
    demonstrations. Min-max normalizing them stretches pure encoder noise to the
    full output range, handing the policy a noise channel to fit. LinearNormalizer's
    own `range_eps` (1e-4) does not catch these -- a joint held still still drifts a
    few tenths of a millimetre, which is above that guard but far below any real
    signal. Those dims are instead mean-centred with unit scale, so they contribute
    a constant ~0 and denormalize back to the value they were pinned at.

    Live dimensions use 1st/99th percentile limits rather than true min/max, so a
    single outlier frame cannot set the scale for the whole dataset.
    """
    data = np.asarray(data, dtype=np.float32)
    stat = array_to_stats(data)
    std = data.std(axis=0)
    frozen = std < frozen_std

    lo = np.percentile(data, pct, axis=0).astype(np.float32)
    hi = np.percentile(data, 100.0 - pct, axis=0).astype(np.float32)
    rng = hi - lo
    rng[rng < 1e-8] = 1.0

    scale = (2.0 / rng).astype(np.float32)
    offset = (-1.0 - scale * lo).astype(np.float32)

    scale[frozen] = 1.0
    offset[frozen] = -stat['mean'][frozen]

    if verbose and frozen.any():
        idx = np.nonzero(frozen)[0]
        label = [(names[i] if names else str(i)) for i in idx]
        print(f"[stretch] {frozen.sum()}/{len(frozen)} dims frozen in this dataset "
              f"(std < {frozen_std}): {label}")
        print(f"[stretch]   their std: {np.array2string(std[idx], precision=6)}")
        print("[stretch]   -> mean-centred, not range-scaled (would amplify noise)")

    stat = dict(stat)
    stat['min'] = lo
    stat['max'] = hi
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat)


class StretchImageDataset(BaseImageDataset):
    """Stretch 4 recorder zarr -> diffusion policy.

    Two things happen here that the raw zarr does not do:

    1. The base action (dims 0:3, raw absolute odometry) is converted to a
       chunk-anchored SE(2) delta, anchored on the base pose at the last
       observation step.
    2. Absolute odometry is kept out of the observation entirely. Its origin is
       arbitrary at test time, so feeding it to the policy trains on noise. The
       base is represented to the policy by its measured velocity instead.
    """

    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 n_obs_steps=2,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 frozen_std=1e-3):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['head_image', 'gripper_image', 'state', 'action'])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed)

        # Only the first n_obs_steps of each OBS key are ever used (the policy
        # slices obs[:, :n_obs_steps] for the global cond), but `action` needs the
        # full horizon.
        self.key_first_k = {k: n_obs_steps
                            for k in ['head_image', 'gripper_image', 'state']}

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=self.key_first_k)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.frozen_std = frozen_std

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            key_first_k=self.key_first_k)
        val_set.train_mask = ~self.train_mask
        return val_set

    def _relative_action_population(self):
        """Every (anchor, chunk) pair the sampler can produce, as relative actions.

        The normalizer has to be fit on the distribution the policy actually sees.
        Fitting on raw absolute odometry would set the base scale from the size of
        the room rather than from how far the base moves in one chunk.
        """
        state = self.replay_buffer['state'][:]
        action = self.replay_buffer['action'][:]
        ends = self.replay_buffer.episode_ends[:]
        starts = np.concatenate([[0], ends[:-1]])

        out = []
        for s, e in zip(starts, ends):
            st_pose = state[s:e, BASE_SLICE]
            act_pose = action[s:e, BASE_SLICE]
            act_joint = action[s:e, JOINT_SLICE]
            L = e - s
            for t in range(L):
                a_idx = min(t + self.n_obs_steps - 1, L - 1)
                w = slice(t, min(t + self.horizon, L))
                rel = se2_relative(act_pose[w], st_pose[a_idx])
                out.append(np.concatenate([rel, act_joint[w]], axis=-1))
        return np.concatenate(out, axis=0).astype(np.float32)

    def get_normalizer(self, mode='limits', **kwargs):
        action_names = ['base_dx', 'base_dy', 'base_dtheta', 'lift', 'arm',
                        'wrist_yaw', 'wrist_pitch', 'wrist_roll', 'grip_mm']
        agent_names = ['lift', 'arm', 'wrist_yaw', 'wrist_pitch', 'wrist_roll',
                       'grip_mm', 'base_vx', 'base_vy', 'base_omega']

        normalizer = LinearNormalizer()
        normalizer['action'] = make_masked_range_normalizer(
            self._relative_action_population(),
            frozen_std=self.frozen_std, names=action_names)

        state = self.replay_buffer['state'][:]
        agent_pos = np.concatenate(
            [state[:, JOINT_SLICE], state[:, VEL_SLICE]], axis=-1)
        normalizer['agent_pos'] = make_masked_range_normalizer(
            agent_pos, frozen_std=self.frozen_std, names=agent_names)

        normalizer['head_image'] = get_image_range_normalizer()
        normalizer['gripper_image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        state = sample['state'].astype(np.float32)
        action = sample['action'].astype(np.float32)

        # Anchor the whole action chunk on the base pose at the last obs step.
        anchor = state[self.n_obs_steps - 1, BASE_SLICE]
        base_rel = se2_relative(action[:, BASE_SLICE], anchor)
        action_out = np.concatenate(
            [base_rel, action[:, JOINT_SLICE]], axis=-1).astype(np.float32)

        agent_pos = np.concatenate(
            [state[:, JOINT_SLICE], state[:, VEL_SLICE]], axis=-1).astype(np.float32)

        # Beyond n_obs_steps the obs arrays are unloaded filler
        # so they must be sliced away here rather than handed to the policy.
        To = self.n_obs_steps
        return {
            'obs': {
                'head_image': np.moveaxis(sample['head_image'][:To], -1, 1).astype(np.float32) / 255.,
                'gripper_image': np.moveaxis(sample['gripper_image'][:To], -1, 1).astype(np.float32) / 255.,
                'agent_pos': agent_pos[:To],
            },
            'action': action_out,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # stop each worker's BLAS from spawning a full thread pool
        threadpool_limits(1)
        sample = self.sampler.sample_sequence(idx)
        return dict_apply(self._sample_to_data(sample), torch.from_numpy)


def test():
    import os
    zarr_path = os.path.expanduser(
        '~/stretch4_policy_recorder/stretch_test.zarr')
    ds = StretchImageDataset(zarr_path, horizon=16, pad_before=1, pad_after=7,
                             n_obs_steps=2, val_ratio=0.125)
    print('len', len(ds), 'episodes', ds.replay_buffer.n_episodes)
    b = ds[0]
    for k, v in b['obs'].items():
        print(' obs', k, tuple(v.shape), v.dtype, float(v.min()), float(v.max()))
    print(' action', tuple(b['action'].shape), b['action'].dtype)

    n = ds.get_normalizer()
    pop = ds._relative_action_population()
    na = n['action'].normalize(pop)
    print('normalized action min/max per dim:')
    print(' min', np.array2string(na.min(0).values.detach().numpy(), precision=3))
    print(' max', np.array2string(na.max(0).values.detach().numpy(), precision=3))

    # SE(2) round trip
    rng = np.random.default_rng(0)
    poses = rng.normal(size=(32, 3)); anchor = rng.normal(size=3)
    back = se2_compose(anchor, se2_relative(poses, anchor))
    err = np.abs(back[:, :2] - poses[:, :2]).max()
    dth = np.abs(np.arctan2(np.sin(back[:, 2] - poses[:, 2]),
                            np.cos(back[:, 2] - poses[:, 2]))).max()
    print(f'se2 round trip: xy {err:.2e}  theta {dth:.2e}')


if __name__ == '__main__':
    test()
