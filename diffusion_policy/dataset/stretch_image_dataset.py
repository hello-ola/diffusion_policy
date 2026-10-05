from typing import Dict
import copy
import os
import numpy as np
import torch
import zarr

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

# Zarr written by stretch4_to_zarr. Columns are read by name from
# the zarr attrs; this is just the expected layout:
#   state     (9|10)  base_vx, base_vy, base_omega, lift, arm, wrist_yaw, wrist_pitch,
#                     wrist_roll, grip_mm [, grip_effort]   -- never odometry. The action's
#                     order (stretch4_policy_recorder's POLICY_STATE_COLUMNS); the names are
#                     saved in the checkpoint (cfg.task.agent_pos_columns) for the robot.
#   base_odom (3)     base_x, base_y, base_theta: the anchor for pose actions only
#   action    (9|10)  base(3) at t+K, lift..grip_mm at t+K [, grip_effort at t+K]
#                     (K = attrs.lookahead, 3 by default)
#                     base(3) = base_x/y/theta   (base_action: pose)
#                             = base_vx/vy/omega (base_action: velocity)
SCHEMA = 'stretch4 zarr v2'
BASE_ACTIONS = ('pose', 'velocity')
BASE_ACTION_COLUMNS = {
    'pose': ['base_x', 'base_y', 'base_theta'],
    'velocity': ['base_vx', 'base_vy', 'base_omega'],
}
ODOM_COLUMNS = BASE_ACTION_COLUMNS['pose']
BASE = slice(0, 3)            # base dims of the action
# Which joints are held
from stretch4_policy_recorder.common.held_joints import (  # noqa: E402
    JOINTS, decide_held_joints, format_hold_report)
# A live dim whose true extremes normalize beyond this gets a warning.
OUTLIER_WARN = 3.0


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

    Live dimensions are scaled from the `pct`..`100-pct` percentiles. pct=0 is true
    min/max, which the action needs: the scheduler's clip_sample clips predictions to
    [-1, 1], so an action beyond the scale could never be produced at inference. The
    observation is never clipped, so it can use pct=1 and ignore stray frames.
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

    # Live dims whose true extremes land far outside [-1, 1]: a few frames move much
    # more than the rest. Usually a joint used in one or two episodes -- see
    # decide_held_joints, which holds those before they reach this point.
    n_lo, n_hi = stat['min'] * scale + offset, stat['max'] * scale + offset
    extreme = np.maximum(np.abs(n_lo), np.abs(n_hi))
    outliers = ~frozen & (extreme > OUTLIER_WARN)
    if verbose and outliers.any():
        for i in np.nonzero(outliers)[0]:
            name = names[i] if names else str(i)
            hint = (" If it is used in only a few episodes, consider task.hold.hold_joints."
                    if name in JOINTS else
                    " Heavy-tailed but real; fine as long as this is an observation (never"
                    " clipped).")
            print(f"[stretch] NOTE {name}: normalizes to [{n_lo[i]:.1f}, {n_hi[i]:.1f}], "
                  f"the {pct:g}-{100 - pct:g}% range is [{lo[i]:.5g}, {hi[i]:.5g}] but the "
                  f"data spans [{stat['min'][i]:.5g}, {stat['max'][i]:.5g}].{hint}")

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
    """Stretch 4 zarr (stretch4_to_zarr) -> diffusion policy.

    - Cameras are the `type: rgb` keys of shape_meta, so any subset of the zarr's
      `<cam>_image` arrays (one head, both heads, gripper) can be trained on.
    - agent_pos is the zarr's state as stored: joints + base velocity, plus grip_effort
      when `include_grip_effort`.
    - base_action 'pose': the base action dims (absolute odometry at t+1) become a
      chunk-anchored SE(2) delta, anchored on base_odom at the last observation step.
      base_action 'velocity': the base dims are body-frame velocities, used as they are.
    - Joints moved in too few episodes are held (decide_held_joints): replaced by a
      constant in both action and agent_pos, and kept still by the robot. The decision
      is printed, written to <run dir>/held_joints.json, and saved in the checkpoint's
      cfg.task.held_joints.
    """

    def __init__(self,
                 shape_meta,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 n_obs_steps=2,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 base_action='pose',
                 include_grip_effort=False,
                 hold=None,
                 frozen_std=1e-3):
        super().__init__()
        zarr_path = os.path.expanduser(zarr_path)
        group = zarr.open(zarr_path, 'r')
        attrs = dict(group.attrs)
        if attrs.get('schema') != SCHEMA:
            raise ValueError(
                f"{zarr_path} has schema {attrs.get('schema')!r}, expected {SCHEMA!r}. "
                "Reconvert it with the current stretch4_to_zarr (older zarrs put odometry "
                "in the state).")

        print(f"[stretch] {zarr_path}: base_action={attrs.get('base_action')}, "
              f"lookahead={attrs.get('lookahead', 1)} (action[t] targets state[t+K]), "
              f"cameras {[k for k in group['data'].array_keys() if k.endswith('_image')]}")

        # --- cameras ---
        self.rgb_keys = [k for k, v in shape_meta['obs'].items() if v.get('type') == 'rgb']
        have = sorted(k for k in group['data'].array_keys() if k.endswith('_image'))
        missing = [k for k in self.rgb_keys if k not in have]
        if missing:
            raise ValueError(f"shape_meta wants {missing}, but {zarr_path} only has {have}")
        for k in self.rgb_keys:
            want = tuple(shape_meta['obs'][k]['shape'])
            h, w, c = group['data'][k].shape[1:]
            if want != (c, h, w):
                raise ValueError(f"shape_meta {k} is {list(want)}, but the zarr's frames are "
                                 f"{[c, h, w]} (CHW)")

        # --- columns ---
        if base_action not in BASE_ACTIONS:
            raise ValueError(f"base_action must be one of {BASE_ACTIONS}, got {base_action!r}")
        if attrs.get('base_action') != base_action:
            raise ValueError(
                f"task base_action is {base_action!r}, but {zarr_path} was converted with "
                f"--base-action {attrs.get('base_action')!r}. Set task.base_action to match "
                "(the robot reads it from the checkpoint to decode the base).")
        state_cols = list(attrs['state_columns'])
        action_cols = list(attrs['action_columns'])
        leaked = [c for c in ODOM_COLUMNS if c in state_cols]
        if leaked:
            raise ValueError(f"{zarr_path} state holds odometry {leaked}; reconvert it")
        if action_cols[BASE] != BASE_ACTION_COLUMNS[base_action]:
            raise ValueError(f"action columns {action_cols[BASE]} are not "
                             f"{BASE_ACTION_COLUMNS[base_action]} for base_action {base_action!r}")
        if include_grip_effort and not ('grip_effort' in state_cols
                                        and 'grip_effort' in action_cols):
            raise ValueError(f"include_grip_effort is true, but {zarr_path} has no grip_effort "
                             "column; reconvert with stretch4_to_zarr --grip-effort")
        keep = (lambda c: include_grip_effort or c != 'grip_effort')
        self.state_index = [i for i, c in enumerate(state_cols) if keep(c)]
        self.action_index = [i for i, c in enumerate(action_cols) if keep(c)]
        self.agent_names = [state_cols[i] for i in self.state_index]
        self.action_names = [action_cols[i] for i in self.action_index]
        if base_action == 'pose':
            self.action_names[BASE] = ['base_dx', 'base_dy', 'base_dtheta']

        for key, width, names in (
                ('obs.agent_pos', shape_meta['obs']['agent_pos']['shape'][0], self.agent_names),
                ('action', shape_meta['action']['shape'][0], self.action_names)):
            if width != len(names):
                raise ValueError(
                    f"shape_meta {key} is [{width}], but include_grip_effort="
                    f"{include_grip_effort} gives {len(names)} columns: {names}")

        # --- buffer and sampler ---
        self.low_dim_keys = ['state', 'action'] + (['base_odom'] if base_action == 'pose' else [])
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=self.rgb_keys + self.low_dim_keys)

        # --- held joints ---
        hold = dict(hold or {})
        self.held_joints, self.hold_report = decide_held_joints(
            self.replay_buffer['action'][:], action_cols, self.replay_buffer.episode_ends[:],
            episode_names=attrs.get('episodes'),
            min_episode_frac=float(hold.get('min_episode_frac', 0.1)),
            min_motion=dict(hold.get('min_motion') or {}),
            hold_joints=list(hold.get('hold_joints') or []),
            keep_joints=list(hold.get('keep_joints') or []),
            auto=bool(hold.get('auto', True)))
        print(format_hold_report(self.hold_report))
        # (column in the output action, column in agent_pos, constant) per held joint
        self._held_cols = [(self.action_names.index(j), self.agent_names.index(j), h['value'])
                           for j, h in self.held_joints.items()]

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed)

        # Only the first n_obs_steps of each OBS key are ever used (the policy
        # slices obs[:, :n_obs_steps] for the global cond, and the pose anchor is
        # base_odom[n_obs_steps-1]), but `action` needs the full horizon.
        self.key_first_k = {k: n_obs_steps for k in self.rgb_keys + self.low_dim_keys
                            if k != 'action'}

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
        self.base_action = base_action
        self.frozen_std = frozen_std

    def _hold(self, action=None, agent_pos=None):
        """Overwrite held joints with their constant, in place (see decide_held_joints)."""
        for a_col, s_col, value in self._held_cols:
            if action is not None:
                action[..., a_col] = value
            if agent_pos is not None:
                agent_pos[..., s_col] = value

    def write_hold_report(self, path):
        """The hold decisions, with the statistics behind them, as JSON."""
        import json
        with open(path, 'w') as f:
            json.dump({'held_joints': self.held_joints, 'joints': self.hold_report}, f, indent=2)

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

    def _action_population(self):
        """The actions as the policy sees them, for fitting the normalizer.

        Pose: every (anchor, chunk) pair the sampler can produce, as relative actions.
        Fitting on raw absolute odometry would set the base scale from the size of the
        room rather than from how far the base moves in one chunk. Velocity: the
        actions as stored, since nothing is re-anchored.
        """
        action = self.replay_buffer['action'][:][:, self.action_index]
        self._hold(action=action)
        if self.base_action == 'velocity':
            return action.astype(np.float32)

        odom = self.replay_buffer['base_odom'][:]
        ends = self.replay_buffer.episode_ends[:]
        starts = np.concatenate([[0], ends[:-1]])
        out = []
        for s, e in zip(starts, ends):
            ep_odom, ep_action = odom[s:e], action[s:e]
            L = e - s
            for t in range(L):
                a_idx = min(t + self.n_obs_steps - 1, L - 1)
                w = slice(t, min(t + self.horizon, L))
                rel = se2_relative(ep_action[w, BASE], ep_odom[a_idx])
                out.append(np.concatenate([rel, ep_action[w, 3:]], axis=-1))
        return np.concatenate(out, axis=0).astype(np.float32)

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        # true min/max (pct=0): clip_sample clips predictions to [-1, 1]
        normalizer['action'] = make_masked_range_normalizer(
            self._action_population(), pct=0.0,
            frozen_std=self.frozen_std, names=self.action_names)

        agent_pos = self.replay_buffer['state'][:][:, self.state_index]
        self._hold(agent_pos=agent_pos)
        normalizer['agent_pos'] = make_masked_range_normalizer(
            agent_pos, frozen_std=self.frozen_std, names=self.agent_names)

        for k in self.rgb_keys:
            normalizer[k] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # Beyond n_obs_steps the obs arrays are unloaded filler
        # so they must be sliced away here rather than handed to the policy.
        To = self.n_obs_steps
        action = sample['action'][:, self.action_index].astype(np.float32)
        if self.base_action == 'pose':
            # Anchor the whole action chunk on the base pose at the last obs step.
            anchor = sample['base_odom'][To - 1].astype(np.float64)
            action[:, BASE] = se2_relative(action[:, BASE].astype(np.float64), anchor)

        obs = {k: np.moveaxis(sample[k][:To], -1, 1).astype(np.float32) / 255.
               for k in self.rgb_keys}
        obs['agent_pos'] = sample['state'][:To, self.state_index].astype(np.float32)
        self._hold(action=action, agent_pos=obs['agent_pos'])
        return {'obs': obs, 'action': action}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # stop each worker's BLAS from spawning a full thread pool
        threadpool_limits(1)
        sample = self.sampler.sample_sequence(idx)
        return dict_apply(self._sample_to_data(sample), torch.from_numpy)


def test(zarr_path, base_action='pose', include_grip_effort=False):
    """python -m diffusion_policy.dataset.stretch_image_dataset <zarr> [pose|velocity] [effort]"""
    group = zarr.open(os.path.expanduser(zarr_path), 'r')
    attrs = dict(group.attrs)
    n_state = len(attrs['state_columns']) - (0 if include_grip_effort else
                                             int('grip_effort' in attrs['state_columns']))
    obs = {k: {'shape': [group['data'][k].shape[3], *group['data'][k].shape[1:3]], 'type': 'rgb'}
           for k in group['data'].array_keys() if k.endswith('_image')}
    obs['agent_pos'] = {'shape': [n_state], 'type': 'low_dim'}
    shape_meta = {'obs': obs, 'action': {'shape': [n_state]}}
    ds = StretchImageDataset(shape_meta, zarr_path, horizon=48, pad_before=3, pad_after=23,
                             n_obs_steps=4, val_ratio=0.05, base_action=base_action,
                             include_grip_effort=include_grip_effort)
    print('len', len(ds), 'episodes', ds.replay_buffer.n_episodes)
    b = ds[0]
    for k, v in b['obs'].items():
        print(' obs', k, tuple(v.shape), v.dtype, float(v.min()), float(v.max()))
    print(' action', tuple(b['action'].shape), b['action'].dtype, ds.action_names)
    if base_action == 'pose':
        # the action at the anchor step is the next frame, so its delta is one step of motion
        print(' base delta at the anchor step', b['action'][ds.n_obs_steps - 1, :3].numpy())

    n = ds.get_normalizer()
    na = n['action'].normalize(ds._action_population())
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
    import sys
    test(sys.argv[1], *(sys.argv[2:3] or ['pose']), include_grip_effort='effort' in sys.argv[3:])
