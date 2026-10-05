"""StretchImageDataset on a tiny stretch4_to_zarr-layout zarr: columns by name in the shared
order, and held joints decided by stretch4_policy_recorder's shared decide_held_joints."""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
held_joints = pytest.importorskip("stretch4_policy_recorder.common.held_joints")

from diffusion_policy.dataset import stretch_image_dataset as S

COLUMNS = ["base_vx", "base_vy", "base_omega", "lift", "arm",
           "wrist_yaw", "wrist_pitch", "wrist_roll", "grip_mm"]
N_EPS, LEN = 12, 10


def make_zarr(path):
    rng = np.random.default_rng(0)
    n = N_EPS * LEN
    state = rng.normal(size=(n, 9)).astype(np.float32)
    state[:, COLUMNS.index("grip_mm")] = 40.0 + rng.normal(0, 0.01, n)   # never moves
    root = zarr.open(str(path), "w")
    data = root.create_group("data")
    data["state"] = state
    data["action"] = state.copy()
    data["head_left_image"] = np.zeros((n, 8, 8, 3), np.uint8)
    root.create_group("meta")["episode_ends"] = np.arange(LEN, n + 1, LEN, dtype=np.int64)
    root.attrs.update({"schema": S.SCHEMA, "base_action": "velocity", "lookahead": 3,
                       "state_columns": COLUMNS, "action_columns": COLUMNS})


def test_columns_and_held_joints(tmp_path):
    make_zarr(tmp_path / "d.zarr")
    shape_meta = {"obs": {"head_left_image": {"shape": [3, 8, 8], "type": "rgb"},
                          "agent_pos": {"shape": [9], "type": "low_dim"}},
                  "action": {"shape": [9]}}
    ds = S.StretchImageDataset(shape_meta, str(tmp_path / "d.zarr"), horizon=4, pad_before=1,
                               pad_after=1, n_obs_steps=2, base_action="velocity")
    assert ds.agent_names == ds.action_names == COLUMNS
    assert S.decide_held_joints is held_joints.decide_held_joints   # one implementation
    assert list(ds.held_joints) == ["grip_mm"]
    b = ds[0]
    assert np.ptp(b["obs"]["agent_pos"][:, COLUMNS.index("grip_mm")].numpy()) == 0
