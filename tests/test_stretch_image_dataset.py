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
                       "state_columns": COLUMNS, "action_columns": COLUMNS,
                       "gripper_tool": "stretch_gripper"})


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


@pytest.mark.parametrize("tag", [None, "parallel_gripper"])
def test_gripper_tool_from_the_zarr(tmp_path, tag):
    # the workspace saves it as cfg.task.image_preproc.gripper_tool for the robot's check;
    # a zarr that does not say which gripper is refused
    make_zarr(tmp_path / "d.zarr")
    attrs = zarr.open(str(tmp_path / "d.zarr"), "r+").attrs
    if tag:
        attrs["gripper_tool"] = tag
    else:
        del attrs["gripper_tool"]
    shape_meta = {"obs": {"head_left_image": {"shape": [3, 8, 8], "type": "rgb"},
                          "agent_pos": {"shape": [9], "type": "low_dim"}},
                  "action": {"shape": [9]}}
    make = lambda: S.StretchImageDataset(shape_meta, str(tmp_path / "d.zarr"), horizon=4,  # noqa: E731
                                         pad_before=1, pad_after=1, n_obs_steps=2,
                                         base_action="velocity")
    if tag is None:
        with pytest.raises(ValueError, match="refused: no gripper_tool"):
            make()
    else:
        assert make().gripper_tool == tag


def test_normalizer_floors_barely_moving_values():
    """The shared stats floor: a value the demos barely moved normalizes to ~0 (not to its
    noise stretched over [-1, 1]); a moving one keeps its own range."""
    from stretch4_policy_recorder.common.stats_floor import STATS_FLOOR
    rng = np.random.default_rng(0)
    lift = rng.uniform(0.2, 0.9, 500)                     # moves
    grip = 40.0 + rng.normal(0, 0.05, 500)                # noise only, std above 1e-3
    data = np.stack([lift, grip], axis=1)
    norm = S.make_floored_range_normalizer(data, ["lift", "grip_mm"], pct=0.0, verbose=False)
    out = norm.normalize(data).detach().numpy()
    assert out[:, 0].min() == pytest.approx(-1, abs=1e-4) and out[:, 0].max() == pytest.approx(1, abs=1e-4)
    assert np.abs(out[:, 1]).max() < 0.5 / STATS_FLOOR["grip_mm"]   # ~0, not stretched to +-1
    assert np.abs(out).max() <= 1 + 1e-4                            # every action still fits
