"""Recorded trajectories are per-episode; this file explains why there is no per-stage dataset.

`collect.py` runs the official motion planner once per attempt and records the whole episode
(reset -> grasp -> align -> insert) as one HDF5 trajectory. `charger.data.SkillDataset` slices
that trajectory by the stage boundaries in `stages.json`, so training a skill does not require
a per-stage copy of the data on disk.

For provenance, `charger.replay_rgb` (now in `archive/`) produced the RGB dataset
`data/formal_rgb` on this machine with the manifest fields below:

  obs_mode = rgb
  render_backend = cpu
  state_replay_source = <path of the state-only dataset that recorded actions/labels>

That intermediate stage was needed only because RGB was rendered separately from physics on
this Windows machine. It is NOT part of the current reproducible pipeline: `collect.py` records
RGB and labels in the same pass. Regenerate the dataset with `collect.py` (needs mplib), as
described in README.md, rather than replaying states.
"""
