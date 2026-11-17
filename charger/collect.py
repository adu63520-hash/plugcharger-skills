"""Official ManiSkill solver + official RecordEpisode, with stage sidecar.

Failures are first-class data
-----------------------------
The official motion planner succeeds on roughly 57% of attempts on PlugCharger-v1 (measured:
569/1000 in data/formal_rgb), and the failures are concentrated in specific skills. Dropping
them leaves nothing to analyse, so this collector records, for every attempt:

  * the trajectory itself (unless --drop-failures), and
  * which skill failed, as `failed_stage`, derived from the stage boundaries that could still
    be computed. A failure after A+B means C failed, etc.

Successful attempts are listed in `episodes`; failed ones in `attempts` with `success=False`,
`failed_stage` and the error. Both point at a real trajectory when it was saved.
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.plug_charger import solve
from mani_skill.utils.wrappers.record import RecordEpisode

from .stages import measurements, boundaries, classify_failure


class Labels(gym.Wrapper):
    """Record stage measurements after every physics step."""

    def reset(self, **kwargs):
        self.labels = []
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(action)
        self.labels.append(measurements(self.env))
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/charger')
    p.add_argument('--seeds', type=int, default=100)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--start-seed', type=int, default=0)
    p.add_argument('--render-backend', default='gpu', choices=['gpu', 'cpu', 'none'])
    p.add_argument('--obs-mode', default='rgb', choices=['rgb', 'none'])
    p.add_argument('--drop-failures', action='store_true',
                   help='Do not save failed trajectories (default: save them).')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'trajectory.h5').exists():
        raise FileExistsError('Use a fresh output directory to preserve existing data')
    base = gym.make('PlugCharger-v1', obs_mode=args.obs_mode, control_mode='pd_joint_pos',
                    sim_backend='cpu', render_backend=args.render_backend,
                    reward_mode='sparse', max_episode_steps=400)
    labeled = Labels(base)
    env = RecordEpisode(labeled, output_dir=str(out), trajectory_name='trajectory',
                        save_video=False, save_on_reset=False,
                        source_type='motionplanning',
                        source_desc='Official PlugCharger solver with stage labels')
    manifest = {'env_id': 'PlugCharger-v1', 'control_mode': 'pd_joint_pos',
                'sim_backend': 'cpu', 'render_backend': args.render_backend,
                'obs_mode': args.obs_mode,
                'episodes': [], 'attempts': [],
                'failures_saved': not args.drop_failures,
                # B ends near the point 5 cm in front of the socket. The planner moves
                # continuously, so 1 cm is the window that captures "arrived at 5 cm";
                # measured expert stops sit 6.6-10.8 mm from that point.
                'b_position_tolerance_m': 0.01, 'b_angle_tolerance_degrees': 10}
    counts = dict(A=0, B=0, C=0, success=0)
    saved_failures = 0
    try:
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            for repeat in range(args.repeats):
                record = dict(seed=seed, repeat=repeat)
                try:
                    solve(env, seed=seed, debug=False, vis=False)
                except Exception as exc:
                    record['error'] = repr(exc)
                failed, partial = classify_failure(labeled.labels)
                success = failed is None
                record.update(success=success, boundaries=partial)
                if success:
                    counts['success'] += 1
                    record['trajectory'] = f'traj_{len(manifest["episodes"])}'
                    manifest['episodes'].append(record)
                else:
                    counts[failed] += 1
                    record['failed_stage'] = failed
                    if not args.drop_failures:
                        # Kept trajectories are distinguishable from successes by prefix.
                        record['trajectory'] = f'fail_{saved_failures}'
                        saved_failures += 1
                manifest['attempts'].append(record)
                env.flush_trajectory(save=success or not args.drop_failures)
                (out / 'stages.json').write_text(json.dumps(manifest, indent=2))
                print(record, flush=True)
    finally:
        env.close()
    manifest['failure_counts'] = counts
    (out / 'stages.json').write_text(json.dumps(manifest, indent=2))
    print('summary', counts, flush=True)
    if not manifest['episodes']:
        raise RuntimeError('No usable demonstrations collected')


if __name__ == '__main__':
    main()
