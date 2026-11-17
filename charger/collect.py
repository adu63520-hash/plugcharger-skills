"""Official ManiSkill solver + official RecordEpisode, with stage sidecar."""
import argparse
import json
from pathlib import Path
import gymnasium as gym
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions.plug_charger import solve
from mani_skill.utils.wrappers.record import RecordEpisode
from .stages import measurements, boundaries


class Labels(gym.Wrapper):
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
                'b_position_tolerance_m': 0.01, 'b_angle_tolerance_degrees': 10}
    try:
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            for repeat in range(args.repeats):
                record = dict(seed=seed, repeat=repeat)
                try:
                    solve(env, seed=seed, debug=False, vis=False)
                    cuts = boundaries(labeled.labels)
                    record.update(success=True, boundaries=cuts,
                                  trajectory=f'traj_{len(manifest["episodes"])}')
                except Exception as exc:
                    record.update(success=False, error=repr(exc))
                env.flush_trajectory(save=record['success'])
                manifest['attempts'].append(record)
                if record['success']:
                    manifest['episodes'].append(record)
                (out / 'stages.json').write_text(json.dumps(manifest, indent=2))
                print(record, flush=True)
    finally:
        env.close()
    if not manifest['episodes']:
        raise RuntimeError('No usable demonstrations collected')


if __name__ == '__main__':
    main()
