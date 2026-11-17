"""Privileged measurements for labels/termination only; never policy inputs."""
import math
import torch


def measurements(env, position_tolerance=0.01):
    e = env.unwrapped
    # Official planner: goal_pose * Pose([-0.05, 0, 0]).
    from mani_skill.utils.structs.pose import Pose
    target = e.goal_pose * Pose.create_from_pq(
        p=torch.tensor([[-0.05, 0., 0.]], device=e.device))
    distance = torch.linalg.vector_norm(e.charger.pose.p - target.p, dim=-1)
    q = e.charger.pose.q
    g = e.goal_pose.q
    angle = 2 * torch.acos((q * g).sum(-1).abs().clamp(0, 1))
    grasped = e.agent.is_grasping(e.charger)
    aligned = grasped & (distance <= position_tolerance) & (angle <= math.radians(10))
    inserted = e.evaluate()['success']
    return dict(A=bool(grasped.item()), B=bool(aligned.item()),
                C=bool(inserted.item()), distance=float(distance.item()),
                angle=float(angle.item()))


def boundaries(labels):
    """Post-action labels -> observation indices; preserve boundary action once."""
    start = 0
    result = {}
    for stage in 'ABC':
        end = next((i + 1 for i in range(start, len(labels)) if labels[i][stage]), None)
        if end is None:
            raise ValueError(f'Missing stage {stage} after observation {start}')
        result[stage] = [start, end]
        start = end
    return result
