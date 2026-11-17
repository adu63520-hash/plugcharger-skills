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


def classify_failure(labels):
    """Return (failed_stage, partial_boundaries) for an episode that did not finish.

    `boundaries` stops at the first skill that never completes, so the number of skills it
    recovered says exactly where the attempt failed: 0 -> A, 1 -> B, 2 -> C, None -> success.
    Kept here rather than in `collect.py` so it can be tested without mplib installed.
    """
    partial = {}
    start = 0
    for stage in 'ABC':
        end = next((i + 1 for i in range(start, len(labels)) if labels[i][stage]), None)
        if end is None:
            return stage, partial
        partial[stage] = [start, end]
        start = end
    return None, partial
