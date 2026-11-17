import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from charger.residual import ActorCritic, ResidualPolicy
from charger.finetune import advantages
from charger.foresight import binomial_nll


class ResidualTests(unittest.TestCase):
    def test_binomial_loss_respects_trial_counts_and_is_stable(self):
        loss = binomial_nll(torch.zeros(2), torch.tensor([0., 1.]), torch.tensor([1., 5.]))
        self.assertAlmostEqual(loss.item(), 3*np.log(2), places=5)
        extreme = binomial_nll(torch.tensor([-1000., 1000.]), torch.tensor([0., 5.]), torch.tensor([5., 5.]))
        self.assertTrue(torch.isfinite(extreme))
    def test_zero_actor_reproduces_base_action(self):
        network = ActorCritic(258, 2)
        base = SimpleNamespace(device='cpu', camera='hand_camera', proprio=False,
                               low=torch.full((2,), -1.), high=torch.ones(2),
                               model=SimpleNamespace(encode=lambda rgb, state: torch.zeros(len(rgb), 256)))
        obs = {'sensor_data': {'hand_camera': {'rgb': np.zeros((1, 16, 16, 3), np.uint8)}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'residual.pt'
            torch.save(dict(network=network.state_dict(), network_config=network.config,
                            alpha=.1, action_low=-torch.ones(2), action_high=torch.ones(2)), path)
            policy = ResidualPolicy(base, path)
            original = torch.tensor([.25, -.5])
            self.assertTrue(torch.equal(original, policy.correct(obs, original)))

    def test_terminal_reset_blocks_future_reward_in_gae(self):
        reward = torch.tensor([1., 0., 0., 1.])
        done = torch.tensor([1., 0., 0., 1.])
        values = torch.zeros(4)
        adv, returns = advantages(reward, done, values, torch.tensor(999.), gamma=1., lam=1.)
        self.assertTrue(torch.equal(adv, torch.ones(4)))
        self.assertTrue(torch.equal(returns, adv))


if __name__ == '__main__':
    unittest.main()
