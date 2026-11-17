import unittest
import torch
from charger.model import VisionDiffusion


class SamplingTests(unittest.TestCase):
    def test_oracle_noise_recovers_known_clean_action(self):
        model = VisionDiffusion(2, horizon=2, diffusion_steps=8)
        model.encode = lambda rgb, state: torch.zeros(len(rgb), 256)
        def oracle(x, t, condition):
            a = model.alpha_bar[t, None, None]
            return (x-a.sqrt()*.25)/(1-a).sqrt()
        model.noise_prediction = oracle
        image = torch.zeros(2, 3, 8, 8, dtype=torch.uint8)
        state = torch.empty(2, 0)
        for steps in [None, 4]:
            result = model.sample(image, state, steps)
            self.assertTrue(torch.allclose(result, torch.full_like(result, .25), atol=1e-5))


if __name__ == '__main__':
    unittest.main()
