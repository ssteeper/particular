"""gpu_sim.simulate_batch against particular.simulate. Run: python -m unittest test_gpu_sim

Over 200 steps the GPU must stay as close to numpy as numpy stays to itself when every starting
coordinate moves by one ulp: rounding differences grow chaotically, a physics mismatch doesn't
stay that small. Skipped without torch and a CUDA GPU.
"""
import importlib.util
import unittest

import numpy as np

import particular as P


def _cuda():
    if importlib.util.find_spec("torch") is None:
        return False
    import torch
    return torch.cuda.is_available()


@unittest.skipUnless(_cuda(), "needs torch with a CUDA GPU")
class Parity(unittest.TestCase):
    def test_scenes(self):
        import gpu_sim
        # every scene, classic and ecosystem, with run lengths that aren't all multiples of log_every
        setups = [{**f(), "steps": 200 + 5 * (k % 3)} for k, f in enumerate(P.SCENES.values())]
        got = gpu_sim.simulate_batch(setups, batch=7)       # several launch groups, sizes mixed
        for name, s, g in zip(P.SCENES, setups, got):
            args = dict(steps=s["steps"], species=s["species"], rules=s["rules"])
            ref = P.simulate(s["pos"], s["vel"], **args)
            floor = np.abs(P.simulate(np.nextafter(s["pos"], np.inf), s["vel"], **args) - ref).max()
            self.assertEqual(g.shape, ref.shape, name)
            self.assertLessEqual(np.abs(g - ref).max(), 10 * floor + 1e-12, name)


if __name__ == "__main__":
    unittest.main()
