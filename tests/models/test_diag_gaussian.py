# Unit test suit
# run with python -m unittest <file.py>

import unittest
import torch

# custom modules
from drone_explorer.models.base.distributions import DiagGaussianPolicy


class TestStringMethods(unittest.TestCase):

    def test_fixed_std_has_no_parameters(self):
        dist = DiagGaussianPolicy(action_dim=3, learned=False)
        params = dict(dist.named_parameters())
        self.assertNotIn("log_std", params, "msg: value not in params")

    def test_learned_std_has_parameter(self):
        dist = DiagGaussianPolicy(action_dim=3, learned=True)
        params = dict(dist.named_parameters())
        self.assertIn("log_std", params, "msg: value in params")

    def test_single_mean_shape(self):
        action_dim = 4
        dist_fn = DiagGaussianPolicy(action_dim, learned=False)
        mean = torch.zeros(action_dim)
        dist = dist_fn(mean)

        self.assertEqual(dist.mean.shape, (action_dim,),
                        "msg: value doesn't equal mean dim shape")
        self.assertEqual(dist.covariance_matrix.shape, (action_dim, action_dim),
                        "msg: value doesn't equal mean dim matrix shape")
    
    def test_batched_mean_shape(self):
        batch_size = 8
        action_dim = 3
        dist_fn = DiagGaussianPolicy(action_dim, learned=False)

        mean = torch.zeros(batch_size, action_dim)
        dist = dist_fn(mean)

        self.assertEqual(dist.mean.shape, (batch_size, action_dim),
                         "msg: value doesn't equal mean dim shape")
        self.assertEqual(dist.covariance_matrix.shape, (batch_size, action_dim, action_dim),
                         "msg: value doesn't equal action dim matrix shape")

    def test_log_std_receives_gradients(self):
        action_dim = 3
        dist_fn = DiagGaussianPolicy(action_dim, learned=True)

        mean = torch.zeros(action_dim, requires_grad=True)
        dist = dist_fn(mean)
        loss = -dist.log_prob(dist.sample())
        loss.backward()

        self.assertIsNotNone(dist_fn.log_std.grad,
                             "msg: log_std value is not none")

    # Check for numerical stability (no NaNs)
    def test_no_nan_log_prob(self):
        dist_fn = DiagGaussianPolicy(2, init_std=1e-3)
        mean = torch.zeros(2)
        dist = dist_fn(mean)

        log_prob = dist.log_prob(dist.sample())
        self.assertTrue(torch.isfinite(log_prob).all(),
                        "msg: check for no NaNs (numerical stability)")


if __name__ == '__main__':
    unittest.main()
